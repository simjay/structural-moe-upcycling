"""Fine-tune an upcycled Qwen1.5-MoE model on OpenMathReasoning (cot split).

Loads a model saved by ``src.qwen15.upcycle`` and trains only:
- Routing expert FFN weights (gate_up_proj, down_proj) — full fine-tuning
- Router (mlp.gate) — full fine-tuning

Attention projections and the shared expert are frozen to isolate the effect
of routing expert initialization. Logs training loss and expert weight
divergence to wandb for comparing convergence across initialization methods.

Example:
    .. code-block:: bash

        python -m src.qwen15.train --model /tmp/moe-direct --run-name direct
"""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DATASET = "nvidia/OpenMathReasoning"
DATASET_SPLIT = "cot"


class ExpertDivergenceCallback(TrainerCallback):
    """Log mean pairwise cosine similarity between experts to wandb.

    At each logging step, computes the average cosine similarity between
    all pairs of expert weight vectors across all layers. Lower values
    indicate greater expert specialization/divergence.
    """

    def __init__(self, model):
        self._model = model

    @torch.no_grad()
    def _compute_divergence(self):
        similarities = []
        for name, param in self._model.named_parameters():
            if "experts.gate_up_proj" in name and param.dim() == 3:
                n_experts = param.shape[0]
                flat = param.view(n_experts, -1).float()
                normed = F.normalize(flat, dim=1)
                sim_matrix = normed @ normed.T
                mask = torch.triu(torch.ones(n_experts, n_experts, device=sim_matrix.device), diagonal=1).bool()
                similarities.append(sim_matrix[mask].mean().item())
        if similarities:
            return sum(similarities) / len(similarities)
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            sim = self._compute_divergence()
            if sim is not None:
                logs["expert/mean_cosine_similarity"] = sim


def format_sample(sample):
    """Format a dataset sample as a single training string."""
    return {"text": f"Problem: {sample['problem']}\n\nSolution: {sample['generated_solution']}"}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune upcycled MoE model")
    parser.add_argument("--model", required=True, help="Path to upcycled model")
    parser.add_argument("--run-name", default="moe-train", help="wandb run name")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", default="/tmp/moe-checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print(f"=== Training {args.run_name} ===\n")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Freezing all, then unfreezing routing experts + router...")
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "experts" in name:
            param.requires_grad = True
        if "mlp.gate" in name or "shared_expert_gate" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.2f}")

    divergence_callback = ExpertDivergenceCallback(model)

    print("Loading dataset...")
    ds = load_dataset(DATASET, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    ds = ds.map(format_sample)

    print("Loading eval dataset (GSM8K train split)...")
    eval_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds = eval_ds.map(lambda s: {"text": f"Problem: {s['question']}\n\nSolution: {s['answer']}"})

    report_to = "none" if args.no_wandb else "wandb"

    training_args = SFTConfig(
        output_dir=f"{args.output}/{args.run_name}",
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=50,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to=report_to,
        dataset_text_field="text",
        max_length=args.seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=[divergence_callback],
    )

    print(f"\nTraining for {args.max_steps} steps...")
    result = trainer.train()


    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps completed: {result.global_step}")

    print(f"Saving final model to {args.output}/{args.run_name}/final...")
    trainer.save_model(f"{args.output}/{args.run_name}/final")

    print("\n=== GSM8K Evaluation ===")
    from src.eval.gsm8k import evaluate
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_ds = gsm8k_ds.select(range(min(200, len(gsm8k_ds))))
    print(f"Evaluating on {len(gsm8k_ds)} problems...")
    model.eval()
    accuracy, correct, total = evaluate(model, tokenizer, gsm8k_ds)
    print(f"GSM8K Accuracy: {correct}/{total} = {100*accuracy:.1f}%")

    import wandb
    if wandb.run is not None:
        wandb.run.summary["gsm8k/accuracy"] = accuracy
        wandb.run.summary["gsm8k/correct"] = correct
        wandb.run.summary["gsm8k/total"] = total

    print("Done.")


if __name__ == "__main__":
    main()
