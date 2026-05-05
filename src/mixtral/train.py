"""Fine-tune an upcycled Mixtral model on OpenMathReasoning (cot split).

Loads a model saved by ``src.mixtral.upcycle`` and trains only:
- Expert FFN weights (gate_up_proj, down_proj) — full fine-tuning
- Router (mlp.gate) — full fine-tuning

Attention projections are frozen to isolate the effect of expert initialization.
Logs training loss and expert weight divergence to wandb for comparing
convergence across initialization methods.

Example:
    .. code-block:: bash

        python -m src.mixtral.train --model /tmp/mixtral-direct --run-name direct
"""

import argparse
import json
import os

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
    def _compute_metrics(self):
        """Compute cosine similarity and L2 divergence between all expert pairs.

        For Mixtral, all 8 experts are full copies of the same dense FFN,
        so all pairwise comparisons are valid.
        """
        cosine_sims = []
        l2_dists = []
        for name, param in self._model.named_parameters():
            if "experts" in name and "gate_up_proj" in name and param.dim() == 3:
                n_experts = param.shape[0]
                flat = param.view(n_experts, -1).float()
                norms = flat.norm(dim=1, keepdim=True)
                normed = flat / norms.clamp(min=1e-8)
                mean_norm = norms.mean().item()
                n = n_experts
                mask = torch.triu(torch.ones(n, n, device=flat.device), diagonal=1).bool()
                sim_matrix = normed @ normed.T
                cosine_sims.append(sim_matrix[mask].mean().item())
                dists = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
                l2_dists.append((dists[mask].mean().item() / mean_norm))
        cos = sum(cosine_sims) / len(cosine_sims) if cosine_sims else None
        l2 = sum(l2_dists) / len(l2_dists) if l2_dists else None
        return cos, l2

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb
        cos, l2 = self._compute_metrics()
        if cos is not None and wandb.run is not None:
            wandb.log({
                "expert/cosine_similarity": cos,
                "expert/l2_divergence": l2,
            }, step=state.global_step, commit=False)


def format_sample(sample):
    """Format a dataset sample as a single training string."""
    return {"text": f"Problem: {sample['problem']}\n\nSolution: {sample['generated_solution']}"}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune upcycled Mixtral model")
    parser.add_argument("--model", required=True, help="Path to upcycled model")
    parser.add_argument("--run-name", default="mixtral-train", help="wandb run name")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/tmp/mixtral-checkpoints")
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

    print("Freezing all, then unfreezing experts + router...")
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "experts" in name or "mlp.gate" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.2f}")

    divergence_callback = ExpertDivergenceCallback(model)

    print("Loading dataset...")
    ds = load_dataset(DATASET, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
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
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=50,
        seed=args.seed,
        optim="paged_adamw_8bit",
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

    step0_cos, step0_l2 = divergence_callback._compute_metrics()
    import wandb
    if wandb.run is not None:
        wandb.log({
            "expert/cosine_similarity": step0_cos,
            "expert/l2_divergence": step0_l2,
        }, step=0)
    print(f"step-0 cosine_similarity = {step0_cos:.6f}, l2_divergence = {step0_l2:.6f}")

    print(f"\nTraining for {args.max_steps} steps...")
    result = trainer.train()

    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps completed: {result.global_step}")

    print("\n=== GSM8K Evaluation ===")
    from src.eval.gsm8k import evaluate
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Evaluating on {len(gsm8k_ds)} problems (full test set)...")
    model.eval()
    accuracy, correct, total = evaluate(model, tokenizer, gsm8k_ds)
    print(f"GSM8K Accuracy: {correct}/{total} = {100*accuracy:.1f}%")

    import wandb
    if wandb.run is not None:
        wandb.log({"gsm8k/accuracy": accuracy, "gsm8k/correct": correct, "gsm8k/total": total})

    results_dir = f"{args.output}/{args.run_name}"
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "run_name": args.run_name,
        "seed": args.seed,
        "gsm8k_accuracy": accuracy,
        "gsm8k_correct": correct,
        "gsm8k_total": total,
        "training_loss": result.training_loss,
        "steps": result.global_step,
    }
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print(f"Saving final model to {results_dir}/final...")
    trainer.save_model(f"{results_dir}/final")

    print("Done.")


if __name__ == "__main__":
    main()
