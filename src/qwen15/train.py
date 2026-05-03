"""Fine-tune an upcycled Qwen1.5-MoE model on OpenMathReasoning (cot split).

Loads a model saved by ``src.qwen15.upcycle`` and trains only:
- Routing expert FFN weights (gate_up_proj, down_proj) — full fine-tuning
- Router (mlp.gate) — full fine-tuning

Attention projections and the shared expert are frozen to isolate the effect
of routing expert initialization. Logs training loss and router entropy to
wandb for comparing convergence across initialization methods.

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


class RouterEntropyCallback(TrainerCallback):
    """Log mean router entropy to wandb during training.

    Registers forward hooks on all router (mlp.gate) modules to capture
    router logits and compute the entropy of the softmax distribution.
    Higher entropy = more uniform expert usage; lower = router collapse.
    """

    def __init__(self, model):
        self._entropy_values = []
        self._hooks = []
        for name, module in model.named_modules():
            if name.endswith("mlp.gate") and hasattr(module, "weight"):
                hook = module.register_forward_hook(self._hook_fn)
                self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
        self._entropy_values.append(entropy.item())

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._entropy_values and logs is not None:
            mean_entropy = sum(self._entropy_values) / len(self._entropy_values)
            logs["router/mean_entropy"] = mean_entropy
            self._entropy_values.clear()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


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
        attn_implementation="flash_attention_2",
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

    entropy_callback = RouterEntropyCallback(model)

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
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=training_args,
        callbacks=[entropy_callback],
    )

    print(f"\nTraining for {args.max_steps} steps...")
    result = trainer.train()

    entropy_callback.remove_hooks()

    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps completed: {result.global_step}")

    print(f"Saving final model to {args.output}/{args.run_name}/final...")
    trainer.save_model(f"{args.output}/{args.run_name}/final")

    print("\n=== GSM8K Evaluation ===")
    from src.eval.gsm8k import evaluate, extract_answer, extract_ground_truth
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_ds = gsm8k_ds.select(range(min(200, len(gsm8k_ds))))
    print(f"Evaluating on {len(gsm8k_ds)} problems...")
    model.eval()
    accuracy, correct, total = evaluate(model, tokenizer, gsm8k_ds)
    print(f"GSM8K Accuracy: {correct}/{total} = {100*accuracy:.1f}%")
    trainer.log({"gsm8k/accuracy": accuracy, "gsm8k/correct": correct, "gsm8k/total": total})

    print("Done.")


if __name__ == "__main__":
    main()
