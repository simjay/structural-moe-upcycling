"""Fine-tune an upcycled MoE model on OpenMathReasoning (cot split) with LoRA.

Loads a model saved by ``src/upcycle.py``, applies LoRA to attention and
shared expert projections, and trains with SFTTrainer. Logs to wandb for
comparing convergence across initialization methods.

Example:
    .. code-block:: bash

        python -m src.train --model /tmp/moe-direct --run-name direct
"""

import argparse
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

DATASET = "nvidia/OpenMathReasoning"
DATASET_SPLIT = "cot"


def format_sample(sample):
    """Format a dataset sample as a single training string.

    Args:
        sample: A dict with ``"problem"`` and ``"generated_solution"`` keys.

    Returns:
        Dict with a ``"text"`` key containing the formatted string.
    """
    return {"text": f"Problem: {sample['problem']}\n\nSolution: {sample['generated_solution']}"}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune upcycled MoE model")
    parser.add_argument("--model", required=True, help="Path to upcycled model")
    parser.add_argument("--run-name", default="moe-train", help="wandb run name")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", default="/tmp/moe-checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print(f"=== Training {args.run_name} ===\n")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "shared_expert.gate_proj", "shared_expert.up_proj",
            "shared_expert.down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    ds = load_dataset(DATASET, split=DATASET_SPLIT, streaming=True)
    ds = ds.map(format_sample)

    report_to = "none" if args.no_wandb else "wandb"

    training_args = SFTConfig(
        output_dir=f"{args.output}/{args.run_name}",
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
    )

    print(f"\nTraining for {args.max_steps} steps...")
    result = trainer.train()

    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps completed: {result.global_step}")

    print(f"Saving final model to {args.output}/{args.run_name}/final...")
    trainer.save_model(f"{args.output}/{args.run_name}/final")

    print("Done.")


if __name__ == "__main__":
    main()
