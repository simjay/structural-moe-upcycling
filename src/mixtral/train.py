"""Fine-tune an upcycled Mixtral model on OpenMathReasoning (cot split).

Loads a model saved by ``src.mixtral.upcycle`` via Unsloth and applies:
- LoRA to attention projections (q/k/v/o_proj)
- LoRA to fused expert FFN parameters (gate_up_proj, down_proj) via
  ``target_parameters`` (requires PEFT >= 0.17)
- Full training of the router (``mlp.gate``) via ``modules_to_save``

Uses Unsloth for optimized gradient checkpointing and fused kernels (~2x
faster, ~30% less VRAM). Logs to wandb for comparing convergence across
initialization methods (direct / gaussian / svd).

Example:
    .. code-block:: bash

        python -m src.mixtral.train --model /tmp/mixtral-direct --run-name direct
"""

import argparse

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
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
    parser = argparse.ArgumentParser(description="Fine-tune upcycled Mixtral model")
    parser.add_argument("--model", required=True, help="Path to upcycled model")
    parser.add_argument("--run-name", default="mixtral-train", help="wandb run name")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", default="/tmp/mixtral-checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print(f"=== Training {args.run_name} ===\n")

    print("Loading model via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq_len,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA (attention + experts) and unfreezing router...")
    num_experts = 8
    effective_r = max(1, args.lora_r // num_experts)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=[
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ],
        rank_pattern={
            "experts.gate_up_proj": effective_r,
            "experts.down_proj": effective_r,
        },
        modules_to_save=["mlp.gate"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
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
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        dataset_text_field="text",
        max_length=args.seq_len,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
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
