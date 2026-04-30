"""Fine-tune an upcycled MoE model on OpenMathReasoning (cot split) with LoRA.

Loads a model saved by ``src.qwen15.upcycle``, applies LoRA to attention and
shared expert projections, and trains with SFTTrainer. Logs to wandb for
comparing convergence across initialization methods.

Example:
    .. code-block:: bash

        python -m src.qwen15.train --model /tmp/moe-direct --run-name direct
"""

from unsloth import FastLanguageModel
import argparse

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import gc
from transformers import TrainerCallback
import wandb

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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", default="/tmp/moe-checkpoints")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print(f"=== Training {args.run_name} ===\n")

    print("Loading model...")
    model_base, tokenizer = FastLanguageModel.from_pretrained(
      model_name = args.model,
      max_seq_length = args.seq_len,
      load_in_4bit = True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
      model_base,
      r = args.lora_r,
      target_modules = [
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"
      ],
      lora_alpha = args.lora_r,
      lora_dropout = 0,
      use_gradient_checkpointing = "unsloth", 
    )
    
    # Manually enable training for the router/gate modules
    for name, param in model.named_parameters():
        if any(x in name for x in ["mlp.gate", "shared_expert_gate"]):
            param.data = param.data.to(torch.bfloat16) 
            param.requires_grad = True
            # print(f"Full Fine-Tuning enabled for: {name} (Casted to BF16)")
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
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        save_steps=args.max_steps,
        optim = "adamw_8bit",
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        report_to=report_to,
        dataset_text_field="text",
        max_length=args.seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=training_args,
    )


    # for name, param in model.named_parameters():
    #     if param.requires_grad and ("gate" in name.lower() or "router" in name.lower()):
    #         print(f"DEBUG: Found trainable gate layer: {name}")
    print(f"\nTraining for {args.max_steps} steps...")
    result = trainer.train()

    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps completed: {result.global_step}")

    print("Saving training history to CSV...")
    history = trainer.state.log_history
    df_history = pd.DataFrame(history)
    
    csv_path = f"{args.output}/{args.run_name}/log_history.csv"
    df_history.to_csv(csv_path, index=False)
    print(f"History saved to {csv_path}")

    print(f"Saving final model to {args.output}/{args.run_name}/final...")
    trainer.save_model(f"{args.output}/{args.run_name}/final")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
