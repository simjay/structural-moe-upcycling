"""Analyze initialization quality: measure preservation and expert diversity.

Upcycles Qwen1.5-1.8B into MoE with the specified initialization method,
then measures eval/loss on the GSM8K train split and pairwise expert cosine
similarity. No training is performed. Results are saved to a JSON file.

Example:
    .. code-block:: bash

        python -m src.qwen15.init_analysis --method svd --k 128 --svd-scale 0.5 \
            --output results/qwen15/step0/svd-k128-s0.5.json
"""

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from src.qwen15.upcycle import upcycle


def compute_cosine_similarity(model):
    """Mean pairwise cosine similarity between experts across all layers."""
    similarities = []
    for name, param in model.named_parameters():
        if "experts.gate_up_proj" in name and param.dim() == 3:
            n_experts = param.shape[0]
            flat = param.view(n_experts, -1).float()
            normed = F.normalize(flat, dim=1)
            sim_matrix = normed @ normed.T
            mask = torch.triu(
                torch.ones(n_experts, n_experts, device=sim_matrix.device),
                diagonal=1,
            ).bool()
            similarities.append(sim_matrix[mask].mean().item())
    if similarities:
        return sum(similarities) / len(similarities)
    return None


def main():
    parser = argparse.ArgumentParser(description="Step-0 quality and diversity measurement")
    parser.add_argument("--method", required=True, choices=["direct", "gaussian", "svd"])
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--svd-scale", type=float, default=0.5)
    parser.add_argument("--output", required=True, help="Path for output JSON file")
    parser.add_argument("--model-dir", default="/tmp/step0-moe",
                        help="Temp directory for the upcycled model")
    args = parser.parse_args()

    print(f"=== Step-0 Evaluation: {args.method} ===\n")

    # 1. Upcycle
    print("[1/4] Upcycling...")
    upcycle(args.method, args.model_dir, sigma=args.sigma, k=args.k,
            svd_scale=args.svd_scale)

    # 2. Load model onto GPU
    print("\n[2/4] Loading upcycled model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Compute expert cosine similarity
    print("[3/4] Computing expert cosine similarity...")
    cosine_sim = compute_cosine_similarity(model)
    print(f"  cosine_similarity = {cosine_sim:.6f}")

    # 4. Compute eval/loss via Trainer.evaluate()
    print("[4/4] Computing eval/loss on GSM8K train split...")
    eval_ds = load_dataset("openai/gsm8k", "main", split="train")
    eval_ds = eval_ds.map(
        lambda s: {"text": f"Problem: {s['question']}\n\nSolution: {s['answer']}"}
    )

    config = SFTConfig(
        output_dir="/tmp/step0-trainer",
        per_device_eval_batch_size=4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataset_text_field="text",
        max_length=2048,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        eval_dataset=eval_ds,
    )
    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    print(f"  eval_loss = {eval_loss:.6f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "method": args.method,
        "k": args.k if args.method == "svd" else None,
        "svd_scale": args.svd_scale if args.method == "svd" else None,
        "sigma": args.sigma if args.method == "gaussian" else None,
        "eval_loss": eval_loss,
        "cosine_similarity": cosine_sim,
        "diversity": 1.0 - cosine_sim if cosine_sim is not None else None,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(json.dumps(results, indent=2))

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    shutil.rmtree(args.model_dir, ignore_errors=True)
    shutil.rmtree("/tmp/step0-trainer", ignore_errors=True)
    print("Cleanup done.")


if __name__ == "__main__":
    main()
