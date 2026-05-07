"""Evaluate an MoE model on GSM8K (grade-school math).

Loads a saved checkpoint, generates solutions to GSM8K test problems using
greedy decoding, extracts the final numeric answer, and compares to ground
truth. Reports accuracy as a percentage.

Example:
    .. code-block:: bash

        python -m src.eval.gsm8k --model /tmp/qwen-moe-direct/final
        python -m src.eval.gsm8k --model /tmp/qwen-moe-direct/checkpoint-50 --max-samples 100
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def extract_answer(text):
    """Extract the final numeric answer from a GSM8K-style solution.

    Looks for the pattern '#### <number>' which is the standard GSM8K
    answer format. Falls back to extracting the last number in the text.

    Returns:
        The extracted number as a string, or None if no number found.
    """
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_ground_truth(answer_text):
    """Extract the numeric answer from a GSM8K ground truth string."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


FEW_SHOT_EXAMPLES = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell in total in April and May?",
     "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
     "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
     "Maila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nShe wants to read 84 / 2 = <<84/2=42>>42 pages tomorrow.\n#### 42"),
]


def build_prompt(question):
    """Build a few-shot prompt with 4 exemplars followed by the target question."""
    prompt = ""
    for q, a in FEW_SHOT_EXAMPLES:
        prompt += f"Problem: {q}\n\nSolution: {a}\n\n"
    prompt += f"Problem: {question}\n\nSolution:"
    return prompt


def evaluate(model, tokenizer, dataset, max_new_tokens=512):
    """Run greedy generation on each problem and score accuracy.

    Args:
        model: The causal LM to evaluate.
        tokenizer: Tokenizer for the model.
        dataset: List of dicts with 'question' and 'answer' keys.
        max_new_tokens: Maximum tokens to generate per problem.

    Returns:
        Tuple of (accuracy float, total correct, total attempted).
    """
    correct = 0
    total = 0

    for i, sample in enumerate(dataset):
        question = sample["question"]
        ground_truth = extract_ground_truth(sample["answer"])
        if ground_truth is None:
            continue

        prompt = build_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        predicted = extract_answer(generated)

        if predicted is not None and predicted == ground_truth:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(dataset)}] accuracy so far: {correct}/{total} ({100*correct/total:.1f}%)")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model on GSM8K")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max problems to evaluate (default 200 for speed)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per problem")
    args = parser.parse_args()

    print(f"=== GSM8K Evaluation ===\n")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
    print(f"Evaluating on {len(ds)} problems...\n")

    accuracy, correct, total = evaluate(model, tokenizer, ds, args.max_new_tokens)

    print(f"\n{'='*40}")
    print(f"GSM8K Accuracy: {correct}/{total} = {100*accuracy:.1f}%")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
