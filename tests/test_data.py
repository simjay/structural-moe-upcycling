"""Sanity check: load and inspect the OpenMathReasoning dataset."""

from datasets import load_dataset

DATASET = "nvidia/OpenMathReasoning"

print(f"Loading {DATASET} (streaming)...")
ds = load_dataset(DATASET, split="train", streaming=True)

print("\nFirst 3 samples:\n")
for i, sample in enumerate(ds):
    if i >= 3:
        break
    print(f"--- Sample {i} ---")
    print(f"Keys: {list(sample.keys())}")
    for k, v in sample.items():
        text = str(v)
        print(f"  {k}: {text[:200]}{'...' if len(text) > 200 else ''}")
    print()

print("Dataset loading test passed!")
