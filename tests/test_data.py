"""Sanity check: load and inspect the OpenMathReasoning dataset."""

import sys
from datasets import load_dataset

DATASET = "nvidia/OpenMathReasoning"

print(f"Loading {DATASET} (streaming)...")
ds = load_dataset(DATASET, split="cot", streaming=True)

print("\nFirst 3 samples:\n")
it = iter(ds)
for i in range(3):
    sample = next(it)
    print(f"--- Sample {i} ---")
    print(f"Keys: {list(sample.keys())}")
    for k, v in sample.items():
        text = str(v)
        print(f"  {k}: {text[:200]}{'...' if len(text) > 200 else ''}")
    print()
del it, ds

print("Dataset loading test passed!")
sys.exit(0)
