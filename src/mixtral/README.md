# Mixtral Experiment: Mistral 7B → Mixtral 8x7B

Large-scale upcycling experiment with a perfectly dimension-matched dense/MoE pair.

## Why Mixtral?

We originally considered Qwen3.5 model pairs (e.g. Qwen3.5-9B → Qwen3.5-35B-A3B),
but these have fundamental architectural mismatches — different hidden sizes,
layer counts, and attention head counts — making direct upcycling impossible.

Mistral 7B and Mixtral 8x7B share **every dimension** perfectly, making this
the cleanest available large-scale upcycling target.

## Architecture

| Component | Dense (Mistral 7B) | MoE (Mixtral 8x7B) | Match |
| --- | --- | --- | --- |
| hidden_size | 4096 | 4096 | exact |
| layers | 32 | 32 | exact |
| attention heads | 32 | 32 | exact |
| KV heads | 8 | 8 | exact |
| vocab_size | 32000 | 32000 | exact |
| FFN intermediate | 14336 | 14336 | exact |
| num experts | 1 (dense) | 8, top-2 active | — |
| total params | ~7B | ~47B (13B active) | — |

## Key differences from the Qwen1.5 experiment

| Aspect | Qwen1.5 | Mixtral |
| --- | --- | --- |
| Shared expert | yes (5504 dim) | **none** |
| Expert FFN size | 1408 (smaller) | 14336 (same as dense) |
| Partition needed | yes (4 chunks, replicate 15x) | **no** (direct full copy) |
| Num experts | 60, top-4 | 8, top-2 |
| Dimension mismatch | shared expert override needed | **none** |
| Model scale | 1.8B → 14.3B | 7B → 47B |

## Expert initialization

Since each expert has the same FFN dimensions as the dense model, initialization
is simpler — no partitioning or padding needed.

1. **Direct copy** — all 8 experts receive identical copies of the dense FFN
2. **Gaussian perturbation** — direct copy + N(0, sigma^2) noise per expert
3. **SVD residual** — decompose dense FFN via SVD, share structural component (top-k singular values), perturb residual per expert

## Usage

```bash
# Upcycle (runs on CPU, ~5 min for direct, ~45 min for SVD)
python3 -m src.mixtral.upcycle --method direct   --output /tmp/mixtral-direct
python3 -m src.mixtral.upcycle --method gaussian  --sigma 0.01 --output /tmp/mixtral-gaussian
python3 -m src.mixtral.upcycle --method svd       --k 256 --output /tmp/mixtral-svd

# Train (multi-GPU)
python3 -m src.mixtral.train --model /tmp/mixtral-direct   --run-name mixtral-direct   --max-steps 2000
python3 -m src.mixtral.train --model /tmp/mixtral-gaussian  --run-name mixtral-gaussian --max-steps 2000
python3 -m src.mixtral.train --model /tmp/mixtral-svd       --run-name mixtral-svd      --max-steps 2000
```

## Hardware

4x H100-80GB node (Prime Intellect). The 47B-parameter Mixtral model requires
~94 GB in bf16, which is auto-sharded across the 4 GPUs via `device_map="auto"`.

Estimated training time: ~6-8 hours per run.
Estimated cost: ~$110 total for all three runs.
