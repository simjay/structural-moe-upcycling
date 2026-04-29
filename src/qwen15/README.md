# Qwen1.5 Experiment: Qwen1.5-1.8B → Qwen1.5-MoE-A2.7B

Small-scale upcycling experiment on a perfectly matched dense/MoE pair.

## Architecture

| Component | Dense (Qwen1.5-1.8B) | MoE (Qwen1.5-MoE-A2.7B) | Match |
| --- | --- | --- | --- |
| hidden_size | 2048 | 2048 | exact |
| layers | 24 | 24 | exact |
| attention heads | 16 | 16 | exact |
| KV heads | 16 | 16 | exact |
| vocab_size | 151936 | 151936 | exact |
| FFN intermediate | 5504 | — | — |
| expert intermediate | — | 1408 | — |
| shared expert intermediate | — | 5504 (overridden) | — |
| num experts | — | 60, top-4 active | — |

All dimensions except the FFN match exactly. The shared expert intermediate size
is overridden from 5632 to 5504 so the dense FFN can be copied without padding.

## MoE block structure

```
x (2048) ───┬── shared expert (2048 → 5504 → 2048)  ← exact copy of dense FFN
            │
            ├── router (2048 → 60 scores) → pick top-4
            │     ├── expert i  (2048 → 1408 → 2048)
            │     ├── expert j  (2048 → 1408 → 2048)
            │     ├── expert k  (2048 → 1408 → 2048)
            │     └── expert l  (2048 → 1408 → 2048)
            │
            └── ADD all outputs → (2048)
```

## Expert initialization

The dense FFN (5504 rows) is partitioned into 4 non-overlapping row-slices
(three of 1408, one of 1280 zero-padded to 1408), then replicated 15x to
fill all 60 experts. The three init methods vary how these copies are
differentiated:

1. **Direct copy** — all copies identical
2. **Gaussian perturbation** — add N(0, sigma^2) noise per expert
3. **SVD residual** — decompose dense FFN via SVD, share structural component, perturb residual per expert

## Usage

```bash
# Upcycle
python3 -m src.qwen15.upcycle --method direct   --output /tmp/qwen-moe-direct
python3 -m src.qwen15.upcycle --method gaussian  --sigma 0.01 --output /tmp/qwen-moe-gaussian
python3 -m src.qwen15.upcycle --method svd       --k 256 --output /tmp/qwen-moe-svd

# Train
python3 -m src.qwen15.train --model /tmp/qwen-moe-direct   --run-name qwen-direct   --max-steps 500
python3 -m src.qwen15.train --model /tmp/qwen-moe-gaussian  --run-name qwen-gaussian --max-steps 500
python3 -m src.qwen15.train --model /tmp/qwen-moe-svd       --run-name qwen-svd      --max-steps 500
```

## Hardware

Single A100-80GB (or equivalent). Upcycling runs on CPU (~3 min for direct,
~15 min for SVD). Training takes ~3-4 hours per run.
