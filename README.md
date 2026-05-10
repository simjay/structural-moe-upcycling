# Structural MoE Upcycling

[[paper]](resource/report.pdf) [[poster]](resource/poster.pdf)

**Spectral residual initialization** for dense-to-MoE upcycling. Decomposes dense FFN weights via SVD, preserves the top-*k* singular values, and perturbs only the residual per expert.

<p align="center">
  <img src="resource/diagram.svg" width="680" alt="Comparison of direct copy, Gaussian perturbation, and spectral residual initialization.">
</p>

## Quick start

```bash
git clone https://github.com/simjay/structural-moe-upcycling.git
cd structural-moe-upcycling
bash setup.sh
```

Run the full sweep for either model:

```bash
# Qwen1.5 (1x A100-80GB)
bash scripts/run_qwen15.sh

# Mixtral (8x A100-80GB)
bash scripts/run_mixtral.sh
```

Or run a single configuration:

```bash
# 1. Upcycle
python3 -m src.qwen15.upcycle --method svd --k 512 --svd-scale 0.5 \
    --output /tmp/qwen-moe-svd

# 2. Train
python3 -m src.qwen15.train --model /tmp/qwen-moe-svd --run-name qwen-svd

# 3. Evaluate
python3 -m src.eval.gsm8k --model /tmp/moe-checkpoints/qwen-svd/final
```

## Supported models

| Dense | MoE | Params | Experts | Shared expert |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen1.5-1.8B` | `Qwen/Qwen1.5-MoE-A2.7B` | 1.8B to 14.3B | 60, top-4 | Yes |
| `mistralai/Mistral-7B-v0.1` | `mistralai/Mixtral-8x7B-v0.1` | 7B to 46.7B | 8, top-2 | No |

See per-experiment details:
- [`src/qwen15/README.md`](src/qwen15/README.md)
- [`src/mixtral/README.md`](src/mixtral/README.md)

## Initialization methods

Three expert initialization strategies are implemented in `upcycle.py`:

| Method | Flag | What it does |
| --- | --- | --- |
| Direct copy | `--method direct` | Copy dense FFN weights into each expert slot |
| Gaussian | `--method gaussian` | Direct copy + i.i.d. noise scaled by mean weight magnitude |
| Spectral residual | `--method svd` | SVD the dense weights, keep top-*k* singular values fixed, perturb the rest per expert |

Key hyperparameters for spectral residual: `--k` (structural rank) and `--svd-scale` (noise scale on residual singular values).

## Results

Evaluated on GSM8K (4-shot, greedy decoding, full test split of 1,319 problems). Only expert FFN weights and the router are trained. Everything else is frozen.

**Qwen1.5** (400 steps, lr=5e-4, 1x A100-80GB):

| Method | Accuracy | vs. direct copy |
| --- | --- | --- |
| Dense baseline | 33.9% | - |
| **Spectral k=512, s=0.5** | **27.9%** | **+2.3%** |
| Spectral k=512, s=0.1 | 27.1% | +1.5% |
| Direct copy | 25.6% | 0.0% |
| Gaussian σ=0.5 | 25.5% | -0.1% |
| Gaussian σ=1.0 | 25.5% | -0.1% |

**Mixtral** (300 steps, lr=1e-5, 8x A100-80GB):

| Method | Accuracy | vs. direct copy |
| --- | --- | --- |
| **Spectral k=512, s=0.1** | **26.7%** | **+1.2%** |
| Gaussian σ=0.1 | 26.1% | +0.6% |
| Direct copy | 25.5% | 0.0% |
| Spectral k=512, s=0.5 | 18.7% | -6.8% |
| Gaussian σ=0.5 | 18.1% | -7.3% |

## Training config

| | Qwen1.5 | Mixtral |
| --- | --- | --- |
| Steps | 400 | 300 |
| Batch size | 1 x 4 grad accum | 1 x 4 grad accum |
| Learning rate | 5e-4, cosine | 1e-5, cosine |
| Warmup | 30 steps | 30 steps |
| Optimizer | AdamW 8-bit | AdamW 8-bit |
| Precision | bfloat16 | bfloat16 |
| Seq length | 2048 | 2048 |
| Hardware | 1x A100-80GB | 8x A100-80GB |

Training is done with [TRL](https://github.com/huggingface/trl) `SFTTrainer` on the [GSM8K](https://arxiv.org/abs/2110.14168) training split (7,473 problems). Results are logged to [wandb](https://wandb.ai).

## Project structure

```
structural-moe-upcycling/
├── src/
│   ├── eval/
│   │   └── gsm8k.py              # GSM8K accuracy eval
│   ├── qwen15/
│   │   ├── upcycle.py             # upcycle dense to MoE (3 init methods)
│   │   ├── train.py               # train + eval
│   │   └── README.md
│   └── mixtral/
│       ├── upcycle.py             # upcycle dense to MoE (3 init methods)
│       ├── train.py               # train + eval
│       └── README.md
├── scripts/
│   ├── run_qwen15.sh
│   └── run_mixtral.sh
├── tests/
├── resource/
├── setup.sh
├── pyproject.toml
└── README.md
```