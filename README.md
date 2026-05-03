# Structure-Preserving Residual Initialization for Dense-to-MoE Upcycling

Dense-to-MoE upcycling comparing direct copy, Gaussian perturbation, and SVD
residual initialization across two model pairs:

1. **Qwen1.5-1.8B → Qwen1.5-MoE-A2.7B** — small-scale experiment with shared expert and dimension adjustment
2. **Mistral 7B → Mixtral 8x7B** — large-scale experiment with perfectly matched dimensions

## Experiments

| Experiment | Dense model | MoE target | Scale | Shared expert | Hardware |
| --- | --- | --- | --- | --- | --- |
| Qwen1.5 | `Qwen/Qwen1.5-1.8B` | `Qwen/Qwen1.5-MoE-A2.7B` | 1.8B → 14.3B | yes (5504) | 1x A100-80GB |
| Mixtral | `mistralai/Mistral-7B-v0.1` | `mistralai/Mixtral-8x7B-v0.1` | 7B → 47B | none | 4x H100-80GB |

See each experiment's README for architecture details and CLI commands:
- [`src/qwen15/README.md`](src/qwen15/README.md)
- [`src/mixtral/README.md`](src/mixtral/README.md)


## Method

We convert a pretrained dense model into a Mixture-of-Experts architecture by
replacing each FFN layer with routed experts. Embeddings, attention (Q/K/V/O),
and layer norms are copied directly (dimensions match between dense and MoE).

The routing experts are initialized from the dense FFN weights using one of three
methods:

1. **Direct copy** — each expert receives (a partition of) the dense FFN weights
2. **Gaussian perturbation** — direct copy + i.i.d. Gaussian noise per expert
3. **SVD residual** — decompose W = U Σ Vᵀ, keep top-k singular values as
   structural component, perturb the residual independently per expert:
   W_e = U (Σ_struct + Σ̃_res,e) Vᵀ

All three methods are trained under identical conditions on OpenMathReasoning and
compared on convergence speed and final loss. Only expert FFN weights and the
router are trained (attention is frozen) to isolate the effect of expert
initialization on convergence.

## Dataset

**OpenMathReasoning** (NVIDIA): supervised fine-tuning on math reasoning.

## Infrastructure

We use [Prime Intellect](https://app.primeintellect.ai) for GPU compute,
managed via the official [prime CLI](https://github.com/PrimeIntellect-ai/prime).

### Prerequisites

- Python 3.10–3.13
- A Prime Intellect account with API key and registered SSH key

### Setup

```bash
pip install prime
prime config set-api-key
prime config set-ssh-key-path
```

### Usage

```bash
# See available GPUs
prime availability list
prime availability list --gpu-type A100_80GB

# Provision a pod (interactive)
prime pods create

# List your pods
prime pods list

# SSH into a pod
prime pods ssh <pod-id>

# Terminate a pod
prime pods terminate <pod-id>
```

### Sanity checks

```bash
# Provision a pod and SSH in
prime pods create
prime pods ssh <pod-id>

# On the pod:
nvidia-smi
git clone https://github.com/simjay/structural-moe-upcycling.git
cd structural-moe-upcycling
bash setup.sh

# Run tests in order:
python3 tests/test_inference.py
python3 tests/test_data.py
python3 tests/test_upcycle.py
python3 tests/test_train.py

# Back on local machine:
prime pods terminate <pod-id>
```

| Script | What it tests | Time |
| --- | --- | --- |
| `test_inference.py` | Load Qwen1.5-1.8B and generate tokens, confirms GPU, CUDA, and model loading work | ~30 s |
| `test_data.py` | Stream and inspect samples from `nvidia/OpenMathReasoning`, confirms dataset access and the `datasets` library | ~10 s |
| `test_upcycle.py` | Convert dense Qwen1.5-1.8B → MoE (14.3B params): exact-copy shared expert (5504), partition dense FFN into 4 row-slices and replicate 15× for 60 experts, run inference | ~3 min |
| `test_train.py` | Build a tiny Mixtral MoE model, freeze attention, train experts + router for 10 SFT steps | ~1 min |

## Running the experiments

Each experiment has a run script in `scripts/` that handles the full pipeline. Results are logged to wandb and saved as JSON locally.

```bash
# Qwen1.5 — init analysis + training dynamics (1x A100-80GB)
bash scripts/run_qwen15.sh

# Mixtral — full sweep (4x H100-80GB)
bash scripts/run_mixtral.sh
```

To run individual steps manually:

```bash
# Upcycle one method
python3 -m src.qwen15.upcycle --method svd --output /tmp/qwen-moe-svd

# Analyze initialization quality (no training)
python3 -m src.qwen15.init_analysis --method svd --k 128 --svd-scale 0.5 \
    --output results/qwen15/step0/svd-k128-s0.5.json

# Train one method
python3 -m src.qwen15.train --model /tmp/qwen-moe-svd --run-name qwen-svd

# Evaluate one checkpoint
python3 -m src.eval.gsm8k --model /tmp/moe-checkpoints/qwen-svd/final
```

### Hyperparameters (held constant across all runs)

| Parameter | Value |
| --- | --- |
| Trained components | expert FFNs + router only (attention frozen) |
| Gaussian sigma | 0.1 |
| SVD k (structural singular values) | 128 |
| SVD residual noise scale | 0.1 |
| Max steps | 300 |
| Batch size | 1 (x 4 gradient accumulation = effective 4) |
| Learning rate | 1e-5, cosine schedule, 30 warmup steps |
| Optimizer | adamw_8bit |
| Sequence length | 2048 |
| Precision | bf16 |
| Gradient checkpointing | enabled |
| Dataset | OpenMathReasoning (cot split, shuffled) |
| Eval dataset | GSM8K train split (7473 problems) |

### Evaluation metrics

All metrics are logged to wandb. Compare the six runs on:

| Metric | Frequency | What it shows |
| --- | --- | --- |
| `eval/loss` | every 50 steps | Generalization to unseen math — primary convergence metric |
| `expert/mean_cosine_similarity` | every 10 steps | Expert divergence; lower = more specialized experts |
| `train/loss` | every 10 steps | Training set fit (expect similar across methods) |
| `gsm8k/accuracy` | end of training | Task performance on 200 GSM8K test problems (in run summary) |

### Project structure

```
structural-moe-upcycling/
├── src/
│   ├── __init__.py
│   ├── eval/
│   │   ├── __init__.py
│   │   └── gsm8k.py      # GSM8K accuracy evaluation
│   ├── qwen15/
│   │   ├── __init__.py
│   │   ├── upcycle.py       # 3 init methods + shared/routing expert setup
│   │   ├── train.py         # Train experts + router, log metrics to wandb
│   │   ├── init_analysis.py # Measure init quality + expert diversity (no training)
│   │   └── README.md
│   └── mixtral/
│       ├── __init__.py
│       ├── upcycle.py       # 3 init methods, full-size experts
│       ├── train.py         # Train experts + router, log metrics to wandb
│       └── README.md
├── scripts/
│   ├── run_qwen15.sh        # Qwen1.5: init analysis sweep + training dynamics
│   └── run_mixtral.sh       # Mixtral: full pipeline (all configs)
├── setup.sh
├── tests/
│   ├── test_inference.py
│   ├── test_data.py
│   ├── test_upcycle.py
│   └── test_train.py
├── pyproject.toml
└── README.md
```

## References

- Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023
- Horoi et al., "Less is More: Undertraining Experts Improves Model Upcycling", 2025
- Hui et al., "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging", 2025
- Liew et al., "Scaling Laws for Upcycling Mixture-of-Experts Language Models", 2025

