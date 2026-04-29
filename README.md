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
compared on convergence speed and final loss.

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
| `test_inference.py` | Load Qwen1.5-1.8B via Unsloth and generate tokens, confirms GPU, CUDA, and model loading work | ~30 s |
| `test_data.py` | Stream and inspect samples from `nvidia/OpenMathReasoning`, confirms dataset access and the `datasets` library | ~10 s |
| `test_upcycle.py` | Convert dense Qwen1.5-1.8B → MoE (14.3B params): exact-copy shared expert (5504), partition dense FFN into 4 row-slices and replicate 15× for 60 experts, run inference | ~3 min |
| `test_train.py` | Apply LoRA to the dense model and run 10 SFT steps on a synthetic math dataset, confirms the full training loop (forward, loss, backward, optimizer) | ~1 min |

## Running the experiments

### Qwen1.5 (small-scale)

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

### Mixtral (large-scale)

```bash
# Upcycle
python3 -m src.mixtral.upcycle --method direct    --output /tmp/mixtral-direct
python3 -m src.mixtral.upcycle --method gaussian  --sigma 0.01 --output /tmp/mixtral-gaussian
python3 -m src.mixtral.upcycle --method svd       --k 256 --output /tmp/mixtral-svd

# Train (on 4x H100-80GB)
python3 -m src.mixtral.train --model /tmp/mixtral-direct    --run-name mixtral-direct   --max-steps 500
python3 -m src.mixtral.train --model /tmp/mixtral-gaussian  --run-name mixtral-gaussian --max-steps 500
python3 -m src.mixtral.train --model /tmp/mixtral-svd       --run-name mixtral-svd      --max-steps 500
```

### Hyperparameters (held constant across all runs)

| Parameter | Value |
| --- | --- |
| LoRA rank | 16 |
| Batch size | 2 (x 4 gradient accumulation = effective 8) |
| Learning rate | 2e-4, cosine schedule, 100 warmup steps |
| Sequence length | 2048 |
| Precision | bf16 |
| Gradient checkpointing | enabled |
| Dataset | OpenMathReasoning (cot split) |

### Compare

Open wandb and compare the six runs on:
- Training loss convergence speed
- Final training loss at step 500
- Cross-scale consistency (does the same init method win at both scales?)

### Project structure

```
structural-moe-upcycling/
├── src/
│   ├── __init__.py
│   ├── qwen15/
│   │   ├── __init__.py
│   │   ├── upcycle.py   # Qwen1.5: 3 init methods + shared/routing expert setup
│   │   ├── train.py     # Qwen1.5: LoRA SFT (attention + shared expert)
│   │   └── README.md
│   └── mixtral/
│       ├── __init__.py
│       ├── upcycle.py   # Mixtral: 3 init methods, full-size experts
│       ├── train.py     # Mixtral: LoRA SFT (attention only), multi-GPU
│       └── README.md
├── tests/
│   ├── test_inference.py
│   ├── test_data.py
│   ├── test_upcycle.py
│   └── test_train.py
├── setup.sh
├── pyproject.toml
└── README.md
```

## References

- Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023
- Horoi et al., "Less is More: Undertraining Experts Improves Model Upcycling", 2025
- Hui et al., "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging", 2025
- Liew et al., "Scaling Laws for Upcycling Mixture-of-Experts Language Models", 2025

