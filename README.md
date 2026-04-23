# Structure-Preserving Residual Initialization for Dense-to-MoE Upcycling

Dense-to-MoE upcycling for **Qwen1.5-1.8B → Qwen1.5-MoE-A2.7B**, comparing
direct copy, Gaussian perturbation, and SVD residual initialization.

## Method

We convert a pretrained dense model into a Mixture-of-Experts architecture by
replacing FFN layers with routed experts plus a shared expert. For each weight
matrix W:

1. Compute SVD: W = U Σ Vᵀ
2. Split: Σ = Σstruct (top-k) + Σres (remainder)
3. Per expert: We = U (Σstruct + Σ̃res,e) Vᵀ, where Σ̃res,e is sampled
  around Σres with variance scaled by singular value magnitude

## Models


| Role         | Model                           | Details                                        |
| ------------ | ------------------------------- | ---------------------------------------------- |
| Dense parent | `Qwen/Qwen1.5-1.8B`             | 24 layers, hidden 2048, intermediate 5504      |
| MoE target   | `Qwen/Qwen1.5-MoE-A2.7B` config | 60 experts, 4 active, expert intermediate 1408 |


## Experimental settings

1. **Direct copy** — all experts initialized as exact copies of the dense FFN
2. **Gaussian perturbation** — copies + i.i.d. Gaussian noise
3. **SVD residual** — structure-preserving initialization described above

## Dataset

**OpenMathReasoning** (NVIDIA): supervised fine-tuning on math reasoning.

## Infrastructure

We use [Prime Intellect](https://app.primeintellect.ai) for GPU compute,
managed via the official `[prime` CLI](https://github.com/PrimeIntellect-ai/prime).

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

### Sanity check

```bash
# Provision a pod and SSH in
prime pods create
prime pods ssh <pod-id>

# On the pod:
nvidia-smi
git clone https://github.com/simjay/structural-moe-upcycling.git
cd structural-moe-upcycling
bash setup.sh
python3 tests/test_inference.py

# Back on local machine:
prime pods terminate <pod-id>
```

## References

- Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023
- Horoi et al., "Less is More: Undertraining Experts Improves Model Upcycling", 2025
- Hui et al., "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging", 2025
- Liew et al., "Scaling Laws for Upcycling Mixture-of-Experts Language Models", 2025

