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


## Upcycling setup

Each of the 24 dense transformer layers has a single FFN (`gate_proj`, `up_proj`,
`down_proj` with intermediate size 5504).  We replace every FFN with an MoE block:

```
x (2048) ───┬── shared expert (2048 → 5504 → 2048)  ← exact copy of dense FFN
            │
            ├── router (2048 → 60 scores) → pick top-4
            │     ├── expert i  (2048 → 1408 → 2048) × weight_i
            │     ├── expert j  (2048 → 1408 → 2048) × weight_j
            │     ├── expert k  (2048 → 1408 → 2048) × weight_k
            │     └── expert l  (2048 → 1408 → 2048) × weight_l
            │
            └── ADD all outputs → (2048)
```

**What stays the same:** embeddings, attention (Q/K/V/O), layer norms. These are copied directly
(dimensions are identical in dense and MoE).

**Dimension adjustment:** the reference Qwen1.5-MoE-A2.7B config uses
`shared_expert_intermediate_size = 5632` and `moe_intermediate_size = 1408`, but the
dense FFN has `intermediate_size = 5504`.  Neither matches exactly.  We override the
shared expert to 5504 so the dense FFN weights can be copied without padding or
truncation.  For the 60 routing experts (1408 each), the dense FFN (5504 rows) is
partitioned into 4 non-overlapping row-slices: three of 1408 rows and one of 1280 rows
(zero-padded to 1408).  This follows Qwen's "fine-grained expert" approach of splitting
rather than shrinking the FFN.

**Shared expert:** exact copy of the dense FFN (same 5504 intermediate size).  Preserves
the original model's behavior.

**Router + shared expert gate:** randomly initialized (no dense counterpart exists).

**60 routing experts:** the shared expert and router are held constant across experiments;
only the expert initialization varies.

## Experimental settings

1. **Direct copy**: partition the dense FFN into 4 row-slices of 1376/1408 rows,
   replicate 15× to fill 60 experts (all copies identical)
2. **Gaussian perturbation**: same partition + i.i.d. Gaussian noise per copy
3. **SVD residual**: structure-preserving initialization described above

All three are trained under identical conditions on OpenMathReasoning and compared
on convergence speed and final loss.

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

## Running the experiment

### 1. Upcycle (create three MoE checkpoints)

```bash
# Direct copy (partition + replicate, ~3 min on CPU)
python3 -m src.upcycle --method direct --output /tmp/moe-direct

# Gaussian perturbation (partition + replicate + noise)
python3 -m src.upcycle --method gaussian --sigma 0.01 --output /tmp/moe-gaussian

# SVD residual (structural/residual split, ~15 min on CPU)
python3 -m src.upcycle --method svd --k 256 --output /tmp/moe-svd
```

Each command loads the dense Qwen1.5-1.8B, builds a 14.3B-parameter MoE model,
and saves it to disk (~28 GB per checkpoint).

### 2. Train (fine-tune each checkpoint)

```bash
python3 -m src.train --model /tmp/moe-direct    --run-name direct   --max-steps 2000
python3 -m src.train --model /tmp/moe-gaussian  --run-name gaussian --max-steps 2000
python3 -m src.train --model /tmp/moe-svd       --run-name svd      --max-steps 2000
```

Training uses LoRA (rank 16) on attention + shared expert, with the
OpenMathReasoning `cot` split.  Loss curves are logged to wandb for side-by-side
comparison.  Expected time: ~3–4 hours per run on a single A100-80GB.

Key hyperparameters (held constant across runs):

| Parameter | Value |
| --- | --- |
| LoRA rank | 16 |
| Batch size | 2 (× 4 gradient accumulation = effective 8) |
| Learning rate | 2e-4, cosine schedule, 100 warmup steps |
| Sequence length | 2048 |
| Precision | bf16 |
| Gradient checkpointing | enabled |

### 3. Compare

Open wandb and compare the three runs (`direct`, `gaussian`, `svd`) on:
- Training loss convergence speed
- Final training loss at step 2000

### Project structure

```
structural-moe-upcycling/
├── src/
│   ├── __init__.py
│   ├── upcycle.py      # 3 init methods + shared weight-copying
│   └── train.py        # LoRA SFT with wandb logging
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

