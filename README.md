# Structure-Preserving Residual Initialization for Dense-to-MoE Upcycling

Dense-to-MoE upcycling for **Qwen1.5-1.8B → Qwen1.5-MoE-A2.7B**, comparing direct copy, Gaussian perturbation, and SVD residual initialization.

## Method

We convert a pretrained dense model into a Mixture-of-Experts architecture by
replacing FFN layers with routed experts plus a shared expert. The core idea is
to decompose each dense FFN weight matrix via SVD into a dominant structure
component and a residual, then generate expert-specific weights by sampling only
in the residual space. This preserves the pretrained behavior in the top-k
singular directions while introducing controlled diversity where it matters
least.

For each weight matrix W:

1. Compute SVD: W = U Σ Vᵀ
2. Split: Σ = Σ_struct (top-k) + Σ_res (remainder)
3. Per expert: W_e = U (Σ_struct + Σ̃_res,e) Vᵀ, where Σ̃_res,e is sampled
   around Σ_res with variance scaled by singular value magnitude

## Models

| Role | Model | Details |
|------|-------|---------|
| Dense parent | `Qwen/Qwen1.5-1.8B` | 24 layers, hidden size 2048, intermediate 5504 |
| MoE target | `Qwen/Qwen1.5-MoE-A2.7B` config | 60 experts, 4 active per token, expert intermediate 1408, shared expert intermediate 5632 |

The MoE target is instantiated as a **fresh config** (not the pretrained MoE
checkpoint) so the experiment isolates the effect of initialization strategy.

## Experiments

| Setting | Strategy | Description |
|---------|----------|-------------|
| A | Direct copy | Every expert receives identical projected weights |
| B | Gaussian perturbation | Projected weights + iid Gaussian noise on full matrix |
| C | SVD residual sampling | Preserve top-k structure, sample diversity in residual space |

Each setting is run with 2 seeds (42, 123), 3000 training steps, and identical
hyperparameters. See `configs/setting_*.yaml` for the full definitions.

## Dataset

**OpenMathReasoning** (NVIDIA): supervised fine-tuning on math reasoning.

| Split | Size |
|-------|------|
| Train | 20k |
| Val | 1k |
| Test | 1k |

A 5k/500/500 pilot subset is used for pipeline sanity checks.

## Training

- **Framework**: Unsloth + Transformers v5, PyTorch, bf16
- **Method**: LoRA on expert FFN layers (gate, up, down projections)
- **LoRA config**: rank 32, alpha 64, dropout 0.05
- **Context length**: 2048 (1024 for sanity runs)
- **Budget**: 3000 steps, 2 seeds per setting
- **Router**: frozen for first 5% of steps, then unfrozen

## Evaluation

- **Preservation**: validation loss at step 0 (before fine-tuning)
- **Convergence**: validation loss vs. step, area under early learning curve
- **Specialization**: routing entropy, expert load balance, pairwise cosine similarity of expert weights

---

## Usage

### Prerequisites

- Python 3.10+
- An SSH key registered with [Prime Intellect](https://app.primeintellect.ai)
  (for remote GPU runs)

### Installation

```bash
git clone <repo-url>
cd structural-moe-upcycling
pip install -e .
```

To also install the training dependencies (torch, transformers, unsloth):

```bash
pip install -e ".[train]"
```

### Configuration

Copy the environment template and add your Prime Intellect API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
PRIME_INTELLECT_API_KEY=your_key_here
```

The remaining variables have sensible defaults (A100_80GB, 300GB disk, CUDA 12.1
image). Override them in `.env` if needed.

### Running Experiments

**Single setting** (provisions one GPU, runs both seeds sequentially):

```bash
python -m experiment.primeintellect --config configs/setting_a_direct_copy.yaml
```

**Single setting, seeds in parallel** (provisions two GPUs simultaneously):

```bash
python -m experiment.primeintellect --config configs/setting_a_direct_copy.yaml --parallel
```

**All three settings sequentially:**

```bash
python -m experiment.primeintellect --config configs/setting_a_direct_copy.yaml
python -m experiment.primeintellect --config configs/setting_b_gaussian.yaml
python -m experiment.primeintellect --config configs/setting_c_svd_residual.yaml
```

**Verbose logging:**

```bash
python -m experiment.primeintellect --config configs/setting_c_svd_residual.yaml -v
```

Results are downloaded to `outputs/<experiment-name>/<timestamp>/`.

### Experiment Configs

Each config is a YAML file under `configs/` that defines the full remote lifecycle: setup commands, the training command, which files to upload, and which result paths to download.  
The three provided configs are:

| Config | Setting | Init strategy | Init-specific params |
|--------|---------|---------------|----------------------|
| `setting_a_direct_copy.yaml` | A | `direct_copy` | — |
| `setting_b_gaussian.yaml` | B | `gaussian` | `--noise-scale 0.01` |
| `setting_c_svd_residual.yaml` | C | `svd_residual` | `--svd-rank 512 --noise-scale 1.0` |

All other hyperparameters (LoRA rank/alpha/dropout, steps, context length,
eval schedule) are identical across settings.

## References

- Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023
- Horoi et al., "Less is More: Undertraining Experts Improves Model Upcycling", 2025
- Hui et al., "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging", 2025
- Liew et al., "Scaling Laws for Upcycling Mixture-of-Experts Language Models", 2025
