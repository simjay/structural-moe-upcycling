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

## Dataset

**OpenMathReasoning** (NVIDIA): supervised fine-tuning on math reasoning.

## Training

- **Framework**: Unsloth + Transformers v5, PyTorch, bf16
- **Method**: LoRA on expert FFN layers (gate, up, down projections)
- **LoRA config**: rank 32, alpha 64, dropout 0.05
- **Context length**: 2048 (1024 for sanity runs)
- **Router**: frozen for first 5% of steps, then unfrozen

## Evaluation

- **Preservation**: validation loss at step 0 (before fine-tuning)
- **Convergence**: validation loss vs. step, area under early learning curve
- **Specialization**: routing entropy, expert load balance, pairwise cosine similarity of expert weights

---

## Setup

### Prerequisites

- Python 3.10+
- A [Prime Intellect](https://app.primeintellect.ai) account with an API key
  and a registered SSH key

### Installation

```bash
git clone <repo-url>
cd structural-moe-upcycling
make setup
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and set `PRIME_INTELLECT_API_KEY`. The other variables have defaults
but you should verify `PI_IMAGE` matches what's available — run `make list-offers`
to check.

## Usage

### Discovery

```bash
make list-offers    # show available GPUs, prices, and supported images
make list-images    # show just the image names
make list-pods      # show your active pods
```

### Running remote tests

```bash
make test-provision    # provision a GPU, run nvidia-smi, tear down
make test-inference    # provision, install PyTorch/transformers, load Qwen1.5-1.8B, generate a few tokens
```

The inference test installs PyTorch and transformers on the pod, downloads
Qwen1.5-1.8B from Hugging Face (~3.6GB), and generates a few tokens.

To run any config directly:

```bash
venv/bin/python -m experiment.primeintellect --config configs/<your-config>.yaml -v
```

### Emergency teardown

```bash
make kill-pod POD_ID=<id>
```

## References

- Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints", 2023
- Horoi et al., "Less is More: Undertraining Experts Improves Model Upcycling", 2025
- Hui et al., "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging", 2025
- Liew et al., "Scaling Laws for Upcycling Mixture-of-Experts Language Models", 2025
