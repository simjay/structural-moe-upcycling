"""Dense-to-MoE upcycling: Mistral 7B → Mixtral 8x7B.

Loads Mistral-7B-v0.1, creates a Mixtral-8x7B-v0.1 architecture, copies all
shared weights (attention, embeddings, norms), then initializes the 8 routing
experts using one of three methods.

Unlike the Qwen1.5 experiment, Mixtral has no shared expert, and each expert's
FFN dimension (14336) matches the dense model exactly — no partitioning or
dimension adjustment needed.

Methods:
    direct:   Copy dense FFN identically into all 8 expert slots.
    gaussian: Direct copy + per-expert Gaussian noise.
    svd:      SVD decomposition with structural/residual split.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DENSE_MODEL = "mistralai/Mistral-7B-v0.1"
MOE_REF = "mistralai/Mixtral-8x7B-v0.1"


def _copy_common_weights(dense, moe, n_layers):
    """Copy embeddings, attention projections, and layer norms.

    All dimensions match exactly between Mistral 7B and Mixtral 8x7B
    (hidden=4096, heads=32, kv_heads=8, vocab=32000).

    Args:
        dense: The dense Mistral-7B model.
        moe: The empty Mixtral MoE model to populate.
        n_layers: Number of transformer layers to copy.
    """
    moe.model.embed_tokens.weight.copy_(dense.model.embed_tokens.weight)
    moe.model.norm.weight.copy_(dense.model.norm.weight)
    moe.lm_head.weight.copy_(dense.lm_head.weight)

    for i in range(n_layers):
        dl = dense.model.layers[i]
        ml = moe.model.layers[i]

        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src, dst = getattr(dl.self_attn, proj), getattr(ml.self_attn, proj)
            dst.weight.copy_(src.weight)
            if src.bias is not None and dst.bias is not None:
                dst.bias.copy_(src.bias)

        ml.input_layernorm.weight.copy_(dl.input_layernorm.weight)
        ml.post_attention_layernorm.weight.copy_(dl.post_attention_layernorm.weight)


def init_direct(dense, moe, cfg):
    """Initialize all experts as identical copies of the dense FFN.

    Each of the 8 experts receives the full dense FFN weights (14336 dim).
    The fused ``gate_up_proj[e]`` is filled with gate_proj in the first half
    and up_proj in the second half.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]

    for i in range(n_layers):
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts

        gate_w = dl.mlp.gate_proj.weight  # (14336, 4096)
        up_w = dl.mlp.up_proj.weight       # (14336, 4096)
        down_w = dl.mlp.down_proj.weight   # (4096, 14336)

        for e in range(n_experts):
            experts.gate_up_proj[e, :expert_dim, :].copy_(gate_w)
            experts.gate_up_proj[e, expert_dim:, :].copy_(up_w)
            experts.down_proj[e].copy_(down_w)


def init_gaussian(dense, moe, cfg, sigma=0.1):
    """Initialize experts as direct copies, then add per-expert Gaussian noise.

    Applies the direct-copy initialization first, then perturbs each expert's
    weights with i.i.d. Gaussian noise scaled by the mean absolute weight
    magnitude.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``.
        sigma: Noise scale relative to mean absolute weight magnitude.
            Defaults to 0.1.
    """
    init_direct(dense, moe, cfg)

    for i in range(cfg["n_layers"]):
        experts = moe.model.layers[i].mlp.experts
        std = sigma * experts.gate_up_proj.abs().mean()
        experts.gate_up_proj.add_(torch.randn_like(experts.gate_up_proj) * std)
        std = sigma * experts.down_proj.abs().mean()
        experts.down_proj.add_(torch.randn_like(experts.down_proj) * std)


def _svd_perturb(W, k, svd_scale):
    """Apply SVD-residual perturbation to a weight matrix.

    Decomposes W = U @ diag(S) @ Vt, keeps top-k singular values unchanged
    (structural), and adds uniform additive noise to the remaining residual
    singular values.

    Args:
        W: Dense weight matrix, shape ``(out_features, in_features)``.
        k: Number of top singular values to preserve as structural.
        svd_scale: Noise scale relative to mean residual singular value
            magnitude (mirrors how Gaussian uses sigma * mean(|W|)).

    Returns:
        Perturbed weight matrix with same shape and dtype as input.
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    rank = min(k, len(S))

    S_res = S[rank:]
    std = svd_scale * S_res.abs().mean()
    S_perturbed = S_res + torch.randn_like(S_res) * std

    S_full = torch.cat([S[:rank], S_perturbed])
    W_new = U @ torch.diag(S_full) @ Vt
    return W_new.to(W.dtype)


def init_svd(dense, moe, cfg, k=8, svd_scale=0.5):
    """Initialize experts via SVD-residual perturbation.

    For each dense FFN matrix, applies SVD independently per expert: keeps
    top-k singular values unchanged (structural) and adds uniform additive
    noise to the residual singular values. Each expert gets an independent
    noise sample, creating diversity while preserving the dominant structure.

    Since expert_dim = dense_dim (14336) in the Mixtral architecture,
    no partitioning is needed — each expert receives the full matrix.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``.
        k: Number of top singular values to keep as structural.
            Defaults to 8.
        svd_scale: Noise scale for residual perturbation (relative to mean
            residual singular value magnitude). Defaults to 0.5.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]

    for i in range(n_layers):
        print(f"  SVD init layer {i}/{n_layers}...")
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts

        gate_w = dl.mlp.gate_proj.weight
        up_w = dl.mlp.up_proj.weight
        down_w = dl.mlp.down_proj.weight

        for e in range(n_experts):
            experts.gate_up_proj[e, :expert_dim, :].copy_(
                _svd_perturb(gate_w, k, svd_scale))
            experts.gate_up_proj[e, expert_dim:, :].copy_(
                _svd_perturb(up_w, k, svd_scale))
            experts.down_proj[e].copy_(
                _svd_perturb(down_w, k, svd_scale))


def upcycle(method, output_dir, sigma=0.1, k=8, svd_scale=0.5):
    """Build Mixtral 8x7B from Mistral 7B using the given init method.

    Loads the dense model, creates an empty MoE shell, copies all shared
    weights, initializes routing experts, and saves the result to disk.

    Args:
        method: One of ``"direct"``, ``"gaussian"``, or ``"svd"``.
        output_dir: Path to save the upcycled model and tokenizer.
        sigma: Noise scale for the gaussian method. Defaults to 0.1.
        k: Number of structural singular values for the svd method.
            Defaults to 8.
        svd_scale: Noise scale for SVD residual perturbation. Defaults to 0.5.
    """
    print(f"=== Upcycling Mistral 7B → Mixtral 8x7B with method={method} ===\n")

    dense_cfg = AutoConfig.from_pretrained(DENSE_MODEL)
    moe_cfg = AutoConfig.from_pretrained(MOE_REF)

    print(f"Dense: hidden={dense_cfg.hidden_size}, ffn={dense_cfg.intermediate_size}, "
          f"layers={dense_cfg.num_hidden_layers}, heads={dense_cfg.num_attention_heads}, "
          f"kv_heads={dense_cfg.num_key_value_heads}")
    print(f"MoE:   experts={moe_cfg.num_local_experts}x{moe_cfg.intermediate_size}, "
          f"active={moe_cfg.num_experts_per_tok}/tok, "
          f"layers={moe_cfg.num_hidden_layers}\n")

    assert dense_cfg.hidden_size == moe_cfg.hidden_size, \
        f"hidden_size mismatch: {dense_cfg.hidden_size} vs {moe_cfg.hidden_size}"
    assert dense_cfg.intermediate_size == moe_cfg.intermediate_size, \
        f"intermediate_size mismatch: {dense_cfg.intermediate_size} vs {moe_cfg.intermediate_size}"
    assert dense_cfg.num_hidden_layers == moe_cfg.num_hidden_layers, \
        f"num_hidden_layers mismatch: {dense_cfg.num_hidden_layers} vs {moe_cfg.num_hidden_layers}"

    print("Loading dense model...")
    dense = AutoModelForCausalLM.from_pretrained(
        DENSE_MODEL, dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)

    print("Creating empty MoE model (bf16)...")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    moe = AutoModelForCausalLM.from_config(moe_cfg)
    torch.set_default_dtype(prev)

    cfg = dict(
        n_layers=dense_cfg.num_hidden_layers,
        n_experts=moe_cfg.num_local_experts,
        expert_dim=moe_cfg.intermediate_size,
    )

    print("Copying shared weights (attention, embeddings, norms)...")
    with torch.no_grad():
        _copy_common_weights(dense, moe, cfg["n_layers"])

        print(f"Initializing {cfg['n_experts']} experts ({method})...")
        if method == "direct":
            init_direct(dense, moe, cfg)
        elif method == "gaussian":
            init_gaussian(dense, moe, cfg, sigma=sigma)
        elif method == "svd":
            init_svd(dense, moe, cfg, k=k, svd_scale=svd_scale)
        else:
            raise ValueError(f"Unknown method: {method}")

    del dense

    print(f"\nSaving to {output_dir}...")
    moe.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    params = sum(p.numel() for p in moe.parameters())
    print(f"MoE total params: {params:,} ({params / 1e9:.1f}B)")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Dense-to-MoE upcycling: Mistral 7B → Mixtral 8x7B")
    parser.add_argument("--method", required=True,
                        choices=["direct", "gaussian", "svd"])
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Noise scale for gaussian method")
    parser.add_argument("--k", type=int, default=128,
                        help="Number of structural singular values for svd method")
    parser.add_argument("--svd-scale", type=float, default=0.1,
                        help="Noise scale for SVD residual perturbation")
    args = parser.parse_args()
    upcycle(args.method, args.output, sigma=args.sigma, k=args.k,
            svd_scale=args.svd_scale)


if __name__ == "__main__":
    main()
