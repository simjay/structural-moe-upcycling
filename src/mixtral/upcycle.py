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


def _svd_init_matrix(W, n_experts, k, svd_scale):
    """Initialize expert matrices from a dense weight via SVD residual sampling.

    Decomposes W = U @ diag(S) @ Vt, keeps top-k singular values as the
    shared structural component, and samples perturbed residuals per expert.

    Since expert_dim = dense_dim for Mixtral, each expert gets the full
    output dimension — no row truncation is needed.

    Args:
        W: Dense weight matrix, shape ``(out_features, in_features)``,
            e.g. ``(14336, 4096)`` for gate/up_proj.
        n_experts: Total number of experts to initialize (8).
        k: Number of top singular values to treat as structural.
        svd_scale: Noise scale for residual perturbation (multiplied by
            each residual singular value's magnitude).

    Returns:
        Tensor of shape ``(n_experts, out_features, in_features)``.
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

    rank = min(k, len(S))
    S_struct = S[:rank]
    U_struct = U[:, :rank]
    Vt_struct = Vt[:rank, :]

    S_res = S[rank:]
    U_res = U[:, rank:]
    Vt_res = Vt[rank:, :]

    result = torch.zeros(n_experts, W.shape[0], W.shape[1], dtype=W.dtype)

    W_struct = U_struct @ torch.diag(S_struct) @ Vt_struct

    for e in range(n_experts):
        noise = torch.randn_like(S_res) * S_res * svd_scale
        S_e = S_res + noise

        W_res = U_res @ torch.diag(S_e) @ Vt_res

        result[e] = (W_struct + W_res).to(W.dtype)

    return result


def init_svd(dense, moe, cfg, k=8, svd_scale=0.5):
    """Initialize experts via SVD decomposition with residual sampling.

    For each dense FFN matrix, the top-k singular values form a shared
    structural component, while the residual singular values are perturbed
    independently per expert to create diversity.

    Since expert_dim = dense_dim (14336) in the Mixtral architecture,
    each expert receives the full reconstruction without row truncation.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``.
        k: Number of top singular values to keep as structural.
            Defaults to 8.
        svd_scale: Noise scale for residual perturbation. Defaults to 0.5.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]

    for i in range(n_layers):
        print(f"  SVD init layer {i}/{n_layers}...")
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts

        gate_init = _svd_init_matrix(dl.mlp.gate_proj.weight, n_experts, k, svd_scale)
        up_init = _svd_init_matrix(dl.mlp.up_proj.weight, n_experts, k, svd_scale)
        down_init = _svd_init_matrix(dl.mlp.down_proj.weight, n_experts, k, svd_scale)

        experts.gate_up_proj[:, :expert_dim, :].copy_(gate_init)
        experts.gate_up_proj[:, expert_dim:, :].copy_(up_init)
        experts.down_proj.copy_(down_init)


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
