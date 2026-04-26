"""Dense-to-MoE upcycling with three expert initialization strategies.

Loads Qwen1.5-1.8B, creates a Qwen1.5-MoE-A2.7B architecture, copies all
shared weights (attention, embeddings, norms, shared expert), then initializes
the 60 routing experts using one of three methods:

Methods:
    direct:   Partition dense FFN into row-slices, replicate 15x.
    gaussian: Partition + replicate + Gaussian noise.
    svd:      SVD decomposition with structural/residual split.
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DENSE_MODEL = "Qwen/Qwen1.5-1.8B"
MOE_REF = "Qwen/Qwen1.5-MoE-A2.7B"


def _copy_shared_weights(dense, moe, n_layers):
    """Copy embeddings, attention, layer norms, and shared expert.

    Args:
        dense: The dense Qwen1.5-1.8B model.
        moe: The empty MoE model to populate.
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

        se = ml.mlp.shared_expert
        se.gate_proj.weight.copy_(dl.mlp.gate_proj.weight)
        se.up_proj.weight.copy_(dl.mlp.up_proj.weight)
        se.down_proj.weight.copy_(dl.mlp.down_proj.weight)


def _partition_and_replicate(gate_w, up_w, down_w, experts, n_experts,
                             expert_dim, dense_dim):
    """Fill fused expert tensors by partitioning dense FFN into row-slices.

    Splits each dense FFN weight matrix into ceil(dense_dim / expert_dim)
    non-overlapping chunks and cycles through them to populate all experts.
    The last chunk is zero-padded if smaller than expert_dim.

    Args:
        gate_w: Dense gate_proj weight, shape ``(dense_dim, hidden)``.
        up_w: Dense up_proj weight, shape ``(dense_dim, hidden)``.
        down_w: Dense down_proj weight, shape ``(hidden, dense_dim)``.
        experts: Fused ``Qwen2MoeExperts`` module to write into.
        n_experts: Total number of routing experts (60).
        expert_dim: Intermediate size per expert (1408).
        dense_dim: Dense FFN intermediate size (5504).
    """
    n_chunks = math.ceil(dense_dim / expert_dim)
    for e in range(n_experts):
        chunk_idx = e % n_chunks
        row_start = chunk_idx * expert_dim
        row_end = min(row_start + expert_dim, dense_dim)
        actual = row_end - row_start

        experts.gate_up_proj[e, :actual].copy_(gate_w[row_start:row_end])
        experts.gate_up_proj[e, expert_dim:expert_dim + actual].copy_(
            up_w[row_start:row_end])
        experts.down_proj[e, :, :actual].copy_(down_w[:, row_start:row_end])

        if actual < expert_dim:
            experts.gate_up_proj[e, actual:expert_dim].zero_()
            experts.gate_up_proj[e, expert_dim + actual:].zero_()
            experts.down_proj[e, :, actual:].zero_()


def init_direct(dense, moe, cfg):
    """Initialize experts by partitioning dense FFN row-slices and replicating.

    All 60 experts receive identical copies (cycled across 4 partitions).

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``,
            ``dense_dim``.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]
    dense_dim = cfg["dense_dim"]

    for i in range(n_layers):
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts
        _partition_and_replicate(
            dl.mlp.gate_proj.weight, dl.mlp.up_proj.weight,
            dl.mlp.down_proj.weight, experts,
            n_experts, expert_dim, dense_dim)


def init_gaussian(dense, moe, cfg, sigma=0.01):
    """Initialize experts via partition + replicate, then add Gaussian noise.

    First applies the direct-copy initialization, then perturbs each expert's
    weights with i.i.d. Gaussian noise scaled by the mean absolute weight
    magnitude.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``,
            ``dense_dim``.
        sigma: Noise scale relative to mean absolute weight magnitude.
            Defaults to 0.01.
    """
    init_direct(dense, moe, cfg)

    for i in range(cfg["n_layers"]):
        experts = moe.model.layers[i].mlp.experts
        std = sigma * experts.gate_up_proj.abs().mean()
        experts.gate_up_proj.add_(torch.randn_like(experts.gate_up_proj) * std)
        std = sigma * experts.down_proj.abs().mean()
        experts.down_proj.add_(torch.randn_like(experts.down_proj) * std)


def _svd_init_matrix(W, expert_dim, n_experts, k):
    """Initialize expert matrices from a dense weight via SVD residual sampling.

    Decomposes W = U @ diag(S) @ Vt, keeps top-k singular values as the
    shared structural component, and samples perturbed residuals per expert.

    Args:
        W: Dense weight matrix, shape ``(out_features, in_features)``,
            e.g. ``(5504, 2048)`` for gate/up_proj.
        expert_dim: Number of output rows per expert (1408).
        n_experts: Total number of experts to initialize (60).
        k: Number of top singular values to treat as structural.

    Returns:
        Tensor of shape ``(n_experts, expert_dim, in_features)``.
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    # U: (out, min(out,in)), S: (min(out,in),), Vt: (min(out,in), in)

    rank = min(k, len(S), expert_dim)
    S_struct = S[:rank]
    U_struct = U[:, :rank]
    Vt_struct = Vt[:rank, :]

    S_res = S[rank:]
    U_res = U[:, rank:]

    result = torch.zeros(n_experts, expert_dim, W.shape[1], dtype=W.dtype)

    for e in range(n_experts):
        noise = torch.randn_like(S_res) * S_res * 0.1
        S_e = S_res + noise

        W_struct = U_struct[:expert_dim, :] @ torch.diag(S_struct) @ Vt_struct
        n_res = min(expert_dim, U_res.shape[1])
        W_res = U_res[:expert_dim, :n_res] @ torch.diag(S_e[:n_res]) @ Vt[rank:rank + n_res, :]

        result[e] = (W_struct + W_res).to(W.dtype)

    return result


def _svd_init_down_matrix(W, expert_dim, n_experts, k):
    """Initialize expert down_proj matrices via SVD residual sampling.

    Same approach as ``_svd_init_matrix`` but transposed for down_proj,
    where the expert dimension is on the column axis rather than the row axis.

    Args:
        W: Dense down_proj weight, shape ``(hidden, dense_dim)``.
        expert_dim: Number of input columns per expert (1408).
        n_experts: Total number of experts to initialize (60).
        k: Number of top singular values to treat as structural.

    Returns:
        Tensor of shape ``(n_experts, hidden, expert_dim)``.
    """
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)

    rank = min(k, len(S), expert_dim)
    S_struct = S[:rank]
    U_struct = U[:, :rank]
    Vt_struct = Vt[:rank, :]

    S_res = S[rank:]
    Vt_res = Vt[rank:, :]

    result = torch.zeros(n_experts, W.shape[0], expert_dim, dtype=W.dtype)

    for e in range(n_experts):
        noise = torch.randn_like(S_res) * S_res * 0.1
        S_e = S_res + noise

        W_struct = U_struct @ torch.diag(S_struct) @ Vt_struct[:, :expert_dim]
        n_res = min(expert_dim, Vt_res.shape[0])
        W_res = U[:, rank:rank + n_res] @ torch.diag(S_e[:n_res]) @ Vt_res[:n_res, :expert_dim]

        result[e] = (W_struct + W_res).to(W.dtype)

    return result


def init_svd(dense, moe, cfg, k=256):
    """Initialize experts via SVD decomposition with residual sampling.

    For each dense FFN matrix, the top-k singular values form a shared
    structural component, while the residual singular values are perturbed
    independently per expert to create diversity.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``.
        k: Number of top singular values to keep as structural.
            Defaults to 256.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]

    for i in range(n_layers):
        print(f"  SVD init layer {i}/{n_layers}...")
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts

        gate_init = _svd_init_matrix(
            dl.mlp.gate_proj.weight, expert_dim, n_experts, k)
        up_init = _svd_init_matrix(
            dl.mlp.up_proj.weight, expert_dim, n_experts, k)
        down_init = _svd_init_down_matrix(
            dl.mlp.down_proj.weight, expert_dim, n_experts, k)

        experts.gate_up_proj[:, :expert_dim, :].copy_(gate_init)
        experts.gate_up_proj[:, expert_dim:, :].copy_(up_init)
        experts.down_proj.copy_(down_init)


def upcycle(method, output_dir, sigma=0.01, k=256):
    """Build an MoE model from the dense checkpoint using the given init method.

    Loads the dense model, creates an empty MoE shell, copies all shared
    weights, initializes routing experts, and saves the result to disk.

    Args:
        method: One of ``"direct"``, ``"gaussian"``, or ``"svd"``.
        output_dir: Path to save the upcycled model and tokenizer.
        sigma: Noise scale for the gaussian method. Defaults to 0.01.
        k: Number of structural singular values for the svd method.
            Defaults to 256.
    """
    print(f"=== Upcycling with method={method} ===\n")

    dense_cfg = AutoConfig.from_pretrained(DENSE_MODEL)
    moe_cfg = AutoConfig.from_pretrained(MOE_REF)
    moe_cfg.shared_expert_intermediate_size = dense_cfg.intermediate_size

    print(f"Dense: hidden={dense_cfg.hidden_size}, ffn={dense_cfg.intermediate_size}, "
          f"layers={dense_cfg.num_hidden_layers}")
    print(f"MoE:   experts={moe_cfg.num_experts}x{moe_cfg.moe_intermediate_size}, "
          f"shared={moe_cfg.shared_expert_intermediate_size}, "
          f"active={moe_cfg.num_experts_per_tok}/tok\n")

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
        n_experts=moe_cfg.num_experts,
        expert_dim=moe_cfg.moe_intermediate_size,
        dense_dim=dense_cfg.intermediate_size,
    )

    print("Copying shared weights (attention, embeddings, norms, shared expert)...")
    with torch.no_grad():
        _copy_shared_weights(dense, moe, cfg["n_layers"])

        print(f"Initializing experts ({method})...")
        if method == "direct":
            init_direct(dense, moe, cfg)
        elif method == "gaussian":
            init_gaussian(dense, moe, cfg, sigma=sigma)
        elif method == "svd":
            init_svd(dense, moe, cfg, k=k)
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
    parser = argparse.ArgumentParser(description="Dense-to-MoE upcycling")
    parser.add_argument("--method", required=True,
                        choices=["direct", "gaussian", "svd"])
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Noise scale for gaussian method")
    parser.add_argument("--k", type=int, default=256,
                        help="Number of structural singular values for svd method")
    args = parser.parse_args()
    upcycle(args.method, args.output, sigma=args.sigma, k=args.k)


if __name__ == "__main__":
    main()
