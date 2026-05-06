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


def init_gaussian(dense, moe, cfg, sigma=0.1):
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
            Defaults to 0.1.
    """
    init_direct(dense, moe, cfg)

    for i in range(cfg["n_layers"]):
        experts = moe.model.layers[i].mlp.experts
        std = sigma * experts.gate_up_proj.abs().mean()
        experts.gate_up_proj.add_(torch.randn_like(experts.gate_up_proj) * std)
        std = sigma * experts.down_proj.abs().mean()
        experts.down_proj.add_(torch.randn_like(experts.down_proj) * std)


def _svd_perturb_chunk(W_chunk, k, svd_scale):
    """Apply SVD-residual perturbation to a single weight chunk.

    Decomposes W_chunk = U @ diag(S) @ Vt, keeps top-k singular values
    unchanged (structural), and adds uniform additive noise to the remaining
    residual singular values.

    Args:
        W_chunk: Weight matrix chunk (e.g. 1408 x 2048 for gate/up,
            or 2048 x 1408 for down_proj).
        k: Number of top singular values to preserve as structural.
        svd_scale: Noise scale relative to mean residual singular value
            magnitude (mirrors how Gaussian uses sigma * mean(|W|)).

    Returns:
        Perturbed weight chunk with same shape and dtype as input.
    """
    U, S, Vt = torch.linalg.svd(W_chunk.float(), full_matrices=False)
    rank = min(k, len(S))

    S_res = S[rank:]
    std = svd_scale * S_res.abs().mean()
    S_perturbed = S_res + torch.randn_like(S_res) * std

    S_full = torch.cat([S[:rank], S_perturbed])
    W_new = U @ torch.diag(S_full) @ Vt
    return W_new.to(W_chunk.dtype)


def init_svd(dense, moe, cfg, k=8, svd_scale=0.5):
    """Initialize experts via partition + SVD-residual perturbation.

    Mirrors the partition-and-replicate structure of init_direct/init_gaussian:
    splits the dense FFN into chunks, cycles experts across chunks, and applies
    SVD-residual perturbation independently to each expert's chunk. This ensures
    experts within the same partition group share the same structural component
    but differ in their residual singular values.

    Args:
        dense: The dense source model.
        moe: The MoE target model.
        cfg: Dict with keys ``n_layers``, ``n_experts``, ``expert_dim``,
            ``dense_dim``.
        k: Number of top singular values to keep as structural.
            Defaults to 8.
        svd_scale: Noise scale for residual perturbation (relative to mean
            residual singular value magnitude). Defaults to 0.5.
    """
    n_layers = cfg["n_layers"]
    n_experts = cfg["n_experts"]
    expert_dim = cfg["expert_dim"]
    dense_dim = cfg["dense_dim"]
    n_chunks = math.ceil(dense_dim / expert_dim)

    for i in range(n_layers):
        print(f"  SVD init layer {i}/{n_layers}...")
        dl = dense.model.layers[i]
        experts = moe.model.layers[i].mlp.experts

        gate_w = dl.mlp.gate_proj.weight
        up_w = dl.mlp.up_proj.weight
        down_w = dl.mlp.down_proj.weight

        for e in range(n_experts):
            chunk_idx = e % n_chunks
            row_start = chunk_idx * expert_dim
            row_end = min(row_start + expert_dim, dense_dim)
            actual = row_end - row_start

            gate_chunk = _svd_perturb_chunk(gate_w[row_start:row_end], k, svd_scale)
            up_chunk = _svd_perturb_chunk(up_w[row_start:row_end], k, svd_scale)
            down_chunk = _svd_perturb_chunk(down_w[:, row_start:row_end].T, k, svd_scale).T

            experts.gate_up_proj[e, :actual].copy_(gate_chunk)
            experts.gate_up_proj[e, expert_dim:expert_dim + actual].copy_(up_chunk)
            experts.down_proj[e, :, :actual].copy_(down_chunk)

            if actual < expert_dim:
                experts.gate_up_proj[e, actual:expert_dim].zero_()
                experts.gate_up_proj[e, expert_dim + actual:].zero_()
                experts.down_proj[e, :, actual:].zero_()


def upcycle(method, output_dir, sigma=0.1, k=8, svd_scale=0.5):
    """Build an MoE model from the dense checkpoint using the given init method.

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
    parser = argparse.ArgumentParser(description="Dense-to-MoE upcycling")
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
