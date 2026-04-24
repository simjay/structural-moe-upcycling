"""Sanity check: convert dense Qwen1.5-1.8B to MoE architecture.

Copies attention, embeddings, and layer norms directly.  Sets
shared_expert_intermediate_size = 5504 to match the dense FFN exactly
(no padding needed).  Initializes the 60 routing experts by partitioning
the dense FFN into 4 row-slices and replicating 15x.  Verifies the
resulting MoE model can run inference on the GPU.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DENSE_MODEL = "Qwen/Qwen1.5-1.8B"
MOE_REF = "Qwen/Qwen1.5-MoE-A2.7B"


def main():
    dense_cfg = AutoConfig.from_pretrained(DENSE_MODEL)
    moe_cfg = AutoConfig.from_pretrained(MOE_REF)

    # Override shared expert size to match dense FFN exactly
    moe_cfg.shared_expert_intermediate_size = dense_cfg.intermediate_size

    print("=== Architecture ===")
    print(f"Dense: hidden={dense_cfg.hidden_size}, ffn={dense_cfg.intermediate_size}, "
          f"layers={dense_cfg.num_hidden_layers}")
    print(f"MoE:   hidden={moe_cfg.hidden_size}, "
          f"experts={moe_cfg.num_experts}x{moe_cfg.moe_intermediate_size}, "
          f"shared={moe_cfg.shared_expert_intermediate_size}, "
          f"active={moe_cfg.num_experts_per_tok}/tok")

    print("\nLoading dense model...")
    dense = AutoModelForCausalLM.from_pretrained(
        DENSE_MODEL, dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)

    print("Creating empty MoE model (bf16)...")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    moe = AutoModelForCausalLM.from_config(moe_cfg)
    torch.set_default_dtype(prev_dtype)

    n_layers = dense_cfg.num_hidden_layers
    n_experts = moe_cfg.num_experts                        # 60
    expert_dim = moe_cfg.moe_intermediate_size             # 1408
    dense_dim = dense_cfg.intermediate_size                # 5504

    # Partition scheme: split dense FFN rows into chunks of expert_dim
    n_chunks = (dense_dim + expert_dim - 1) // expert_dim  # ceil(5504/1408) = 4
    n_replicas = (n_experts + n_chunks - 1) // n_chunks    # ceil(60/4) = 15

    print(f"Expert init: {n_chunks} partitions of {expert_dim} rows, "
          f"replicated {n_replicas}x = {n_chunks * n_replicas} experts")

    print("Copying weights...")
    with torch.no_grad():
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

            # Shared expert: exact copy (dimensions match)
            se = ml.mlp.shared_expert
            se.gate_proj.weight.copy_(dl.mlp.gate_proj.weight)
            se.up_proj.weight.copy_(dl.mlp.up_proj.weight)
            se.down_proj.weight.copy_(dl.mlp.down_proj.weight)

            # Routing experts: partition dense FFN into row-slices, replicate.
            # transformers >=5.x fuses gate_proj and up_proj into one tensor:
            #   gate_up_proj: (num_experts, 2*expert_dim, hidden_dim)
            #   down_proj:    (num_experts, hidden_dim, expert_dim)
            experts = ml.mlp.experts
            for e in range(n_experts):
                chunk_idx = e % n_chunks
                row_start = chunk_idx * expert_dim
                row_end = min(row_start + expert_dim, dense_dim)
                actual_rows = row_end - row_start

                experts.gate_up_proj[e, :actual_rows].copy_(
                    dl.mlp.gate_proj.weight[row_start:row_end])
                experts.gate_up_proj[e, expert_dim:expert_dim + actual_rows].copy_(
                    dl.mlp.up_proj.weight[row_start:row_end])
                experts.down_proj[e, :, :actual_rows].copy_(
                    dl.mlp.down_proj.weight[:, row_start:row_end])

                if actual_rows < expert_dim:
                    experts.gate_up_proj[e, actual_rows:expert_dim].zero_()
                    experts.gate_up_proj[e, expert_dim + actual_rows:].zero_()
                    experts.down_proj[e, :, actual_rows:].zero_()

    del dense
    torch.cuda.empty_cache()

    print("\nMoving MoE model to GPU...")
    moe = moe.to("cuda")

    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = moe.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")

    params = sum(p.numel() for p in moe.parameters())
    print(f"\nMoE total params: {params:,} ({params / 1e9:.1f}B)")
    print("Upcycling test passed!")


if __name__ == "__main__":
    main()
