"""Sanity check: run 10 SFT steps on a tiny Mixtral-style MoE model.

Creates a minimal Mixtral config from scratch (2 layers, 2 experts, small dims)
to verify that the training loop works with:
- Expert FFN weights trained (unfrozen)
- Router trained (unfrozen)
- Attention frozen
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MAX_SEQ = 128


def build_tiny_mixtral():
    """Create a tiny Mixtral-style MoE model for testing (no network needed)."""
    config = MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        vocab_size=32000,
    )

    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float32)
    return model, config


def test_moe_training():
    print("Building tiny Mixtral MoE model...")
    model, config = build_tiny_mixtral()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Freezing all, then unfreezing experts + router...")
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "experts" in name or "mlp.gate" in name:
            param.requires_grad = True

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    has_expert = any("experts" in n for n in trainable_names)
    has_router = any("gate" in n and "experts" not in n for n in trainable_names)
    attn_frozen = all("self_attn" not in n for n in trainable_names)

    assert has_expert, "Expected expert FFN weights to be trainable"
    assert has_router, "Expected router (gate) to be trainable"
    assert attn_frozen, "Expected attention to be frozen"

    print(f"  Expert FFN:     {has_expert}")
    print(f"  Router (gate):  {has_router}")
    print(f"  Attention frozen: {attn_frozen}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    samples = [{"text": f"Problem: {a}+{b}=?\nAnswer: {a + b}"}
               for a in range(1, 9) for b in range(1, 9)]
    ds = Dataset.from_list(samples)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir="/tmp/test_moe_train",
            max_steps=10,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            logging_steps=1,
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",
            dataset_text_field="text",
            max_length=MAX_SEQ,
        ),
    )

    print("Training for 10 steps...")
    result = trainer.train()

    print(f"\nFinal loss: {result.training_loss:.4f}")
    print(f"Steps: {result.global_step}")
    assert result.global_step == 10, f"Expected 10 steps, got {result.global_step}"
    print("MoE training test passed!")


if __name__ == "__main__":
    test_moe_training()
