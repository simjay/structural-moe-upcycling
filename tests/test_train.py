"""Sanity check: run 10 LoRA SFT steps on a tiny Mixtral-style MoE model.

Creates a minimal Mixtral config from scratch (2 layers, 2 experts, small dims)
to verify that the full training loop works with:
- LoRA on attention (target_modules)
- LoRA on fused expert parameters (target_parameters)
- Full training of the router (modules_to_save)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralConfig
from peft import LoraConfig, get_peft_model
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
        vocab_size=1000,
    )

    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float32)
    return model, config


def test_moe_lora_training():
    print("Building tiny Mixtral MoE model...")
    model, config = build_tiny_mixtral()

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_experts = 2
    lora_r = 8
    effective_r = max(1, lora_r // num_experts)

    print("Applying LoRA + target_parameters + modules_to_save...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters=[
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ],
        rank_pattern={
            "experts.gate_up_proj": effective_r,
            "experts.down_proj": effective_r,
        },
        modules_to_save=["mlp.gate"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    has_attn_lora = any("lora" in n and "self_attn" in n for n in trainable_names)
    has_expert_lora = any("experts" in n and "lora" in n for n in trainable_names)
    has_router = any("gate" in n and "experts" not in n for n in trainable_names)

    assert has_attn_lora, "Expected attention LoRA adapters to be trainable"
    assert has_expert_lora, "Expected expert LoRA adapters to be trainable"
    assert has_router, "Expected router (gate) to be trainable"

    print(f"  Attention LoRA: {has_attn_lora}")
    print(f"  Expert LoRA:    {has_expert_lora}")
    print(f"  Router (gate):  {has_router}")

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
    print("MoE LoRA training test passed!")


if __name__ == "__main__":
    test_moe_lora_training()
