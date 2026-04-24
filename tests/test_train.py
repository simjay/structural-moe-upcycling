"""Sanity check: run 10 LoRA SFT steps on dense Qwen1.5-1.8B.

Uses a tiny synthetic math dataset to verify the full training loop
(forward, loss, backward, optimizer step) completes without errors.
"""

from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL = "Qwen/Qwen1.5-1.8B"
MAX_SEQ = 512

print("Loading model...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=False
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

samples = [{"text": f"Problem: {a}+{b}=?\nAnswer: {a + b}"}
           for a in range(1, 9) for b in range(1, 9)]
ds = Dataset.from_list(samples)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="/tmp/test_train",
        max_steps=10,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
    ),
)

print("Training for 10 steps...")
result = trainer.train()

print(f"\nFinal loss: {result.training_loss:.4f}")
print(f"Steps: {result.global_step}")
print("Training test passed!")
