"""Sanity check: load Qwen1.5-1.8B via Unsloth and generate a few tokens."""

from unsloth import FastModel

MODEL = "Qwen/Qwen1.5-1.8B"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
)

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
