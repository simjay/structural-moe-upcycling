"""Sanity check: load Qwen1.5-1.8B and generate a few tokens."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen1.5-1.8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
