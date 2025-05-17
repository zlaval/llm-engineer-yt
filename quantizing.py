import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

LLM = "google/gemma-3-1b-it"

qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(LLM )#, quantization_config=qc)

m = model.get_memory_footprint() / 1e6
print(f"Model memory footprint: {m:.2f} MB")
print(model)
