import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================
# PATHS (LOCAL WINDOWS)
# ============================
BASE_MODEL_PATH = r"P:\loan prediction via sml\models"
LORA_PATH = r"P:\loan prediction via sml\lora_output"

DEVICE = "cpu"   # 🔒 force CPU on Windows

# ============================
# TOKENIZER
# ============================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<END_OF_DECISION>"]
})



# ============================
# LOAD BASE MODEL (NO 4-BIT)
# ============================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32   # 🔒 CPU safe
)
model.resize_token_embeddings(len(tokenizer))

# ============================
# LOAD LoRA ADAPTER
# ============================
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("✅ Qwen2.5 + LoRA loaded successfully")

# ============================
# TEST PROMPT (MATCH TRAINING)
# ============================
prompt = """You are a conservative bank loan officer.

Applicant Risk Profile:
- Gender: Male
- Marital Status: Married
- Dependents: High
- Employment Status: Employed
- Coapplicant Income: No
- Place of Living: Urban
- Credit Score Quality: Good
- EMI Repayment Capacity: High
- Document Verification Status: Risk

REASONING:
"""

inputs = tokenizer(prompt, return_tensors="pt")

# ============================
# GENERATE
# ============================
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ============================
# 🔒 HARD STOP AT END TOKEN
# ============================
if "<END_OF_DECISION>" in response:
    response = response.split("<END_OF_DECISION>")[0] + "<END_OF_DECISION>"

print("\n===== CLEAN MODEL RESPONSE =====\n")
print(response)