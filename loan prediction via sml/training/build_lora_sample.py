import json

INPUT_FILE = r"P:\loan prediction via sml\training\derived_samples.jsonl"      # your current samples
OUTPUT_FILE = "lora_train.jsonl"        # for training

def build_text(sample):
    text = f"""You are a conservative bank loan officer.

Applicant Risk Profile:
- Gender: {sample['gender']}
- Marital Status: {sample['marital_status']}
- Dependents: {sample['dependents']}
- Employment Status: {sample['emp_status']}
- Coapplicant Income: {sample['coapplicant_income']}
- Place of Living: {sample['place_of_living']}
- Credit Score Quality: {sample['cibil_score']}
- EMI Repayment Capacity: {sample['emi_repayment_capacity']}
- Document Verification Status: {sample['document_verification']}

REASONING:
{sample['reasoning']}

DECISION:
{sample['decision']}
"""
    return text.strip()


with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        sample = json.loads(line)
        text = build_text(sample)
        fout.write(json.dumps({"text": text}) + "\n")

print("✅ LoRA training file created:", OUTPUT_FILE)
