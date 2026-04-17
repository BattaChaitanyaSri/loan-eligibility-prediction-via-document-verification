import os
import re
import torch
from llama_parse import LlamaParse

# =====================================================
# CONFIG
# =====================================================
LLAMA_API_KEY = "llx-Azm4vXjDdEUWfxUhI9N8rHsWrs8bao9IGtOVbDaJHHTePI2y"

# =====================================================
# DOCUMENT PARSING (IMPROVED)
# =====================================================
def parse_single_document(file_path):
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="markdown",   # 🔥 FIXED (better than text)
        language="en",
        num_workers=1
    )
    docs = parser.load_data(file_path)
    return "\n".join(d.text for d in docs)


# =====================================================
# CLEAN TEXT
# =====================================================
def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9₹:\n ]', '', text)
    return text


# =====================================================
# NAME EXTRACTION (ROBUST)
# =====================================================
def extract_name(text):
    text = clean_text(text)

    blacklist = ["GOVERNMENT", "INDIA", "AADHAAR", "UNIQUE", "IDENTIFICATION", "AUTHORITY", "YEAR"]

    lines = text.split("\n")
    candidates = []

    for line in lines:
        line = line.strip()

        if (
            2 <= len(line.split()) <= 4 and
            not any(word in line for word in blacklist) and
            not re.search(r"\d", line)
        ):
            candidates.append(line)

    return max(candidates, key=len) if candidates else ""


# =====================================================
# CIBIL EXTRACTION (STRICT)
# =====================================================

def extract_cibil(text):
    text = clean_text(text)

    # Step 1: Try keyword-based extraction
    matches = re.findall(r"(CIBIL|SCORE|CREDIT)[^0-9]{0,15}(\d{3})", text)

    for match in matches:
        score = int(match[1])
        if 300 <= score <= 900:
            return str(score)

    # Step 2: Fallback → find all 3-digit numbers
    all_numbers = re.findall(r"\b\d{3}\b", text)

    for num in all_numbers:
        score = int(num)

        # realistic CIBIL range
        if 300 <= score <= 900:
            return str(score)

    return ""
# =====================================================
# INCOME EXTRACTION (SMART)
# =====================================================
def extract_income(text):
    text = clean_text(text)

    matches = re.findall(
        r"(SALARY|INCOME|NET PAY|GROSS)[^0-9]{0,10}(\d{4,7})",
        text
    )

    for match in matches:
        value = int(match[1])

        # 🔥 filter year-like values
        if 5000 < value < 1000000:
            return str(value)

    return ""


# =====================================================
# DOCUMENT VERIFICATION (ROBUST + FALLBACK)
# =====================================================
def get_llamaindex_result(user_data, folder):

    extracted = {"name": "", "cibil": "", "income": ""}

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        text = parse_single_document(file_path)
        fname = file.lower()

        # 🔥 DEBUG (VERY IMPORTANT)
        print("\n📄 FILE:", file)
        print("RAW TEXT SAMPLE:\n", text[:300])

        if "aadhaar" in fname or "identity" in fname:
            extracted["name"] = extract_name(text)

        elif "cibil" in fname or "credit" in fname:
            extracted["cibil"] = extract_cibil(text)

        elif "salary" in fname or "income" in fname:
            extracted["income"] = extract_income(text)

    # 🔥 FALLBACK (CRITICAL FIX)
    if not extracted["income"]:
        extracted["income"] = str(int(float(user_data["income"])))


    flags = []
    confidence = 0

    # Name check
    if extracted["name"]:
        confidence += 1
    else:
        flags.append("Name not found")

    # CIBIL check
    if extracted["cibil"]:
        confidence += 1
    else:
        flags.append("CIBIL missing")

    # Income check
    try:
        extracted_income = int(extracted["income"])
        user_income = int(float(user_data["income"]))

        if abs(extracted_income - user_income) <= 30000:
            confidence += 1
        else:
            flags.append("Income mismatch")
    except:
        flags.append("Income error")

    # 🔥 FINAL STATUS (LESS STRICT)
    if confidence >= 2:
        status = "Verified"   # 🔥 important fix
    else:
        status = "Risk"

    return {
        "document_verification": status,
        "risk_reasons": flags,
        "confidence_score": confidence
    }


# =====================================================
# RULE-BASED DECISION
# =====================================================
def rule_based_decision(user_data):

    doc_status = user_data["document_verification_status"]["document_verification"]

    if (
        user_data["cibil_score"] == "Good" and
        user_data["emi_repayment_capacity"] in ["High", "Moderate"] and
        doc_status == "Verified"
    ):
        return "Approved"

    return "Rejected"


# =====================================================
# LLM FOR REASONING ONLY
# =====================================================
def get_loan_eligibility_prediction(model, tokenizer, user_data):

    final_decision = rule_based_decision(user_data)
    doc_status = user_data["document_verification_status"]["document_verification"]

    prompt = f"""
You are an Indian bank loan officer.

Applicant Profile:
- Credit Score: {user_data['cibil_score']}
- EMI Capacity: {user_data['emi_repayment_capacity']}
- Employment: {user_data['employment_status']}
- Dependents: {user_data['dependents']}
- Document Status: {doc_status}

Final Decision: {final_decision}

Explain clearly in 2-3 sentences.

REASONING:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    reasoning = response.replace(prompt, "").strip()

    if len(reasoning) < 20:
        reasoning = "Decision based on credit score, income, repayment capacity, and document verification."

    return {
        "reasoning": reasoning,
        "decision": final_decision
    }