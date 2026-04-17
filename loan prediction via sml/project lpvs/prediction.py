import os
import re
import torch
from difflib import SequenceMatcher
from llama_parse import LlamaParse

# =====================================================
# DOCUMENT PARSING (ONE BY ONE – SAFE)
# =====================================================
LLAMA_API_KEY = "llx-2Ls1IwoAHYv21BrSdLql9ngtx36KSJoXolps6p1hrZKvVdjK"


def parse_single_document(file_path):
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="text",
        language="en",
        num_workers=1  # 🔥 CRITICAL
    )
    docs = parser.load_data(file_path)
    return "\n".join(d.text for d in docs)


def extract_number(text):
    match = re.search(r"\d+", str(text))
    return int(match.group()) if match else None


def extract_name(text):
    match = re.search(r"name\s*[:\-]?\s*([A-Za-z ]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_cibil(text):
    match = re.search(r"(cibil|score)\D*(\d{3})", text, re.IGNORECASE)
    return match.group(2) if match else ""


def extract_income(text):
    match = re.search(r"₹?\s?([\d,]+)", text)
    return match.group(1) if match else ""


# =====================================================
# DOCUMENT VERIFICATION (NO LLM HERE)
# =====================================================
def get_llamaindex_result(user_data, folder):
    extracted = {
        "name": "",
        "cibil": "",
        "income": ""
    }

    for file in os.listdir(folder):
        text = parse_single_document(os.path.join(folder, file))
        fname = file.lower()

        if "aadhaar" in fname or "identity" in fname:
            extracted["name"] = extract_name(text)

        elif "cibil" in fname or "credit" in fname:
            extracted["cibil"] = extract_cibil(text)

        elif "salary" in fname or "income" in fname:
            extracted["income"] = extract_income(text)

    flags = []

    if not extracted["name"]:
        flags.append("Name not found in identity document")

    if not extracted["cibil"]:
        flags.append("CIBIL score missing")

    if not extracted["income"]:
        flags.append("Income document missing")

    if flags:
        return {
            "document_verification": "Risk",
            "risk_reasons": flags,
            "confidence_score": 0
        }

    return {
        "document_verification": "Verified",
        "risk_reasons": [],
        "confidence_score": 3
    }


# =====================================================
# LLM DECISION ENGINE (FIXED AND ROBUST)
# =====================================================
def get_loan_eligibility_prediction(model, tokenizer, user_data):

    # 🔥 FLATTEN document verification
    doc_status = user_data["document_verification_status"]["document_verification"]

    # 🔥 IMPROVED PROMPT - More explicit instructions
    prompt = f"""You are a conservative Indian bank loan officer analyzing a loan application.

Applicant Profile:
- Gender: {user_data['gender']}
- Marital Status: {user_data['marital_status']}
- Dependents: {user_data['dependents']}
- Employment Status: {user_data['employment_status']}
- Coapplicant Income: {user_data['coapplicant_income']}
- Place of Living: {user_data['place_of_living']}
- Credit Score Quality: {user_data['cibil_score']}
- EMI Repayment Capacity: {user_data['emi_repayment_capacity']}
- Document Verification Status: {doc_status}

Analyze this application and provide your decision. Your response must follow this exact format:

REASONING: [Provide 2-3 sentences explaining your decision based on the applicant's credit score, repayment capacity, employment status, and document verification]

DECISION: [Either "Approved" or "Rejected"]

<END_OF_DECISION>"""

    # 🔥 TOKENIZE WITH PROPER SETTINGS
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 🔥 IMPROVED GENERATION PARAMETERS
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,  # Increased from 150
            min_new_tokens=50,   # Force minimum output
            do_sample=True,      # Enable sampling for more natural text
            temperature=0.7,     # Add some randomness but stay controlled
            top_p=0.9,          # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )

    # 🔥 DECODE RESPONSE
    response = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # 🔥 DEBUG: Print the raw response (you can comment this out later)
  #remove thiis after debugging  
    print("\n" + "="*60)
    print("RAW MODEL OUTPUT:")
    print("="*60)
    print(response)
    print("="*60 + "\n")

    return extract_reasoning_and_decision(response, prompt)


# =====================================================
# IMPROVED EXTRACTION (MORE ROBUST)
# =====================================================
def extract_reasoning_and_decision(text, original_prompt=""):
    
    # Remove the original prompt from the response to get only the generated part
    if original_prompt:
        text = text.replace(original_prompt, "").strip()
    
    # 🔥 MULTIPLE PATTERN ATTEMPTS
    
    # Pattern 1: Standard format with REASONING: and DECISION:
    reasoning_match = re.search(
        r"REASONING:\s*(.*?)(?=DECISION:|\<END_OF_DECISION\>|$)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    
    decision_match = re.search(
        r"DECISION:\s*(Approved|Rejected)",
        text,
        re.IGNORECASE
    )
    
    # Pattern 2: Try to find decision anywhere in text if not found
    if not decision_match:
        decision_match = re.search(
            r"\b(Approved|Rejected)\b",
            text,
            re.IGNORECASE
        )
    
    # 🔥 EXTRACT OR USE DEFAULTS
    reasoning = ""
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        # Clean up reasoning
        reasoning = re.sub(r'\<END_OF_DECISION\>.*', '', reasoning, flags=re.DOTALL)
        reasoning = reasoning.strip()
    
    # If reasoning is still empty, try to extract any meaningful text
    if not reasoning:
        # Look for text before DECISION or before Approved/Rejected
        pre_decision = re.search(
            r"(.*?)(?:DECISION:|Approved|Rejected)",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if pre_decision:
            potential_reasoning = pre_decision.group(1).strip()
            # Remove common artifacts
            potential_reasoning = re.sub(r'^(REASONING:|reasoning:)\s*', '', potential_reasoning, flags=re.IGNORECASE)
            if len(potential_reasoning) > 20:  # Only use if substantial
                reasoning = potential_reasoning
    
    # 🔥 FALLBACK REASONING if still empty
    if not reasoning or len(reasoning) < 20:
        reasoning = "The application was evaluated based on creditworthiness, income stability, repayment capacity, and document verification status."
    
    # 🔥 EXTRACT DECISION
    decision = "Rejected"  # Default to conservative decision
    if decision_match:
        decision = decision_match.group(1).title()
    
    # Clean up reasoning - remove extra whitespace and newlines
    reasoning = " ".join(reasoning.split())
    
    return {
        "reasoning": reasoning,
        "decision": decision
    }