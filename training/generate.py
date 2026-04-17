import random
import json

# Feature options
genders = ["Male", "Female"]
coapplicant_income = ["Yes", "No"]
marital_status = ["Married", "Unmarried"]
dependents = ["Low", "Moderate", "High"]
cibil_score = ["Good", "Bad"]
emp_status = ["Employed", "Self-Employed", "Unemployed"]
emi_capacity = ["Low", "Moderate", "High"]
place = ["Urban", "Semiurban", "Rural"]
doc_verification = ["Verified", "Risk"]

def generate_reasoning(data, decision):
    if decision == "Approved":
        return "Strong financial indicators, stable income, and verified documentation indicate low lending risk."
    else:
        return "Multiple risk factors such as poor credit history, low repayment capacity, or document issues increase default risk."

def make_decision(data):
    # Hard rejection rules
    if data["cibil_score"] == "Bad":
        return "Rejected"
    if data["document_verification"] == "Risk":
        return "Rejected"
    if data["emi_repayment_capacity"] == "Low":
        return "Rejected"
    if data["emp_status"] == "Unemployed":
        return "Rejected"

    # Soft scoring
    score = 0

    if data["cibil_score"] == "Good":
        score += 2
    if data["document_verification"] == "Verified":
        score += 2
    if data["emi_repayment_capacity"] == "High":
        score += 2
    if data["coapplicant_income"] == "Yes":
        score += 1
    if data["dependents"] == "High":
        score -= 1

    return "Approved" if score >= 3 else "Rejected"

def generate_sample():
    data = {
        "gender": random.choice(genders),
        "coapplicant_income": random.choice(coapplicant_income),
        "marital_status": random.choice(marital_status),
        "dependents": random.choice(dependents),
        "cibil_score": random.choice(cibil_score),
        "emp_status": random.choice(emp_status),
        "emi_repayment_capacity": random.choice(emi_capacity),
        "place_of_living": random.choice(place),
        "document_verification": random.choice(doc_verification)
    }

    decision = make_decision(data)
    reasoning = generate_reasoning(data, decision)

    data["reasoning"] = reasoning
    data["decision"] = decision

    return data

def generate_dataset(n=1000, output_file="loan_dataset.jsonl"):
    with open(output_file, "w") as f:
        for _ in range(n):
            sample = generate_sample()
            f.write(json.dumps(sample) + "\n")

    print(f"✅ Dataset with {n} samples saved to {output_file}")

if __name__ == "__main__":
    generate_dataset(1200)