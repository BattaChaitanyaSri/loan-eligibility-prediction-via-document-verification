import pandas as pd

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(r"P:\loan prediction via sml\datasets\Loan Eligibility Prediction.csv")
df.dropna(inplace=True)

derived_samples = []

for _, row in df.iterrows():

    # -------------------------
    # BASIC FIELDS
    # -------------------------
    gender = row["Gender"]

    marital_status = "Married" if row["Married"] == "Yes" else "Unmarried"

    if row["Dependents"] == "3+" or int(row["Dependents"]) >= 3:
        dependents = "High"
    elif int(row["Dependents"]) == 2:
        dependents = "Moderate"
    else :
        dependents = "Low"

    place_of_living = row["Property_Area"]

    coapplicant_income = 'Yes' if row["Coapplicant_Income"] > 0 else 'No'

    # -------------------------
    # EMPLOYMENT STATUS
    # -------------------------
    if row["Self_Employed"] == "Yes":
        emp_status = "Self-Employed"
    else:
        emp_status = "Employed"

    # -------------------------
    # CREDIT QUALITY
    # -------------------------
    if row["Credit_History"] == 1:
        cibil_score = "Good"
    else:
        cibil_score = "Bad"

    # -------------------------
    # EMI REPAYMENT CAPACITY
    # -------------------------
    applicant_income = float(row["Applicant_Income"])
    total_income = applicant_income + float(row["Coapplicant_Income"])

    usable_income = total_income * 0.5  # 50% rule

    loan_amount = float(row["Loan_Amount"])
    loan_term = float(row["Loan_Amount_Term"])

    if usable_income > 0 and loan_term > 0:
        emi = loan_amount / loan_term
        emi_ratio = emi / usable_income
    else:
        emi_ratio = 999  # force low capacity

    if emi_ratio <= 1.0:
        emi_repayment_capacity = "High"
    elif emi_ratio <= 1.5:
        emi_repayment_capacity = "Moderate"
    else:
        emi_repayment_capacity = "Low"

    # -------------------------
    # FINAL DERIVED SAMPLE
    # -------------------------
    derived_samples.append({
        "gender": gender,
        "coapplicant_income": coapplicant_income,
        "marital_status": marital_status,
        "dependents": dependents,
        "cibil_score": cibil_score,
        "emp_status": emp_status,
        "emi_repayment_capacity": emi_repayment_capacity,
        "place_of_living": place_of_living,
        "document_verification":None,
        "reasoning": None,
        "decision": None
    })

# =========================
# RESULT
# =========================
derived_df = pd.DataFrame(derived_samples)
print(derived_df.head())

import json

OUTPUT_FILE = "derived_samples.jsonl"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in derived_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(derived_samples)} samples to {OUTPUT_FILE}")

