from datasets import load_dataset
import json

# =========================
# PATHS (EDIT THESE)
# =========================
INPUT_PATH = r"P:\loan prediction via sml\training\dataset.jsonl"
OUTPUT_PATH = r"P:\loan prediction via sml\training\updated_dataset.jsonl"

# =========================
# LOAD DATASET
# =========================
dataset = load_dataset("json", data_files=INPUT_PATH, split="train")

print("Original dataset size:", len(dataset))

# =========================
# CONVERT
# =========================
approved_count = 0
rejected_count = 0

with open(OUTPUT_PATH, "w") as f:
    for ex in dataset:
        # ---- Reasoning ----
        reasoning_list = ex.get("reasoning", [])
        if isinstance(reasoning_list, list):
            reasoning_text = " ".join(reasoning_list)
        else:
            reasoning_text = str(reasoning_list)

        # ---- Decision ----
        decision_obj = ex.get("decision", {})
        eligibility = decision_obj.get("eligibility", "").lower()

        if "not" in eligibility:
            decision_text = "Rejected"
            rejected_count += 1
        else:
            decision_text = "Approved"
            approved_count += 1

        # ---- Final training text ----
        text = (
            "REASONING:\n"
            f"{reasoning_text}\n\n"
            "DECISION:\n"
            f"{decision_text}"
        )

        f.write(json.dumps({"text": text}) + "\n")

print("✅ Conversion completed")
print("Approved samples:", approved_count)
print("Rejected samples:", rejected_count)
print("Saved to:", OUTPUT_PATH)
