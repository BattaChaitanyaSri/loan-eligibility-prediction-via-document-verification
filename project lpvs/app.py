from flask import Flask, render_template, request, session, jsonify
import os, uuid, shutil, stat, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prediction import get_loan_eligibility_prediction, get_llamaindex_result

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)
app.secret_key = "super-secret-key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# MODEL PATHS
# =====================================================
BASE_MODEL_PATH = r"P:\loan prediction via sml\models"
LORA_PATH = r"P:\loan prediction via sml\lora_output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# GLOBAL MODEL CACHE
# =====================================================
model = None
tokenizer = None

# =====================================================
# LOAD MODEL (WITH LORA MERGE)
# =====================================================
def load_model_once():
    global model, tokenizer

    if model is not None:
        return model, tokenizer

    print("🔄 Loading Qwen + LoRA model (ONE TIME)...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<END_OF_DECISION>"]})

    # Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    base_model.resize_token_embeddings(len(tokenizer))

    # 🔥 Load LoRA
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

    # 🔥 Merge LoRA into base model (faster inference)
    model = lora_model.merge_and_unload()

    model.to(DEVICE)
    model.eval()

    print("✅ Model loaded & merged successfully")
    return model, tokenizer


# =====================================================
# HELPERS
# =====================================================
def remove_folder_safely(path):
    def onerror(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=onerror)
        except Exception:
            time.sleep(0.2)
            shutil.rmtree(path, onerror=onerror, ignore_errors=True)


def loan_user_data_modification(user_data):
    marital_status = "Married" if user_data["marital_status"] == "Yes" else "Unmarried"

    dependents_raw = user_data["dependents"]
    if dependents_raw == "3+" or int(dependents_raw) >= 3:
        dependents = "High"
    elif int(dependents_raw) == 2:
        dependents = "Moderate"
    else:
        dependents = "Low"

    coapplicant_income = "Yes" if float(user_data["coapplicant_income"]) > 0 else "No"
    cibil_score = "Good" if int(user_data["cibil_score"]) >= 750 else "Bad"

    total_income = float(user_data["income"]) + float(user_data["coapplicant_income"])
    usable_income = total_income * 0.5

    loan_amount = float(user_data["loan_amount"])
    loan_period = float(user_data["loan_period"])

    emi_ratio = (loan_amount / loan_period) / usable_income if usable_income > 0 else 999

    if emi_ratio <= 1:
        emi_repayment_capacity = "High"
    elif emi_ratio <= 1.5:
        emi_repayment_capacity = "Moderate"
    else:
        emi_repayment_capacity = "Low"

    return {
        "gender": user_data["gender"],
        "coapplicant_income": coapplicant_income,
        "marital_status": marital_status,
        "dependents": dependents,
        "cibil_score": cibil_score,
        "employment_status": user_data["employment_status"],
        "emi_repayment_capacity": emi_repayment_capacity,
        "place_of_living": user_data["property_location"],
    }


def document_verification(user_data, folder):
    return get_llamaindex_result(user_data, folder)


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit-step1", methods=["POST"])
def submit_step1():
    session["user_data"] = dict(request.form)
    session["upload_id"] = str(uuid.uuid4())
    return jsonify({"success": True})


@app.route("/submit-step2", methods=["POST"])
def submit_step2():
    user_data = session.get("user_data")
    upload_id = session.get("upload_id")

    if not user_data or not upload_id:
        return "Session expired. Please start again."

    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], upload_id)
    remove_folder_safely(user_folder)
    os.makedirs(user_folder, exist_ok=True)

    for key in ["gov_id", "salary_slip", "cibil"]:
        file = request.files.get(key)
        if file:
            file.save(os.path.join(user_folder, file.filename))

    updated_user_data = loan_user_data_modification(user_data)
    updated_user_data["document_verification_status"] = document_verification(user_data, user_folder)

    # 🔥 Load model safely (only once)
    model, tokenizer = load_model_once()

    result = get_loan_eligibility_prediction(model, tokenizer, updated_user_data)

    return render_template(
        "result.html",
        reasoning=result["reasoning"],
        decision=result["decision"],
        user_data=user_data
    )


# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False
    )