import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\bharg\OneDrive\Desktop\loan prediction via sml\models\models--microsoft--phi-1_5\snapshots\77aa61eeac94fbf33d492b9f2744c98b42d5b5eb"
DATA_PATH = r"C:\Users\bharg\OneDrive\Desktop\loan prediction via sml\training\final_output_only_samples.jsonl"
OUTPUT_DIR = r"C:\Users\bharg\OneDrive\Desktop\loan prediction via sml\lora_output"

MAX_LENGTH = 256
LR = 1e-4
EPOCHS = 6
GRAD_ACCUM = 8
WEIGHT_DECAY = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model.config.use_cache = False
model.to(DEVICE)

# =========================
# LoRA
# =========================
lora_config = LoraConfig(
    r=16,                     # 🔥 increased rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# DATASET
# =========================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = [
        t if m == 1 else -100
        for t, m in zip(tokens["input_ids"], tokens["attention_mask"])
    ]
    return tokens

dataset = dataset.map(tokenize, remove_columns=["text"])

# =========================
# OPTIMIZER
# =========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# =========================
# TRAINING LOOP
# =========================
model.train()
step = 0

for epoch in range(EPOCHS):
    print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")
    for idx, sample in enumerate(dataset):
        inputs = {
            "input_ids": torch.tensor(sample["input_ids"]).unsqueeze(0).to(DEVICE),
            "attention_mask": torch.tensor(sample["attention_mask"]).unsqueeze(0).to(DEVICE),
            "labels": torch.tensor(sample["labels"]).unsqueeze(0).to(DEVICE)
        }

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        if (idx + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            print(f"Step {step} | Loss: {loss.item():.4f}")

print("===== TRAINING COMPLETE =====")

# =========================
# SAVE
# =========================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("===== MODEL SAVED =====")
