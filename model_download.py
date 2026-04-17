from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = r"P:\loan prediction via sml\models"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False

)

print("✅ Qwen model downloaded to:", LOCAL_DIR)
