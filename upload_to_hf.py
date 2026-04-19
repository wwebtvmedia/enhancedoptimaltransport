
import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model(repo_id, local_file_path, token):
    api = HfApi()
    
    print(f"🚀 Creating/Verifying repository: {repo_id}...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print(f"📦 Uploading {local_file_path} to Hugging Face...")
    try:
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo="latest.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"✨ Success! Your model is now at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    # CONFIGURATION
    TOKEN = input("Enter your Hugging Face 'Write' Token: ").strip()
    USER_NAME = input("Enter your Hugging Face Username: ").strip()
    REPO_NAME = "enhanced-schrodinger-bridge"
    
    LOCAL_PATH = "enhanced_label_sb/checkpoints/latest.pt"
    REPO_ID = f"{USER_NAME}/{REPO_NAME}"
    
    if os.path.exists(LOCAL_PATH):
        upload_model(REPO_ID, LOCAL_PATH, TOKEN)
    else:
        print(f"❌ File not found: {LOCAL_PATH}")
