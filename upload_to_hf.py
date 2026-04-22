
import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_to_hf(repo_id, local_path, path_in_repo, token, repo_type="model"):
    api = HfApi()
    print(f"📦 Uploading {local_path} to {repo_id}/{path_in_repo}...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token
        )
        return True
    except Exception as e:
        print(f"❌ Upload failed for {local_path}: {e}")
        return False

if __name__ == "__main__":
    TOKEN = os.environ.get("HF_TOKEN")
    if not TOKEN:
        TOKEN = input("Enter your Hugging Face 'Write' Token: ").strip()
    
    api = HfApi()
    try:
        user_info = api.whoami(token=TOKEN)
        USER_NAME = user_info['name']
        print(f"✅ Authenticated as: {USER_NAME}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("Tip: If you just pasted this token in a chat, it was likely revoked automatically.")
        print("Create a new token at: https://huggingface.co/settings/tokens")
        exit(1)

    REPO_NAME = "enhanced-schrodinger-bridge"
    REPO_ID = f"{USER_NAME}/{REPO_NAME}"
    
    print(f"🚀 Verifying repository: {REPO_ID}...")
    try:
        create_repo(repo_id=REPO_ID, token=TOKEN, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Files to upload
    files_to_upload = [
        ("enhanced_label_sb/checkpoints/latest.pt", "latest.pt"),
        ("enhanced_label_sb/onnx/drift.onnx", "onnx/drift.onnx"),
        ("enhanced_label_sb/onnx/generator.onnx", "onnx/generator.onnx"),
        ("label_map.json", "label_map.json"),
        ("config.py", "config.py"),
    ]

    success_count = 0
    for local_path, repo_path in files_to_upload:
        if os.path.exists(local_path):
            if upload_to_hf(REPO_ID, local_path, repo_path, TOKEN):
                success_count += 1
        else:
            print(f"⚠️ Skipping missing file: {local_path}")

    if success_count > 0:
        print(f"\n✨ Success! {success_count} files uploaded.")
        print(f"🔗 View your model at: https://huggingface.co/{REPO_ID}")
    else:
        print("\n❌ No files were uploaded.")
