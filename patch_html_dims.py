
import re
from pathlib import Path

def update_html():
    html_path = Path("onnx_generate_image.html")
    if not html_path.exists():
        print("❌ onnx_generate_image.html not found.")
        return

    print(f"🔄 Updating {html_path} to 12x12 SharpFlow dimensions...")
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update LATENT_SHAPE [1, 8, 6, 6] -> [1, 8, 12, 12]
    # We use a broad regex to catch varied spacing
    content = re.sub(
        r'LATENT_SHAPE\s*=\s*\[1,\s*8,\s*\d+,\s*\d+\]',
        'LATENT_SHAPE = [1, 8, 12, 12]',
        content
    )

    # 2. Ensure IMG_SIZE is 96 (standard for SharpFlow)
    content = re.sub(
        r'let\s+IMG_SIZE\s*=\s*\d+',
        'let IMG_SIZE = 96',
        content
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Success! HTML file updated. Please refresh your browser.")

if __name__ == "__main__":
    update_html()
