
import torch
from pathlib import Path

def inspect_checkpoint(path):
    print(f"\n🔍 Inspecting: {path}")
    try:
        # Load on CPU
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        epoch = ckpt.get('epoch', 'N/A')
        phase = ckpt.get('phase', 'N/A')
        best_loss = ckpt.get('best_loss', 'N/A')
        best_score = ckpt.get('best_composite_score', 'N/A')
        
        print(f"  📅 Epoch: {epoch}")
        print(f"  🏗️ Phase: {phase}")
        print(f"  📉 Best Loss: {best_loss}")
        print(f"  🏆 Best Score: {best_score}")
        
        if 'config' in ckpt:
            print(f"  ⚙️ Config: {ckpt['config']}")
            
    except Exception as e:
        print(f"  ❌ Error loading: {e}")

if __name__ == "__main__":
    ckpt_dir = Path("enhanced_label_sb/checkpoints")
    inspect_checkpoint(ckpt_dir / "latest.pt")
    inspect_checkpoint(ckpt_dir / "latest.pt.old")
