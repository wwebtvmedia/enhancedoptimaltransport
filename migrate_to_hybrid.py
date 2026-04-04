
import torch
from pathlib import Path
import config
import models

def migrate_checkpoint_hybrid(input_path, output_path):
    print(f"📦 Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    old_vae_state = checkpoint['vae_state']
    
    new_model = models.LabelConditionedVAE()
    new_vae_state = new_model.state_dict()
    
    # Map for our new Hybrid structure
    # Old: dec_blocks.X.1 (Conv)
    # New: dec_blocks.X.fallback.1 (Conv)
    hybrid_map = {
        "dec_blocks.0.fallback.1": "dec_blocks.0.1",
        "dec_blocks.3.fallback.1": "dec_blocks.3.1",
        "dec_blocks.6.fallback.1": "dec_blocks.6.1",
        "dec_blocks.8.fallback.1": "dec_blocks.8.1"
    }

    migrated_count = 0
    for key in new_vae_state.keys():
        found = False
        if key in old_vae_state:
            new_vae_state[key] = old_vae_state[key]
            migrated_count += 1
            found = True
        else:
            for new_p, old_p in hybrid_map.items():
                if new_p in key:
                    old_key = key.replace(new_p, old_p)
                    if old_key in old_vae_state:
                        print(f"🩹 Rescuing {old_key} -> {key}")
                        new_vae_state[key] = old_vae_state[old_key]
                        migrated_count += 1
                        found = True
                        break
    
    print(f"✅ Migrated {migrated_count} tensors.")
    checkpoint['vae_state'] = new_vae_state
    # Keep opt state cleared for stability during transition
    checkpoint['opt_vae_state'] = None
    
    torch.save(checkpoint, output_path)
    print(f"🚀 Saved Hybrid checkpoint to {output_path}")

if __name__ == "__main__":
    ckpt_dir = Path("enhanced_label_sb/checkpoints")
    # Work on the original pre-subpixel backup to ensure clean data
    original = ckpt_dir / "latest.pt.pre_subpixel"
    if not original.exists():
        original = ckpt_dir / "latest.pt"
        
    migrate_checkpoint_hybrid(original, ckpt_dir / "latest_hybrid.pt")
    (ckpt_dir / "latest_hybrid.pt").replace(ckpt_dir / "latest.pt")
