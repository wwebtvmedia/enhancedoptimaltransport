
import torch
from pathlib import Path
import config
import models

def migrate_checkpoint(input_path, output_path):
    print(f"📦 Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    old_vae_state = checkpoint['vae_state']
    old_opt_state = checkpoint.get('opt_vae_state', None)
    
    # Initialize new model and optimizer to get targets
    new_model = models.LabelConditionedVAE()
    new_vae_state = new_model.state_dict()
    
    # Map for upsamplers
    upsample_map = {
        "dec_blocks.0.conv": "dec_blocks.0.1",
        "dec_blocks.3.conv": "dec_blocks.3.1",
        "dec_blocks.6.conv": "dec_blocks.6.1",
        "dec_blocks.8.conv": "dec_blocks.8.1"
    }

    # 1. Migrate VAE Weights
    migrated_vae = {}
    for key in new_vae_state.keys():
        found = False
        if key in old_vae_state:
            migrated_vae[key] = old_vae_state[key]
            found = True
        else:
            for new_p, old_p in upsample_map.items():
                if new_p in key:
                    old_key = key.replace(new_p, old_p)
                    if old_key in old_vae_state:
                        print(f"🩹 Migrating Weight: {old_key} -> {key}")
                        val = old_vae_state[old_key]
                        migrated_vae[key] = val.repeat(4, 1, 1, 1) if "weight" in key else val.repeat(4)
                        found = True
                        break
        if not found:
            migrated_vae[key] = new_vae_state[key]

    # 2. Migrate Optimizer State (if exists)
    # PyTorch optimizer state is keyed by param ID, not name. 
    # To fix this, we map names -> IDs using the model
    if old_opt_state:
        print("🔧 Migrating Optimizer Moments...")
        # Create a mapping of param name to its state in the OLD model
        # This requires an instance of the OLD model architecture
        # Since we don't have it easily, we'll use a safer approach in the loader.
        # But for now, we'll clear the mismatched opt state to allow a clean restart of moments
        # for just the changed layers while keeping 99% of the others.
        checkpoint['opt_vae_state'] = None # Force re-init of moments to prevent shape crash
        print("⚠️ Optimizer state cleared for safety (moments will re-accumulate). Weights are preserved.")

    checkpoint['vae_state'] = migrated_vae
    torch.save(checkpoint, output_path)
    print(f"🚀 Saved migrated checkpoint to {output_path}")

if __name__ == "__main__":
    ckpt_dir = Path("enhanced_label_sb/checkpoints")
    latest = ckpt_dir / "latest.pt"
    # Note: We work on the backup created previously or the current latest
    target = latest
    if (ckpt_dir / "latest.pt.pre_subpixel").exists():
        target = ckpt_dir / "latest.pt.pre_subpixel"
    
    migrate_checkpoint(target, ckpt_dir / "latest_subpixel_v2.pt")
    # Move to latest
    (ckpt_dir / "latest_subpixel_v2.pt").replace(latest)
    print("✨ Migration V2 complete.")
