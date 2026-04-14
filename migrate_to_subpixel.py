
import torch
from pathlib import Path
import config
import models

def migrate_checkpoint(input_path, output_path):
    print(f"📦 Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    old_vae_state = checkpoint['vae_state']
    old_opt_state = checkpoint.get('opt_vae_state', None)
    
    # Initialize new model to get targets
    new_model = models.LabelConditionedVAE()
    new_vae_state = new_model.state_dict()
    
    # Mapping for dec_blocks based on 4-stage to 3-stage transition
    # New Index -> Old Index
    # Old was 6->12(0), 12->12(1), 12->12(2), 12->24(3), 24->24(4), 24->24(5), 24->48(6), 48->48(7), 48->96(8), 96->96(9)
    # New is 12->24(0), 24->24(1), 24->24(2), 24->48(3), 48->48(4), 48->48(5), 48->96(6), 96->96(7)
    
    dec_block_map = {
        0: 3, # Upsample 12->24
        1: 4, # LCB 24->24
        2: 5, # Attn 24->24
        3: 6, # Upsample 24->48
        4: 7, # LCB 48->48
        # 5 is Attention at 48 (new)
        6: 8, # Upsample 48->96
        7: 9  # LCB 96->96
    }

    # 1. Migrate VAE Weights
    migrated_vae = {}
    for key in new_vae_state.keys():
        found = False
        
        # Handle dec_blocks specifically
        if "dec_blocks" in key:
            # Extract block index
            parts = key.split('.')
            new_idx = int(parts[1])
            if new_idx in dec_block_map:
                old_idx = dec_block_map[new_idx]
                
                # Try to map the rest of the key
                # e.g. dec_blocks.0.conv.weight -> dec_blocks.3.1.weight (if it was Sequential)
                suffix = ".".join(parts[2:])
                
                # Check for special upsample conv mapping
                if "conv.weight" in suffix:
                    old_key = f"dec_blocks.{old_idx}.1.weight"
                    if old_key in old_vae_state:
                        print(f"🩹 Migrating Subpixel Weight: {old_key} -> {key}")
                        val = old_vae_state[old_key]
                        migrated_vae[key] = val.repeat_interleave(4, dim=0)
                        found = True
                elif "conv.bias" in suffix:
                    old_key = f"dec_blocks.{old_idx}.1.bias"
                    if old_key in old_vae_state:
                        print(f"🩹 Migrating Subpixel Bias: {old_key} -> {key}")
                        val = old_vae_state[old_key]
                        migrated_vae[key] = val.repeat_interleave(4)
                        found = True
                else:
                    # Regular mapping
                    old_key = f"dec_blocks.{old_idx}.{suffix}"
                    if old_key in old_vae_state:
                        if old_vae_state[old_key].shape == new_vae_state[key].shape:
                            # print(f"🩹 Mapping {old_key} -> {key}")
                            migrated_vae[key] = old_vae_state[old_key]
                            found = True
                        else:
                            print(f"⚠️ Shape mismatch for {key} (old: {old_key}): checkpoint {list(old_vae_state[old_key].shape)} vs model {list(new_vae_state[key].shape)}")

        # General mapping for keys that exist in both
        if not found and key in old_vae_state:
            if old_vae_state[key].shape == new_vae_state[key].shape:
                migrated_vae[key] = old_vae_state[key]
                found = True
            else:
                 print(f"⚠️ Shape mismatch for {key}: checkpoint {list(old_vae_state[key].shape)} vs model {list(new_vae_state[key].shape)}")

        if not found:
            # Re-initialize if not found or mismatch
            migrated_vae[key] = new_vae_state[key]

    # 2. Migrate Optimizer State (Clear it)
    checkpoint['opt_vae_state'] = None 
    print("🔧 Optimizer state cleared for safety.")

    checkpoint['vae_state'] = migrated_vae
    torch.save(checkpoint, output_path)
    print(f"🚀 Saved migrated checkpoint to {output_path}")

if __name__ == "__main__":
    ckpt_dir = Path("enhanced_label_sb/checkpoints")
    latest = ckpt_dir / "latest.pt"
    # Note: We work on the backup created previously
    target = ckpt_dir / "latest.pt.pre_subpixel"
    
    if not target.exists():
        print(f"❌ Target {target} not found!")
    else:
        migrate_checkpoint(target, ckpt_dir / "latest_subpixel_v3.pt")
        # Move to latest
        (ckpt_dir / "latest_subpixel_v3.pt").replace(latest)
        print("✨ Migration V3 complete.")
