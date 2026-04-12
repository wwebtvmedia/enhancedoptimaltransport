
import torch
import collections
from torch.serialization import add_safe_globals

def patch_checkpoint(input_path, output_path):
    print(f"📦 Patching {input_path}...")
    add_safe_globals([collections.defaultdict])
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    d = ckpt.get('drift_state', {})
    if not d:
        print("No drift_state found")
        return

    # Reset rescale layers and statistics
    reset_count = 0
    for k in list(d.keys()):
        if '.rescale.low' in k:
            d[k] = torch.zeros_like(d[k])
            reset_count += 1
        elif '.rescale.high' in k:
            d[k] = torch.ones_like(d[k])
            reset_count += 1
        elif k in ['drift_mean', 'n_samples']:
            d[k] = torch.zeros_like(d[k])
            reset_count += 1
        elif k == 'drift_std':
            d[k] = torch.ones_like(d[k])
            reset_count += 1

    print(f"✅ Reset {reset_count} parameters/buffers in drift_state.")
    
    # Also reset VAE rescale if they exploded?
    v = ckpt.get('vae_state', {})
    v_reset = 0
    for k in list(v.keys()):
        if '.rescale.low' in k:
            v[k] = torch.zeros_like(v[k])
            v_reset += 1
        elif '.rescale.high' in k:
            v[k] = torch.ones_like(v[k])
            v_reset += 1
    print(f"✅ Reset {v_reset} parameters in vae_state.")

    torch.save(ckpt, output_path)
    print(f"🚀 Saved patched checkpoint to {output_path}")

patch_checkpoint('enhanced_label_sb/checkpoints/latest.pt.bak', 'enhanced_label_sb/checkpoints/latest_patched.pt')
