
import torch
import os

def check_checkpoint(path):
    if not os.path.exists(path):
        print(f"{path} not found")
        return
    print(f"\nChecking {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f"  Epoch: {ckpt.get('epoch')}")
        print(f"  Phase: {ckpt.get('phase')}")
        
        vae_state = ckpt.get('vae_state', {})
        drift_state = ckpt.get('drift_state', {})
        
        if vae_state:
            vae_norm = 0
            for k, v in vae_state.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    vae_norm += v.norm().item()
            print(f"  VAE floating-point weight norm sum: {vae_norm:.4f}")
        
        if drift_state:
            drift_norm = 0
            for k, v in drift_state.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    drift_norm += v.norm().item()
            print(f"  Drift floating-point weight norm sum: {drift_norm:.4f}")
            
            # Check output_scale
            if 'output_scale' in drift_state:
                print(f"  Drift output_scale: {drift_state['output_scale'].item():.4f}")
            if 'time_scales' in drift_state:
                print(f"  Drift time_scales: {drift_state['time_scales']}")

    except Exception as e:
        print(f"  Error loading {path}: {e}")

check_checkpoint('enhanced_label_sb/checkpoints/latest.pt')
check_checkpoint('enhanced_label_sb/checkpoints/latest.pt.bak')
