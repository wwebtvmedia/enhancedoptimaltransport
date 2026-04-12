
import torch
import collections
from torch.serialization import add_safe_globals

def check_weights(path):
    print(f"\nAnalyzing weights in {path}...")
    add_safe_globals([collections.defaultdict])
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    d = ckpt.get('drift_state', {})
    if not d:
        print("No drift_state found")
        return

    # Filter for floating point tensors
    float_weights = {k: v for k, v in d.items() if isinstance(v, torch.Tensor) and torch.is_floating_point(v)}
    
    # Sort by norm
    sorted_weights = sorted(float_weights.items(), key=lambda x: x[1].norm().item(), reverse=True)
    
    print(f"Top 10 weights by norm:")
    for k, v in sorted_weights[:10]:
        print(f"  {k:40} | shape={str(list(v.shape)):15} | norm={v.norm().item():.4f} | mean={v.abs().mean().item():.4f}")

    print(f"\nBottom 10 weights by norm:")
    for k, v in sorted_weights[-10:]:
        print(f"  {k:40} | shape={str(list(v.shape)):15} | norm={v.norm().item():.4f} | mean={v.abs().mean().item():.4f}")

    # Count small weights
    small_weights = [k for k, v in float_weights.items() if v.norm().item() < 0.1]
    print(f"\nWeights with norm < 0.1: {len(small_weights)} / {len(float_weights)}")
    if len(small_weights) > 0:
        print(f"  Example small weights: {small_weights[:5]}")

check_weights('enhanced_label_sb/checkpoints/latest.pt')
check_weights('enhanced_label_sb/checkpoints/latest.pt.corrupted')
