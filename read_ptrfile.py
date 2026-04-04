import torch
import collections
from pathlib import Path

# Path to your checkpoint
checkpoint_path = 'enhanced_label_sb/checkpoints/latest.pt'

def inspect_checkpoint(path):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"❌ Error: File not found at {path}")
        return

    print(f"🔍 --- Inspecting Checkpoint: {path} ---\n")

    try:
        # Using weights_only=False to allow loading the 'defaultdict' in kpi_metrics
        # map_location='cpu' ensures we can read it even without a GPU
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # 1. Basic Metadata
        print("📊 [ METADATA ]")
        print(f"  - Current Epoch:      {ckpt.get('epoch', 'N/A')}")
        print(f"  - Total Steps:        {ckpt.get('step', 'N/A')}")
        print(f"  - Training Phase:     {ckpt.get('phase', 'N/A')}")
        
        best_loss = ckpt.get('best_loss', 'N/A')
        if isinstance(best_loss, float):
            print(f"  - Best Loss:          {best_loss:.6f}")
        else:
            print(f"  - Best Loss:          {best_loss}")
            
        print(f"  - Best Composite:     {ckpt.get('best_composite_score', 'N/A')}")
        
        # 2. Config Snapshot
        if 'config' in ckpt:
            print("\n⚙️ [ SAVED CONFIG ]")
            for k, v in ckpt['config'].items():
                print(f"  - {k:15}: {v}")

        # 3. Model Weights Summary
        print("\n🧠 [ MODEL WEIGHTS ]")
        vae_weights = ckpt.get('vae_state', {})
        drift_weights = ckpt.get('drift_state', {})
        print(f"  - VAE layers:         {len(vae_weights)}")
        print(f"  - Drift layers:       {len(drift_weights)}")
        
        if 'vae_ref_state' in ckpt:
            print(f"  - VAE Anchor layers:  {len(ckpt['vae_ref_state'])}")

        # 4. Metrics History
        if 'kpi_metrics' in ckpt:
            print("\n📈 [ KPI METRICS ]")
            metrics = ckpt['kpi_metrics']
            for m_name in sorted(metrics.keys()):
                data = metrics[m_name]
                if isinstance(data, list):
                    count = len(data)
                    last = data[-1] if count > 0 else 'N/A'
                    if isinstance(last, float):
                        print(f"  - {m_name:18} ({count:4} pts) Last: {last:.4f}")
                    else:
                        print(f"  - {m_name:18} ({count:4} pts) Last: {last}")
                else:
                    print(f"  - {m_name:18}: {data}")

        # 5. Optimizer and Scheduler Status
        print("\n🔄 [ OPTIMIZERS & SCHEDULERS ]")
        opt_vae = "✅ Present" if 'opt_vae_state' in ckpt else "❌ Missing"
        opt_drift = "✅ Present" if 'opt_drift_state' in ckpt else "❌ Missing"
        sch_vae = "✅ Present" if 'scheduler_vae_state' in ckpt else "❌ Missing"
        sch_drift = "✅ Present" if 'scheduler_drift_state' in ckpt else "❌ Missing"
        
        print(f"  - VAE Optimizer:      {opt_vae}")
        print(f"  - Drift Optimizer:    {opt_drift}")
        print(f"  - VAE Scheduler:      {sch_vae}")
        print(f"  - Drift Scheduler:    {sch_drift}")

        print("\n✅ --- Inspection Complete ---")

    except Exception as e:
        print(f"❌ Failed to read checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_checkpoint(checkpoint_path)
