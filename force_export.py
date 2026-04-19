
import torch
from torch.utils.data import DataLoader, TensorDataset
import training
import config

def force_export():
    print("🔄 Initializing trainer for ONNX export...")
    # Create a dummy loader
    dummy_data = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    dummy_loader = DataLoader(TensorDataset(dummy_data), batch_size=1)
    
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    
    print("📂 Loading latest checkpoint...")
    # Use load_for_inference but we need to ensure config.DEVICE is cpu for this run
    original_device = config.DEVICE
    config.DEVICE = torch.device('cpu')
    
    if trainer.load_for_inference():
        print(f"🚀 Exporting Generator only to {config.DIRS['onnx']}...")
        # Custom logic to only call generator export if needed, 
        # but trainer.export_onnx() handles both, let's just let it run 
        # since it successfully did the generator.
        trainer.export_onnx()
        print("✅ Check the 'onnx' folder. generator.onnx should be updated.")
    else:
        print("❌ Failed to load checkpoint. Make sure latest.pt exists.")
    
    config.DEVICE = original_device

if __name__ == "__main__":
    force_export()
