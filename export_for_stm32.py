import torch
import training
import config
from pathlib import Path
import os

def export_for_stm32():
    print("Exporting models with STM32-compatible layers...")
    
    # Initialize hardware config
    info = config.initialize_hardware()
    print(f"Hardware initialized: {info}")
    
    # Initialize trainer with a dummy loader for export
    from torch.utils.data import DataLoader, TensorDataset
    dummy = DataLoader(TensorDataset(torch.randn(1,3,config.IMG_SIZE,config.IMG_SIZE)), batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy)
    
    # Load latest checkpoint
    if not trainer.load_for_inference():
        print("Error: No checkpoint found to export.")
        return

    # Create export directory
    onnx_dir = Path(config.DIRS["onnx"])
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    # Export
    trainer.export_onnx()
    print(f"Models exported to {onnx_dir}")

if __name__ == "__main__":
    export_for_stm32()
