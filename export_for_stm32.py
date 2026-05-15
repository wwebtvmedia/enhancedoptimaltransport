import torch
import training
import config
from pathlib import Path
import os

def export_for_stm32():
    print("Exporting models with STM32-compatible layers...")

    info = config.initialize_hardware()
    print(f"Hardware initialized: {info}")

    from torch.utils.data import DataLoader, TensorDataset
    dummy = DataLoader(TensorDataset(torch.randn(1,3,config.IMG_SIZE,config.IMG_SIZE)), batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy)

    if not trainer.load_for_inference():
        print("Error: No checkpoint found to export.")
        return

    onnx_dir = Path(config.DIRS["onnx"])
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Opset 17 exports nn.LayerNorm as a single LayerNormalization op instead of
    # decomposed ReduceMean/Sub/Pow primitives.  The decomposed form leaves unquantized
    # float32 intermediate tensors (ReduceMean, Pow, Sub) that break stedgeai's shape
    # inference in QDQ mode.  Opset 17 avoids this entirely.
    original_opset = config.ONNX_OPSET_VERSION
    config.ONNX_OPSET_VERSION = 17
    try:
        trainer.export_onnx()
    finally:
        config.ONNX_OPSET_VERSION = original_opset

    print(f"Models exported to {onnx_dir}")

if __name__ == "__main__":
    export_for_stm32()
