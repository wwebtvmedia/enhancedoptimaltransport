# ============================================================================
# LABEL-CONDITIONED INFERENCE FOR SCHRÖDINGER BRIDGE
# ============================================================================

import torch
from pathlib import Path
from typing import List, Optional
import training
import data_management as dm
import config

def run_inference(labels: Optional[List[int]] = None,
                  samples_per_label: Optional[int] = None,
                  temperature: Optional[float] = None,
                  method: str = 'rk4') -> None:
    """
    Run inference with label conditioning.
    
    Args:
        labels: List of class labels. If None, prompts user.
        samples_per_label: Number of samples per label. If None, prompts user.
        temperature: Sampling temperature. If None, prompts user.
        method: Integration method ('euler' or 'rk4'). Default 'rk4'.
    """
    checkpoint_path = dm.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        training.logger.error("No trained model found! Train a model first.")
        return
    
    # Create dummy loader for trainer initialization
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, 64, 64))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    
    if not trainer.load_for_inference():
        training.logger.error("Failed to load checkpoint")
        return
    
    training.logger.info("\n" + "="*50)
    training.logger.info("LABEL-CONDITIONED INFERENCE")
    training.logger.info("="*50)
    
    # Interactive input if parameters not provided
    if labels is None:
        print("\nAvailable labels: 0-9 for CIFAR-10 classes")
        print("(0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,")
        print(" 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)")
        label_input = input("\nEnter labels (comma-separated, e.g., 0,1,2,3) [default: 0,1,2,3]: ").strip()
        if label_input:
            labels = [int(x.strip()) for x in label_input.split(',')]
        else:
            labels = [0, 1, 2, 3]
    
    if samples_per_label is None:
        samples_input = input(f"Samples per label [default: 2]: ").strip()
        samples_per_label = int(samples_input) if samples_input else 2
    
    if temperature is None:
        temp_input = input(f"Temperature [default: {config.INFERENCE_TEMPERATURE}] (0.3-1.2): ").strip()
        temperature = float(temp_input) if temp_input else config.INFERENCE_TEMPERATURE
    
    # Ensure method is valid
    method = method if method in ['euler', 'rk4'] else 'rk4'
    
    # Build list of labels
    all_labels = []
    for label in labels:
        all_labels.extend([label] * samples_per_label)
    
    print(f"\n Generating {len(all_labels)} samples with temperature {temperature} using {method.upper()}...")
    
    # Generate samples
    grid_path = trainer.generate_samples(
        labels=all_labels,
        temperature=temperature,
        method=method
    )
    
    print(f"\n Generated {len(all_labels)} samples")
    print(f" Saved to: {grid_path}")
    print(f" Individual images in: {dm.DIRS['samples']}")
    
    # Print summary
    print("\n Sample Summary:")
    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                  "dog", "frog", "horse", "ship", "truck"]
    for label in sorted(set(labels)):
        count = labels.count(label) * samples_per_label
        name = class_names[label] if label < 10 else f"class_{label}"
        print(f"   Class {label} ({name}): {count} images")

if __name__ == "__main__":
    run_inference()