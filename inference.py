# ============================================================================
# LABEL-CONDITIONED INFERENCE FOR SCHRÖDINGER BRIDGE
# ============================================================================

import torch
from pathlib import Path
from typing import List, Optional
import training
import data_management as dm
import config

def run_inference() -> None:
    """Run inference with label conditioning – uses fixed forward integration."""
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
    training.logger.info("LABEL-CONDITIONED INFERENCE (FORWARD FLOW)")
    training.logger.info("="*50)
    
    # Display class information
    print("\nAvailable labels: 0-9 for CIFAR-10 classes")
    print("(0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,")
    print(" 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)")
    
    # Get user input with validation
    label_input = input("\nEnter labels (comma-separated, e.g., 0,1,2,3) [default: 0,1,2,3]: ").strip()
    labels = []
    if label_input:
        try:
            labels = [int(x.strip()) for x in label_input.split(',')]
            # Validate labels
            for label in labels:
                if label < 0 or label >= config.NUM_CLASSES:
                    print(f"Warning: Label {label} may be out of range (0-{config.NUM_CLASSES-1})")
        except ValueError:
            print("Invalid input. Using default labels.")
            labels = [0, 1, 2, 3]
    else:
        labels = [0, 1, 2, 3]
    
    # Get samples per label
    samples_input = input(f"Samples per label [default: 2]: ").strip()
    try:
        samples_per_label = int(samples_input) if samples_input else 2
        samples_per_label = max(1, min(10, samples_per_label))  # Clamp to reasonable range
    except ValueError:
        samples_per_label = 2
    
    # Get temperature
    temp_input = input(f"Temperature [default: 0.8] (0.3-1.2): ").strip()
    try:
        temperature = float(temp_input) if temp_input else training.INFERENCE_TEMPERATURE
        temperature = max(0.3, min(1.2, temperature))  # Clamp to reasonable range
    except ValueError:
        temperature = training.INFERENCE_TEMPERATURE
    
    # Create label list
    all_labels = []
    for label in labels:
        all_labels.extend([label] * samples_per_label)
    
    print(f"\n Generating {len(all_labels)} samples with temperature {temperature}...")
    
    # Generate samples
    grid_path = trainer.generate_samples(
        labels=all_labels, 
        temperature=temperature
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