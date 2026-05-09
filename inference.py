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
                  method: str = 'rk4',
                  langevin_steps: Optional[int] = None,
                  langevin_step_size: Optional[float] = None,
                  langevin_score_scale: Optional[float] = None,
                  cfg_scale: Optional[float] = None,
                  use_lora: Optional[bool] = None) -> None:
    """
    Run inference with label conditioning.
    """
    # Override config LoRA setting if specified
    if use_lora is not None:
        config.USE_LORA = use_lora
        
    # Ensure hardware is initialized
    config.initialize_hardware()
    
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        config.logger.error("No trained model found! Train a model first.")
        return
    
    # Create dummy loader for trainer initialization
    from torch.utils.data import DataLoader, TensorDataset
    # Minimal single-batch dummy dataset — Trainer won't iterate it during inference
    dummy = TensorDataset(
        torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE),  # image
    )
    dummy_loader = DataLoader(dummy, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    
    if not trainer.load_for_inference():
        config.logger.error("Failed to load checkpoint")
        return
    
    config.logger.info("\n" + "="*50)
    config.logger.info("LABEL-CONDITIONED INFERENCE")
    if config.USE_LORA:
        config.logger.info("  🚀 [LoRA-ENABLED MODE]")
    config.logger.info("="*50)
    
    # Interactive input if parameters not provided
    if labels is None:
        print("\nAvailable labels: 0-9 for standardized classes")
        names_display = ", ".join(f"{i}: {n}" for i, n in enumerate(config.CLASS_NAMES))
        print(f"({names_display})")
        print("(10: NULL/Unconditional)")
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
    
    # CFG Option
    if cfg_scale is None:
        default_cfg = getattr(config, 'CFG_SCALE', 1.0)
        cfg_input = input(f"CFG Scale [default: {default_cfg}] (1.0=disabled): ").strip()
        cfg_scale = float(cfg_input) if cfg_input else default_cfg

    # Langevin refinement options
    if langevin_steps is None:
        default_l_steps = getattr(config, 'DEFAULT_LANGEVIN_STEPS', 0)
        langevin_input = input(f"Langevin refinement steps [default: {default_l_steps}]: ").strip()
        langevin_steps = int(langevin_input) if langevin_input else default_l_steps
    
    if langevin_step_size is None:
        if langevin_steps > 0:
            step_input = input("Langevin step size [default: 0.1]: ").strip()
            langevin_step_size = float(step_input) if step_input else 0.1
        else:
            langevin_step_size = 0.1  # placeholder, not used
    
    if langevin_score_scale is None:
        if langevin_steps > 0:
            scale_input = input("Langevin score scale [default: 1.0]: ").strip()
            langevin_score_scale = float(scale_input) if scale_input else 1.0
        else:
            langevin_score_scale = 1.0
    
    # Ensure method is valid
    method = method if method in ['euler', 'heun', 'rk4'] else 'heun'
    
    # Build list of labels
    all_labels = []
    for label in labels:
        all_labels.extend([label] * samples_per_label)
    
    print(f"\n Generating {len(all_labels)} samples with temperature {temperature} using {method.upper()}...")
    print(f"  CFG Scale: {cfg_scale}")
    if langevin_steps > 0:
        print(f"  + {langevin_steps} Langevin refinement steps (step_size={langevin_step_size}, scale={langevin_score_scale})")
    
    # Generate samples
    grid_path = trainer.generate_samples(
        labels=all_labels,
        temperature=temperature,
        method=method,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
        langevin_score_scale=langevin_score_scale,
        cfg_scale=cfg_scale
    )
    
    print(f"\n Generated {len(all_labels)} samples")
    print(f" Saved to: {grid_path}")
    print(f" Individual images in: {config.DIRS['samples']}")
    
    # Print summary
    print("\n Sample Summary:")
    class_names = config.CLASS_NAMES
    for label in sorted(set(labels)):
        count = labels.count(label) * samples_per_label
        name = class_names[label] if label < 10 else f"class_{label}"
        print(f"   Class {label} ({name}): {count} images")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Schrödinger Bridge Inference")
    parser.add_argument("--labels", type=str, help="Comma-separated labels (e.g. 0,1,2)", default=None)
    parser.add_argument("--samples_per_label", type=int, help="Number of samples per label", default=None)
    parser.add_argument("--temperature", type=float, help="Generation temperature", default=None)
    parser.add_argument("--method", type=str, choices=['euler', 'heun', 'rk4'], default='rk4', help="ODE Solver method")
    parser.add_argument("--cfg_scale", type=float, help="Classifier-free guidance scale", default=None)
    parser.add_argument("--langevin_steps", type=int, help="Langevin refinement steps", default=None)
    parser.add_argument("--langevin_step_size", type=float, help="Langevin step size", default=None)
    parser.add_argument("--langevin_score_scale", type=float, help="Langevin score scale", default=None)
    parser.add_argument("--use_lora", action="store_true", help="Explicitly enable LoRA for inference")
    
    args = parser.parse_args()
    
    labels_list = None
    if args.labels is not None:
        labels_list = [int(x.strip()) for x in args.labels.split(',')]
        
    run_inference(
        labels=labels_list,
        samples_per_label=args.samples_per_label,
        temperature=args.temperature,
        method=args.method,
        langevin_steps=args.langevin_steps,
        langevin_step_size=args.langevin_step_size,
        langevin_score_scale=args.langevin_score_scale,
        cfg_scale=args.cfg_scale,
        use_lora=args.use_lora
    )