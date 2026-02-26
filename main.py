# ============================================================================
# MAIN ENTRY POINT FOR SCHRÖDINGER BRIDGE TRAINING
# ============================================================================

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Import modules
import config
import training
import data_management as dm
import inference
import models

# Re-export logger
logger = config.logger

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
def get_device() -> torch.device:
    """Get the best available device with support for AMD and Apple Silicon."""
    # Check CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS device")
            return device
        except Exception as e:
            logger.warning(f"MPS available but failed to initialize: {e}")
    
    # Check ROCm (AMD)
    try:
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using AMD ROCm device (via CUDA API)")
                return device
    except:
        pass
    
    # Check DirectML (Windows with AMD/Intel)
    try:
        import torch_directml
        if hasattr(torch_directml, 'is_available') and torch_directml.is_available():
            device = torch_directml.device()
            logger.info(f"Using DirectML device: {torch_directml.device_name(device)}")
            return device
    except ImportError:
        pass
    
    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU device (no GPU acceleration available)")
    return device

def get_dtype_for_device(device: torch.device) -> torch.dtype:
    """Get optimal dtype for device."""
    return torch.float32

def initialize_for_device() -> None:
    """Initialize PyTorch with device-specific settings."""
    if config.DEVICE.type == 'mps':
        torch.set_default_dtype(torch.float32)
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    if config.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif config.DEVICE.type == 'mps':
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Set seeds
    torch.manual_seed(config.DEFAULT_SEED)
    if config.DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(config.DEFAULT_SEED)
    elif config.DEVICE.type == 'mps':
        try:
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'manual_seed'):
                torch.mps.manual_seed(config.DEFAULT_SEED)
        except:
            pass
    
    np.random.seed(config.DEFAULT_SEED)
    logger.info(f"Initialized for {config.DEVICE.type.upper()} device")

def check_device_compatibility() -> bool:
    """Check if the current device is compatible with the training setup."""
    issues = []
    
    if config.DEVICE.type == 'mps':
        try:
            import platform
            mac_version = platform.mac_ver()[0]
            logger.info(f"macOS version: {mac_version}")
            
            if not hasattr(torch.backends, 'mps'):
                issues.append("PyTorch not built with MPS support")
            elif not torch.backends.mps.is_available():
                issues.append("MPS not available on this system")
            elif not torch.backends.mps.is_built():
                issues.append("PyTorch not built with MPS")
        except Exception as e:
            issues.append(f"MPS check failed: {e}")
    
    elif config.DEVICE.type == 'directml':
        try:
            import torch_directml
            device_name = torch_directml.device_name(torch_directml.device())
            logger.info(f"DirectML device: {device_name}")
        except Exception as e:
            issues.append(f"DirectML check failed: {e}")
    
    if issues:
        logger.warning("Device compatibility issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("Training may be slower or fail on this device.")
    
    return len(issues) == 0

def configure_device_specific() -> None:
    """Adjust global constants based on detected device."""
    if config.DEVICE.type == 'cpu':
        config.BATCH_SIZE = 32
        config.LR = 1e-4
        config.USE_AMP = False
    elif config.DEVICE.type == 'mps':
        config.BATCH_SIZE = 48
        config.LR = 1.5e-4
        config.USE_AMP = False
        # Set MPS-specific environment variables
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        try:
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(0.5)
        except:
            pass
    elif config.DEVICE.type == 'directml':
        config.BATCH_SIZE = 48
        config.LR = 2e-4
        config.USE_AMP = False
    else:  # CUDA
        config.BATCH_SIZE = 64
        config.LR = 2e-4
        config.USE_AMP = True  # Enable AMP for CUDA
    
    # Update module-level constants
    dm.BATCH_SIZE = config.BATCH_SIZE
    dm.LR = config.LR
    dm.DEVICE = config.DEVICE
    
    training.LR = config.LR
    training.BATCH_SIZE = config.BATCH_SIZE
    training.DEVICE = config.DEVICE
    training.USE_AMP = config.USE_AMP
    training.DIRS = config.DIRS

    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Learning rate: {config.LR}")
    logger.info(f"AMP enabled: {config.USE_AMP}")

# ============================================================
# INTERACTIVE SNAPSHOT UTILITIES
# ============================================================
def select_snapshot_interactive() -> Optional[Path]:
    """Interactive snapshot selection with detailed info."""
    snap_files = list(config.DIRS["snaps"].glob("*_snapshot_epoch_*.pt"))
    if not snap_files:
        print("\n No snapshots found!")
        return None
    
    snap_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    print("\n" + "="*70)
    print(" SNAPSHOT SELECTION")
    print("="*70)
    
    snapshots_info = []
    for i, snap_path in enumerate(snap_files):
        try:
            snapshot = torch.load(snap_path, map_location='cpu', weights_only=False)
            epoch = snapshot.get('epoch', 'unknown')
            loss = snapshot.get('loss', 'N/A')
            if loss != 'N/A' and isinstance(loss, (int, float)):
                loss = f"{loss:.6f}"
            timestamp = snapshot.get('timestamp', 'unknown')[:16] if 'timestamp' in snapshot else 'unknown'
            model_type = snapshot.get('model_type', 'unknown')
            
            has_vae = 'model_state' in snapshot or model_type == 'vae'
            has_drift = 'drift_state' in snapshot or model_type == 'drift'
            
            info = {
                'index': i,
                'path': snap_path,
                'epoch': epoch,
                'loss': loss,
                'timestamp': timestamp,
                'has_vae': has_vae,
                'has_drift': has_drift,
                'model_type': model_type
            }
            snapshots_info.append(info)
            
            vae_symbol = "✓" if has_vae else "✗"
            drift_symbol = "✓" if has_drift else "✗"
            type_display = f"[{model_type.upper()}]" if model_type != 'unknown' else ""
            
            print(f"\n{i+1:2d}. {snap_path.name} {type_display}")
            print(f"     Epoch: {epoch} | Loss: {loss} | {timestamp}")
            print(f"    VAE: {vae_symbol}  Drift: {drift_symbol}")
            
        except Exception as e:
            print(f"\n{i+1:2d}. {snap_path.name} (⚠️ Metadata unavailable: {e})")
            info = {
                'index': i,
                'path': snap_path,
                'epoch': 'unknown',
                'loss': 'N/A',
                'timestamp': 'unknown',
                'has_vae': False,
                'has_drift': False,
                'model_type': 'unknown'
            }
            snapshots_info.append(info)
    
    print("\n" + "="*70)
    
    while True:
        try:
            choice = input("\n Select snapshot number (or 0 to cancel): ").strip()
            if choice == '0':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(snapshots_info):
                selected = snapshots_info[idx]
                print(f"\n Selected: {selected['path'].name}")
                return selected['path']
            else:
                print(f" Invalid selection. Please enter 1-{len(snapshots_info)}")
                
        except ValueError:
            print(" Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n Cancelled by user")
            return None

def interactive_snapshot_loader(trainer) -> Optional[Tuple[Path, bool, bool, Optional[int]]]:
    """Interactive menu for loading snapshots."""
    snap_path = select_snapshot_interactive()
    if snap_path is None:
        return None
    
    info = trainer.inspect_snapshot(snap_path)
    
    print("\n Snapshot Details:")
    print(f"  Path: {info.get('path', 'N/A')}")
    print(f"  Epoch: {info.get('epoch', 'N/A')}")
    print(f"  Loss: {info.get('loss', 'N/A')}")
    print(f"  Model Type: {info.get('model_type', 'N/A')}")
    print(f"  Contains VAE: {'Yes' if info.get('has_vae', False) else 'No'}")
    print(f"  Contains Drift: {'Yes' if info.get('has_drift', False) else 'No'}")
    
    load_vae = True
    load_drift = True
    phase = None
    
    if info.get('has_vae', False) and info.get('has_drift', False):
        # Both available, ask what to load
        print("\n Both VAE and Drift available in snapshot.")
        load_choice = input(" Load [1] VAE only, [2] Drift only, [3] Both [default: 3]: ").strip()
        if load_choice == '1':
            load_drift = False
        elif load_choice == '2':
            load_vae = False
    
    # Ask for phase
    phase_input = input("\n Set phase? [1] Phase 1 (VAE), [2] Phase 2 (Drift), [0] No change [default: 0]: ").strip()
    if phase_input == '1':
        phase = 1
    elif phase_input == '2':
        phase = 2
    
    return snap_path, load_vae, load_drift, phase

def configure_training_schedule_interactive() -> None:
    """Interactive menu for configuring training schedule."""
    print("\n" + "="*70)
    print(" TRAINING SCHEDULE CONFIGURATION")
    print("="*70)
    
    print("\nCurrent schedule:")
    print(f"  Mode: {config.TRAINING_SCHEDULE['mode']}")
    print(f"  Switch epoch: {config.TRAINING_SCHEDULE.get('switch_epoch', 50)}")
    
    print("\nSelect mode:")
    print("  1. Auto (VAE for first N epochs, then Drift)")
    print("  2. VAE only")
    print("  3. Drift only")
    print("  4. Alternate (switch every N epochs)")
    print("  5. Custom (manual epoch mapping)")
    
    choice = input("\n Enter choice (1-5): ").strip()
    
    if choice == '1':
        switch = input(f" Switch epoch [default: 50]: ").strip()
        switch_epoch = int(switch) if switch else 50
        training.configure_training_schedule(mode='auto', switch_epoch=switch_epoch)
    
    elif choice == '2':
        training.configure_training_schedule(mode='vae_only')
    
    elif choice == '3':
        training.configure_training_schedule(mode='drift_only')
    
    elif choice == '4':
        freq = input(f" Alternate frequency [default: 5]: ").strip()
        alt_freq = int(freq) if freq else 5
        training.configure_training_schedule(mode='alternate', alternate_freq=alt_freq)
    
    elif choice == '5':
        print("\n Enter custom schedule as epoch:phase pairs")
        print(" Example: 0:1,10:1,20:2,30:2")
        custom = input(" Schedule: ").strip()
        schedule = {}
        if custom:
            for pair in custom.split(','):
                if ':' in pair:
                    e, p = pair.split(':')
                    schedule[int(e.strip())] = int(p.strip())
        training.configure_training_schedule(mode='custom', custom_schedule=schedule)
    
    print(f"\n Schedule updated to mode: {config.TRAINING_SCHEDULE['mode']}")

def continue_training_from_snapshot(trainer, remaining_epochs: Optional[int] = None) -> None:
    """Continue training from a loaded snapshot."""
    if remaining_epochs is None:
        remaining_epochs = config.EPOCHS - trainer.epoch
    
    print(f"\n Continuing training from epoch {trainer.epoch}")
    print(f"  Remaining epochs: {remaining_epochs}")
    print(f"  Current phase: {trainer.phase}")
    
    confirm = input("\n Start training? (y/n): ").strip().lower()
    if confirm == 'y':
        # Load real data
        loader = dm.load_data()
        trainer.loader = loader
        
        # Continue training
        for epoch in range(trainer.epoch, trainer.epoch + remaining_epochs):
            trainer.epoch = epoch
            trainer.train_epoch()
            
            # Save checkpoints periodically
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint()
            
            # Generate samples
            if (epoch + 1) % 10 == 0:
                trainer.generate_samples()
    else:
        print("Training cancelled.")

# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    """Main entry point."""
    # Detect device and update global constants
    config.DEVICE = get_device()
    config.DTYPE = get_dtype_for_device(config.DEVICE)
    configure_device_specific()
    initialize_for_device()
    
    # Display banner
    print("\n" + "="*70)
    print("ENHANCED LABEL-CONDITIONED SCHRÖDINGER BRIDGE")
    print("WITH ORNSTEIN-UHLENBECK REFERENCE (mvOU-SBP)")
    print("="*70)
    print(f"Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Latent: {config.LATENT_CHANNELS}x{config.LATENT_H}x{config.LATENT_W}")
    print(f"Label Conditioning: {config.NUM_CLASSES} classes")
    print(f"Percentile Rescaling: {config.USE_PERCENTILE}")
    print(f"KPI Tracking: {config.USE_KPI_TRACKING}")
    print(f"Snapshot Learning: {config.USE_SNAPSHOTS}")
    print(f"Diversity Loss Weight: {config.DIVERSITY_WEIGHT}")
    print(f"Device: {config.DEVICE}")
    print(f"Training Mode: {config.TRAINING_SCHEDULE['mode']}")
    print(f"OU Bridge: {'Enabled' if config.USE_OU_BRIDGE else 'Disabled'} (theta={config.OU_THETA})")
    print("="*70)
    
    if not check_device_compatibility():
        print("\n⚠️ Device compatibility warnings detected!")
        print("   Training may proceed with reduced performance.")
        print("   Consider using a different device if available.\n")
    
    # Main menu
    while True:
        print("\n MAIN MENU:")
        print("  1. Enhanced training (fresh start)")
        print("  2. Quick test (5 epochs)")
        print("  3. Export models to ONNX")
        print("  4. Generate samples from checkpoint")
        print("  5. Label-conditioned inference")
        print("  6. Snapshot management & recovery")
        print("  7. Resume from latest checkpoint")
        print("  8. Configure training schedule")
        print("  9. Toggle OU bridge (currently {})".format("ON" if config.USE_OU_BRIDGE else "OFF"))
        print("  0. Exit")
        
        choice = input("\n Enter choice (0-9): ").strip()
        
        # Handle OU bridge toggle
        if choice == '9':
            config.USE_OU_BRIDGE = not config.USE_OU_BRIDGE
            training.USE_OU_BRIDGE = config.USE_OU_BRIDGE
            print(f"OU bridge is now {'ON' if config.USE_OU_BRIDGE else 'OFF'}")
            continue
        
        # Handle exit
        if choice == '0':
            print("\nExiting...")
            break
        
        # Create dummy loader for trainer initialization
        from torch.utils.data import DataLoader, TensorDataset
        dummy_dataset = TensorDataset(torch.randn(1, 3, 64, 64))
        dummy_loader = DataLoader(dummy_dataset, batch_size=1)
        
        if choice == '1':
            epochs_input = input(f"Number of epochs [{config.EPOCHS}]: ").strip()
            epochs = int(epochs_input) if epochs_input else config.EPOCHS
            training.train_model(num_epochs=epochs)
            
        elif choice == '2':
            print("\n Running quick test (5 epochs)...")
            training.train_model(num_epochs=5)
            
        elif choice == '3':
            trainer = training.EnhancedLabelTrainer(dummy_loader)
            checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
            if checkpoint_path.exists():
                trainer.load_for_inference()
                trainer.export_onnx()
            else:
                print(" No trained model found!")
                
        elif choice == '4':
            # Generate samples from checkpoint with interactive parameters
            trainer = training.EnhancedLabelTrainer(dummy_loader)
            checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
            if checkpoint_path.exists():
                trainer.load_for_inference()
                
                # Prompt for generation parameters
                num_samples = input("Number of samples to generate [8]: ").strip()
                num_samples = int(num_samples) if num_samples else 8
                
                labels_input = input("Labels (comma-separated, e.g., 0,1,2,3) [random]: ").strip()
                if labels_input:
                    labels = [int(x.strip()) for x in labels_input.split(',')]
                else:
                    labels = None  # random labels
                
                temp_input = input(f"Temperature [default: {config.INFERENCE_TEMPERATURE}] (0.3-1.2): ").strip()
                temperature = float(temp_input) if temp_input else config.INFERENCE_TEMPERATURE
                
                method_input = input("Integration method (euler / heun / rk4) [default: heun]: ").strip().lower()
                method = method_input if method_input in ['euler', 'rk4'] else 'rk4'
                
                # Langevin refinement options
                langevin_steps_input = input("Langevin refinement steps [default: 0 (disabled)]: ").strip()
                langevin_steps = int(langevin_steps_input) if langevin_steps_input else 0
                langevin_step_size = 0.1
                langevin_score_scale = 1.0
                if langevin_steps > 0:
                    step_input = input("Langevin step size [default: 0.1]: ").strip()
                    langevin_step_size = float(step_input) if step_input else 0.1
                    scale_input = input("Langevin score scale [default: 1.0]: ").strip()
                    langevin_score_scale = float(scale_input) if scale_input else 1.0
                
                print(f"\n Generating {num_samples} samples with temperature {temperature} using {method.upper()}...")
                if langevin_steps > 0:
                    print(f"  + {langevin_steps} Langevin steps (step_size={langevin_step_size}, scale={langevin_score_scale})")
                
                trainer.generate_samples(labels=labels, num_samples=num_samples,
                                        temperature=temperature, method=method,
                                        langevin_steps=langevin_steps,
                                        langevin_step_size=langevin_step_size,
                                        langevin_score_scale=langevin_score_scale)
                print(f"\n Samples saved to: {config.DIRS['samples']}")
            else:
                print(" No trained model found!")       
        elif choice == '5':
            # Label-conditioned inference – gather all inputs and pass to inference.run_inference()
            print("\n Inference Configuration")
            print("-" * 30)
            
            # Labels
            print("\nAvailable labels: 0-9 for CIFAR-10 classes")
            print("(0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,")
            print(" 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)")
            label_input = input("\nEnter labels (comma-separated, e.g., 0,1,2,3) [default: 0,1,2,3]: ").strip()
            if label_input:
                labels = [int(x.strip()) for x in label_input.split(',')]
            else:
                labels = [0, 1, 2, 3]
            
            # Samples per label
            samples_input = input(f"Samples per label [default: 2]: ").strip()
            samples_per_label = int(samples_input) if samples_input else 2
            
            # Temperature
            temp_input = input(f"Temperature [default: {config.INFERENCE_TEMPERATURE}] (0.3-1.2): ").strip()
            temperature = float(temp_input) if temp_input else config.INFERENCE_TEMPERATURE
            
            # Integration method
            method_input = input("Integration method (euler / heun / rk4) [default: heun]: ").strip().lower()
            method = method_input if method_input in ['euler', 'rk4'] else 'rk4'
            
            # Langevin refinement
            langevin_steps_input = input("Langevin refinement steps [default: 0 (disabled)]: ").strip()
            langevin_steps = int(langevin_steps_input) if langevin_steps_input else 0
            langevin_step_size = 0.1
            langevin_score_scale = 1.0
            if langevin_steps > 0:
                step_input = input("Langevin step size [default: 0.1]: ").strip()
                langevin_step_size = float(step_input) if step_input else 0.1
                scale_input = input("Langevin score scale [default: 1.0]: ").strip()
                langevin_score_scale = float(scale_input) if scale_input else 1.0
            
            # Call inference with the collected parameters
            inference.run_inference(labels=labels,
                                    samples_per_label=samples_per_label,
                                    temperature=temperature,
                                    method=method,
                                    langevin_steps=langevin_steps,
                                    langevin_step_size=langevin_step_size,
                                    langevin_score_scale=langevin_score_scale)
        elif choice == '6':
            trainer = training.EnhancedLabelTrainer(dummy_loader)
            result = interactive_snapshot_loader(trainer)
            
            if result is not None:
                snapshot_path, load_vae, load_drift, phase = result
                print("\n Loading real dataset for training...")
                real_loader = dm.load_data()
                real_trainer = training.EnhancedLabelTrainer(real_loader)
                
                if real_trainer.load_from_snapshot(snapshot_path, load_vae, load_drift, phase):
                    epochs_input = input(f"\n Additional epochs to train: ").strip()
                    if epochs_input:
                        try:
                            additional_epochs = int(epochs_input)
                            continue_training_from_snapshot(real_trainer, additional_epochs)
                        except ValueError:
                            continue_training_from_snapshot(real_trainer)
                    else:
                        continue_training_from_snapshot(real_trainer)
                else:
                    print(" Failed to load snapshot into real trainer.")
            else:
                print("Snapshot loading cancelled.")                
        elif choice == '7':
            trainer = training.EnhancedLabelTrainer(dummy_loader)
            if trainer.load_checkpoint():
                print(f"\n Loaded checkpoint from epoch {trainer.epoch}")
                remaining = config.EPOCHS - trainer.epoch
                epochs_input = input(f"Additional epochs to train [{remaining}]: ").strip()
                additional = int(epochs_input) if epochs_input else remaining
                continue_training_from_snapshot(trainer, additional)
            else:
                print(" No checkpoint found!")
                
        elif choice == '8':
            configure_training_schedule_interactive()
            
        else:
            print(" Invalid choice")

if __name__ == "__main__":
    try:
        # Verify scipy is available
        import scipy.stats
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install scipy")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()