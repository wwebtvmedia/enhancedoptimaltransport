import os
import sys
import torch
import numpy as np

# Import modules
import training
import data_management as dm
import inference

# Re-export logger
logger = training.logger

# ============================================================
# DEVICE CONFIGURATION - SUPPORTS AMD & APPLE SILICON
# ============================================================
def get_device():
    """Get the best available device with support for AMD and Apple Silicon."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS device")
            return device
        except Exception as e:
            logger.warning(f"MPS available but failed to initialize: {e}")
    try:
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using AMD ROCm device (via CUDA API)")
                return device
    except:
        pass
    try:
        import torch_directml
        if hasattr(torch_directml, 'is_available') and torch_directml.is_available():
            device = torch_directml.device()
            logger.info(f"Using DirectML device: {torch_directml.device_name(device)}")
            return device
    except ImportError:
        pass
    device = torch.device("cpu")
    logger.info("Using CPU device (no GPU acceleration available)")
    return device

def get_dtype_for_device(device):
    return torch.float32

def initialize_for_device():
    """Initialize PyTorch with device-specific settings."""
    if dm.DEVICE.type == 'mps':
        torch.set_default_dtype(torch.float32)
    if dm.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    elif dm.DEVICE.type == 'mps':
        torch.mps.empty_cache()
    torch.manual_seed(training.DEFAULT_SEED)
    if dm.DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(training.DEFAULT_SEED)
    elif dm.DEVICE.type == 'mps':
        try:
            torch.mps.manual_seed(training.DEFAULT_SEED)
        except:
            pass
    np.random.seed(training.DEFAULT_SEED)
    logger.info(f"Initialized for {dm.DEVICE.type.upper()} device")

def check_device_compatibility():
    """Check if the current device is compatible with the training setup."""
    issues = []
    if dm.DEVICE.type == 'mps':
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
    elif dm.DEVICE.type == 'directml':
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

# ============================================================
# DEVICE-SPECIFIC CONFIGURATION UPDATE
# ============================================================
def configure_device_specific():
    """Adjust global constants based on detected device."""
    if dm.DEVICE.type == 'cpu':
        dm.BATCH_SIZE = 32
        dm.LR = 1e-4
    elif dm.DEVICE.type == 'mps':
        dm.BATCH_SIZE = 48
        dm.LR = 1.5e-4
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        try:
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(0.5)
        except:
            pass
    elif dm.DEVICE.type == 'directml':
        dm.BATCH_SIZE = 48
        dm.LR = 2e-4
    else:
        dm.BATCH_SIZE = 64
        dm.LR = 2e-4

    # Also update training constants
    training.LR = dm.LR
    training.BATCH_SIZE = dm.BATCH_SIZE
    training.DEVICE = dm.DEVICE
    training.DIRS = dm.DIRS

    logger.info(f"Device: {dm.DEVICE}")
    logger.info(f"Batch size: {dm.BATCH_SIZE}")
    logger.info(f"Learning rate: {dm.LR}")

    # AMP only on CUDA
    if dm.DEVICE.type == 'cuda':
        try:
            from torch.cuda.amp import autocast, GradScaler
            training.AMP_AVAILABLE = True
            logger.info("Automatic Mixed Precision (AMP) enabled for NVIDIA GPU")
        except ImportError:
            training.AMP_AVAILABLE = False
    else:
        training.AMP_AVAILABLE = False
        logger.info("AMP not available for this device, using float32")

# ============================================================
# INTERACTIVE SNAPSHOT UTILITIES (shortened for brevity)
# ============================================================
def select_snapshot_interactive():
    """Interactive snapshot selection with detailed info."""
    snap_files = list(dm.DIRS["snaps"].glob("*_snapshot_epoch_*.pt"))
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
            print(f"     Epoch: {epoch} | Loss: {loss:.6f} | {timestamp}")
            print(f"    VAE: {vae_symbol}  Drift: {drift_symbol}")
        except Exception as e:
            print(f"\n{i+1:2d}. {snap_path.name} (⚠️ Metadata unavailable)")
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

def interactive_snapshot_loader(trainer):
    # (full implementation as before - omitted for brevity, but can be included)
    pass

def configure_training_schedule_interactive():
    # (full implementation as before)
    pass

def continue_training_from_snapshot(trainer, remaining_epochs=None):
    # (full implementation as before)
    pass

# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    # Detect device and update global constants
    dm.DEVICE = get_device()
    dm.DTYPE = get_dtype_for_device(dm.DEVICE)
    configure_device_specific()
    initialize_for_device()
    
    print("\n" + "="*70)
    print("ENHANCED LABEL-CONDITIONED SCHRÖDINGER BRIDGE")
    print("WITH ORNSTEIN-UHLENBECK REFERENCE (mvOU-SBP)")
    print("="*70)
    print(f"Image Size: {dm.IMG_SIZE}x{dm.IMG_SIZE}")
    print(f"Latent: {dm.LATENT_CHANNELS}x{dm.LATENT_H}x{dm.LATENT_W}")
    print(f"Label Conditioning:  {dm.NUM_CLASSES} classes")
    print(f"Percentile Rescaling: {training.USE_PERCENTILE}")
    print(f"KPI Tracking: {training.USE_KPI_TRACKING}")
    print(f"Snapshot Learning: {training.USE_SNAPSHOTS}")
    print(f"Diversity Loss Weight: {training.DIVERSITY_WEIGHT}")
    print(f"Device: {dm.DEVICE}")
    print(f"Training Mode: {training.TRAINING_SCHEDULE['mode']}")
    print(f"OU Bridge: {'Enabled' if training.USE_OU_BRIDGE else 'Disabled'} (theta={training.OU_THETA})")
    print("="*70)
    
    if not check_device_compatibility():
        print("\n⚠️ Device compatibility warnings detected!")
        print("   Training may proceed with reduced performance.")
        print("   Consider using a different device if available.\n")
    
    print("\n MAIN MENU:")
    print("  1. Enhanced training (fresh start)")
    print("  2. Quick test (5 epochs)")
    print("  3. Export models to ONNX")
    print("  4. Generate samples from checkpoint")
    print("  5. Label-conditioned inference")
    print("  6. Snapshot management & recovery")
    print("  7. Resume from latest checkpoint")
    print("  8. Configure training schedule")
    print("  9. Toggle OU bridge (currently {})".format("ON" if training.USE_OU_BRIDGE else "OFF"))
    
    choice = input("\n Enter choice (1-9): ").strip()
    
    # Handle OU bridge toggle
    if choice == '9':
        training.USE_OU_BRIDGE = not training.USE_OU_BRIDGE
        print(f"OU bridge is now {'ON' if training.USE_OU_BRIDGE else 'OFF'}")
        # Re-enter main menu
        main()
        return
    
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, 64, 64))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    
    if choice == '1':
        epochs_input = input(f"Number of epochs [{training.EPOCHS}]: ").strip()
        epochs = int(epochs_input) if epochs_input else training.EPOCHS
        training.train_model(num_epochs=epochs)
    elif choice == '2':
        print("\n Running quick test (5 epochs)...")
        training.train_model(num_epochs=5)
    elif choice == '3':
        trainer = training.EnhancedLabelTrainer(dummy_loader)
        checkpoint_path = dm.DIRS["ckpt"] / "latest.pt"
        if checkpoint_path.exists():
            trainer.load_for_inference()
            trainer.export_onnx()
        else:
            print(" No trained model found!")
    elif choice == '4':
        trainer = training.EnhancedLabelTrainer(dummy_loader)
        checkpoint_path = dm.DIRS["ckpt"] / "latest.pt"
        if checkpoint_path.exists():
            trainer.load_for_inference()
            trainer.generate_samples()
            print(f"\n Samples saved to: {dm.DIRS['samples']}")
        else:
            print(" No trained model found!")
    elif choice == '5':
        inference.run_inference()
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
                    except:
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
            remaining = training.EPOCHS - trainer.epoch
            epochs_input = input(f"Additional epochs to train [{remaining}]: ").strip()
            additional = int(epochs_input) if epochs_input else remaining
            continue_training_from_snapshot(trainer, additional)
        else:
            print(" No checkpoint found!")
    elif choice == '8':
        configure_training_schedule_interactive()
        main()
    else:
        print(" Invalid choice")

if __name__ == "__main__":
    try:
        import scipy.stats as stats
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install scipy")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()