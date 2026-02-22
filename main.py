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
# INTERACTIVE SNAPSHOT UTILITIES
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
    """Interactive menu for loading snapshots. Returns (snapshot_path, load_vae, load_drift, phase) if continuing, else None."""
    print("\n" + "="*70)
    print(" SNAPSHOT LOADING MENU")
    print("="*70)
    while True:
        print("\n📋 Options:")
        print("  1. List all snapshots")
        print("  2. Inspect a snapshot")
        print("  3. Compare two snapshots")
        print("  4. Load snapshot for training")
        print("  5. Restart from VAE (Phase 2 only)")
        print("  6. Configure training schedule")
        print("  7. Back to main menu")
        choice = input("\n Enter choice (1-7): ").strip()
        if choice == '1':
            trainer.list_available_snapshots()
        elif choice == '2':
            snapshot_path = select_snapshot_interactive()
            if snapshot_path:
                trainer.inspect_snapshot(snapshot_path)
        elif choice == '3':
            print("\n Select first snapshot:")
            snap1 = select_snapshot_interactive()
            if snap1:
                print("\n Select second snapshot:")
                snap2 = select_snapshot_interactive()
                if snap2:
                    trainer.compare_snapshots(snap1, snap2)
        elif choice == '4':
            snapshot_path = select_snapshot_interactive()
            if not snapshot_path:
                continue
            print("\nLoading options:")
            print("  1. Load VAE only")
            print("  2. Load Drift only")
            print("  3. Load both VAE and Drift")
            print("  4. Cancel")
            load_choice = input(" Choose (1-4): ").strip()
            if load_choice == '4':
                continue
            load_vae = load_choice in ['1', '3']
            load_drift = load_choice in ['2', '3']
            phase_choice = input("\n Force training phase? (1=VAE, 2=Drift, 0=Auto): ").strip()
            phase = int(phase_choice) if phase_choice in ['1', '2'] else None
            print("\n" + "="*70)
            print("⚡ LOADING SNAPSHOT...")
            print("="*70)
            success = trainer.load_from_snapshot(
                snapshot_path=snapshot_path,
                load_vae=load_vae,
                load_drift=load_drift,
                phase=phase
            )
            if success:
                print("\n Snapshot loaded successfully!")
                print(f"   Current epoch: {trainer.epoch}")
                continue_training = input("\n🎯 Continue training from this point? (y/n): ").strip().lower()
                if continue_training == 'y':
                    return (snapshot_path, load_vae, load_drift, phase)
            else:
                print("\n Failed to load snapshot")
        elif choice == '5':
            print("\n Restart from VAE snapshot (Phase 2 only)")
            print("This will load a VAE snapshot and start fresh Drift training")
            snapshot_path = select_snapshot_interactive()
            if not snapshot_path:
                continue
            print("\n" + "="*70)
            print("⚡ RESTARTING FROM VAE SNAPSHOT...")
            print("="*70)
            success = trainer.restart_from_vae_snapshot(snapshot_path)
            if success:
                print("\n VAE snapshot loaded successfully!")
                print(f"   Current epoch: {trainer.epoch}")
                print(f"   Ready for Phase 2 (Drift) training")
                continue_training = input("\n Start Phase 2 training? (y/n): ").strip().lower()
                if continue_training == 'y':
                    return (snapshot_path, True, False, 2)
            else:
                print("\n Failed to restart from VAE snapshot")
        elif choice == '6':
            configure_training_schedule_interactive()
        elif choice == '7':
            break
        else:
            print(" Invalid choice")
    return None

def configure_training_schedule_interactive():
    """Interactive menu to configure training schedule."""
    print("\n" + "="*70)
    print("CONFIGURE TRAINING SCHEDULE")
    print("="*70)
    print(f"\nCurrent mode: {training.TRAINING_SCHEDULE['mode']}")
    print("\nAvailable modes:")
    print("  1. auto - Switch at specified epoch")
    print("  2. manual - Force VAE or Drift only")
    print("  3. custom - Custom schedule per epoch")
    print("  4. alternate - Alternate every N epochs")
    print("  5. vae_only - Train VAE only")
    print("  6. drift_only - Train Drift only")
    print("  7. Keep current")
    mode_choice = input("\nSelect mode (1-7): ").strip()
    if mode_choice == '1':
        switch_epoch = input(f"Switch epoch [current: {training.TRAINING_SCHEDULE.get('switch_epoch', 50)}]: ").strip()
        switch_epoch = int(switch_epoch) if switch_epoch else training.TRAINING_SCHEDULE.get('switch_epoch', 50)
        training.configure_training_schedule(mode='auto', switch_epoch=switch_epoch)
    elif mode_choice == '2':
        force_phase = input("Force phase (1=VAE, 2=Drift): ").strip()
        if force_phase in ['1', '2']:
            training.configure_training_schedule(mode='manual', force_phase=int(force_phase))
    elif mode_choice == '3':
        print("\nEnter custom schedule as epoch:phase pairs (e.g., 10:1,20:2,30:1)")
        schedule_str = input("Schedule: ").strip()
        custom_schedule = {}
        if schedule_str:
            for pair in schedule_str.split(','):
                if ':' in pair:
                    e, p = pair.split(':')
                    custom_schedule[int(e)] = int(p)
            training.configure_training_schedule(mode='custom', custom_schedule=custom_schedule)
    elif mode_choice == '4':
        alt_freq = input(f"Alternate frequency [current: {training.TRAINING_SCHEDULE.get('alternate_freq', 5)}]: ").strip()
        alt_freq = int(alt_freq) if alt_freq else training.TRAINING_SCHEDULE.get('alternate_freq', 5)
        training.configure_training_schedule(mode='alternate', alternate_freq=alt_freq)
    elif mode_choice == '5':
        training.configure_training_schedule(mode='vae_only')
    elif mode_choice == '6':
        training.configure_training_schedule(mode='drift_only')
    elif mode_choice == '7':
        return
    print(f"\n Training schedule updated: {training.TRAINING_SCHEDULE['mode']}")

def continue_training_from_snapshot(trainer, remaining_epochs=None):
    """Continue training from currently loaded snapshot state."""
    if remaining_epochs is None:
        max_epochs = input(f"\n Train until epoch (current: {trainer.epoch}, max: {training.EPOCHS}): ").strip()
        if max_epochs:
            try:
                target_epoch = int(max_epochs)
                remaining_epochs = target_epoch - trainer.epoch
            except:
                remaining_epochs = training.EPOCHS - trainer.epoch
        else:
            remaining_epochs = training.EPOCHS - trainer.epoch
    if remaining_epochs <= 0:
        print(" Target epoch must be greater than current epoch")
        return
    print(f"\n Continuing training for {remaining_epochs} epochs...")
    print(f"   Current epoch: {trainer.epoch + 1}")
    print(f"   Target epoch: {trainer.epoch + remaining_epochs}")
    print("="*70)
    for epoch in range(trainer.epoch, trainer.epoch + remaining_epochs):
        trainer.epoch = epoch
        epoch_losses = trainer.train_epoch()
        if training.USE_KPI_TRACKING:
            kpi_update = {}
            kpi_update['lr_vae'] = trainer.opt_vae.param_groups[0]['lr']
            kpi_update['lr_drift'] = trainer.opt_drift.param_groups[0]['lr']
            for key in ['snr', 'latent_std', 'min_channel_std', 'recon', 'kl', 'diversity']:
                if key in epoch_losses:
                    kpi_update[key] = epoch_losses[key]
            if trainer.phase == 1:
                if 'total' in epoch_losses:
                    kpi_update['loss'] = epoch_losses['total']
                if 'recon' in epoch_losses:
                    kpi_update['recon_loss'] = epoch_losses['recon']
                if 'kl' in epoch_losses:
                    kpi_update['kl_loss'] = epoch_losses['kl']
                if 'diversity' in epoch_losses:
                    kpi_update['diversity_loss'] = epoch_losses['diversity']
            else:
                if 'drift' in epoch_losses:
                    kpi_update['loss'] = epoch_losses['drift']
                    kpi_update['drift_loss'] = epoch_losses['drift']
                if 'consistency' in epoch_losses:
                    kpi_update['consistency_loss'] = epoch_losses['consistency']
                if 'temperature' in epoch_losses:
                    kpi_update['temperature'] = epoch_losses['temperature']
            if len(kpi_update) > 2:
                trainer.kpi_tracker.update(kpi_update)
        if trainer.phase == 1:
            current_total_loss = epoch_losses.get('total', float('inf'))
        else:
            current_total_loss = epoch_losses.get('drift', float('inf'))
        if current_total_loss < trainer.best_loss and current_total_loss != float('inf'):
            trainer.best_loss = current_total_loss
            trainer.save_checkpoint(is_best=True)
        elif (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(is_best=False)
        if (epoch + 1) % 10 == 0 and current_total_loss != float('inf'):
            logger.info("Generating samples...")
            trainer.generate_samples()
        if training.USE_KPI_TRACKING and trainer.phase == 2:
            if trainer.kpi_tracker.should_stop(phase=trainer.phase):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    print(f"\n Training complete! Stopped at epoch {trainer.epoch + 1}")
    print(f"   Best loss achieved: {trainer.best_loss:.4f}")

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
    print("WITH ANTI-COLLAPSE & IMPROVED DRIFT")
    print("="*70)
    print(f"Image Size: {dm.IMG_SIZE}x{dm.IMG_SIZE}")
    print(f"Latent: {dm.LATENT_CHANNELS}x{dm.LATENT_H}x{dm.LATENT_W}")
    print(f"Label Conditioning:  ({dm.NUM_CLASSES} classes)")
    print(f"Percentile Rescaling: {training.USE_PERCENTILE}")
    print(f"KPI Tracking: {training.USE_KPI_TRACKING}")
    print(f"Snapshot Learning: {training.USE_SNAPSHOTS}")
    print(f"Diversity Loss Weight: {training.DIVERSITY_WEIGHT}")
    print(f"Device: {dm.DEVICE}")
    print(f"Training Mode: {training.TRAINING_SCHEDULE['mode']}")
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
    
    choice = input("\n Enter choice (1-8): ").strip()
    
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