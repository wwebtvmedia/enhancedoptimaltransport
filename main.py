#!/usr/bin/env python3
"""
main.py - ENTRY POINT for Schrödinger Bridge Trainer
Uses the MCP (Model-Context-Protocol) design pattern to separate logic and display.
"""

import sys
import os
import logging
from pathlib import Path

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clean_gpu():
    """Aggressively reclaim GPU memory before startup."""
    try:
        import torch
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # Try to release memory held by any lingering tensors
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.ipc_collect()
            print("🧹 GPU Memory cleaned.")
    except Exception as e:
        print(f"⚠️ GPU clean failed: {e}")

# Clean GPU immediately before any significant logic
clean_gpu()

# Set environment variable for better memory management before torch import
if 'PYTORCH_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from app_context import AppContext
from app_processor import TrainingProcessor
import config

def setup_terminal_logging(context):
    """Bridge processing logs to terminal and context queue."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Check if a StreamHandler already exists
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in config.logger.handlers)
    
    if not has_stream_handler:
        # Terminal handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        config.logger.addHandler(ch)
    
    config.logger.setLevel(logging.INFO)

def run_headless_training(force_fresh: bool = False):
    """Run training in the terminal without a GUI."""
    ctx = AppContext()
    engine = TrainingProcessor(ctx)
    
    # Initialize hardware
    device_info = engine.initialize_hardware()
    print(f"🚀 Initializing Hardware: {device_info}")
    setup_terminal_logging(ctx)
    
    print("🧠 Starting Schrödinger Bridge Training (Headless Mode)...")
    print(f"📈 Total Epochs: {ctx.config.EPOCHS}")
    print(f"📂 Dataset: {ctx.config.BASE_DIR}")
    
    # Start training with a simple terminal callback
    def on_epoch_done(epoch, losses):
        print(f"\n✅ Epoch {epoch+1} complete!")
        print(f"   Total Loss: {losses.get('total', 0):.4f} | SNR: {losses.get('snr', 0):.2f}dB")

    engine.start_training(on_epoch_done=on_epoch_done, force_fresh=force_fresh)
    
    # Wait for training thread
    try:
        while engine.ctx.is_training:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping training safely...")
        engine.stop_training()
        while engine.ctx.is_training:
            time.sleep(0.5)
    
    print("🏁 Training session finished.")

def run_menu():
    """Show the full interactive CLI menu to the user."""
    # Ensure hardware is initialized
    ctx = AppContext()
    engine = TrainingProcessor(ctx)
    device_info = engine.initialize_hardware()
    
    while True:
        print("\n" + "="*60)
        print(f" 🧠 SCHRÖDINGER BRIDGE IMAGE GENERATOR 🧠 ")
        print(f" Hardware: {device_info}")
        print("="*60)
        print("  1. 🚀 Enhanced training (fresh start)")
        print("  2. ⚡ Quick test (5 epochs)")
        print("  3. 📤 Export models to ONNX")
        print("  4. 🖼️  Generate samples from latest checkpoint")
        print("  5. 🔮 Label-conditioned inference (Interactive)")
        print("  6. 📂 Snapshot management & recovery")
        print("  7. ⏯️  Resume from latest checkpoint")
        print("  8. 📅 Configure training schedule")
        print("  9. 🌉 Toggle OU bridge (currently {})".format("ON" if config.USE_OU_BRIDGE else "OFF"))
        print("  ----------------------------------------------------")
        print("  G. 🖥️  Launch Desktop GUI (Tkinter)")
        print("  S. 📱 Launch Web Dashboard (Streamlit)")
        print("  0. 🚪 Exit")
        print("="*60)
        
        choice = input("\nSelect an option: ").strip().lower()
        
        if choice == '1':
            print("\n🚀 Starting fresh training...")
            epochs_input = input(f"Enter TOTAL number of epochs [default {config.EPOCHS}]: ").strip()
            if epochs_input:
                config.EPOCHS = int(epochs_input)
            run_headless_training(force_fresh=True)
            break
            
        elif choice == '2':
            config.EPOCHS = 5
            print("\n⚡ Starting quick test (5 epochs)...")
            run_headless_training(force_fresh=True)
            break
            
        elif choice == '3':
            print("\n📤 Exporting models to ONNX...")
            import training
            import data_management as dm
            loader = dm.load_data()
            trainer = training.EnhancedLabelTrainer(loader)
            if trainer.load_for_inference():
                trainer.export_onnx()
            else:
                print("❌ Failed to load checkpoint for export.")
                
        elif choice == '4':
            print("\n🖼️ Generating samples...")
            import training
            import data_management as dm
            loader = dm.load_data()
            trainer = training.EnhancedLabelTrainer(loader)
            if trainer.load_for_inference():
                trainer.generate_samples()
            else:
                print("❌ Failed to load checkpoint for generation.")
                
        elif choice == '5':
            print("\n🔮 Launching Inference Interface...")
            import inference
            inference.run_inference()
            break
            
        elif choice == '6':
            manage_snapshots()
            
        elif choice == '7':
            print("\n⏯️ Resuming from latest checkpoint...")
            
            # Check current progress to provide helpful default
            import data_management as dm
            ckpt_path = config.DIRS["ckpt"] / "latest.pt"
            current_epoch = 0
            current_step = 0
            if ckpt_path.exists():
                try:
                    import torch, collections
                    from torch.serialization import add_safe_globals
                    add_safe_globals([collections.defaultdict])
                    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    current_epoch = checkpoint.get('epoch', 0)
                    current_step = checkpoint.get('step', 0)
                    print(f"Current progress: Epoch {current_epoch+1} (Step {current_step:,})")
                except Exception as e:
                    print(f"⚠️ Could not read checkpoint metadata: {e}")
            
            target_input = input(f"Enter TOTAL target epochs [current: {current_epoch+1}, config default: {config.EPOCHS}]: ").strip()
            if target_input:
                config.EPOCHS = int(target_input)
            
            if config.EPOCHS <= current_epoch:
                print(f"❌ Error: Target epoch ({config.EPOCHS}) must be greater than current epoch ({current_epoch+1}).")
                continue

            print(f"🚀 Resuming training until TOTAL epoch {config.EPOCHS}...")
            run_headless_training()
            break
            
        elif choice == '8':
            configure_schedule()
            
        elif choice == '9':
            config.USE_OU_BRIDGE = not config.USE_OU_BRIDGE
            print("\n🌉 OU Bridge toggled to: {}".format("ON" if config.USE_OU_BRIDGE else "OFF"))
            
        elif choice == 'g':
            print("\n🖥️ Launching Tkinter GUI...")
            import appmain_tk
            appmain_tk.main()
            break
            
        elif choice == 's':
            print("\n📱 Launching Streamlit Server...")
            os.system("streamlit run app_streamlit.py")
            break
            
        elif choice == '0':
            print("\nGoodbye! 👋")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please try again.")

def manage_snapshots():
    """Sub-menu for snapshot management."""
    import training
    import data_management as dm
    loader = dm.load_data()
    trainer = training.EnhancedLabelTrainer(loader)
    
    while True:
        snapshots = trainer.list_available_snapshots()
        print("\n" + "-"*40)
        print(" 📂 SNAPSHOT MANAGEMENT")
        print("-"*40)
        if not snapshots:
            print(" No snapshots found in enhanced_label_sb/snapshots/")
            return

        for i, snap in enumerate(snapshots):
            info = trainer.inspect_snapshot(snap)
            print(f" {i+1}. Epoch {info['epoch']} | Loss: {info['loss']:.4f} | {snap.name}")
        
        print(f" {len(snapshots)+1}. Back to Main Menu")
        print("-"*40)
        
        choice = input("\nSelect a snapshot to load or go back: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if idx == len(snapshots):
                return
            if 0 <= idx < len(snapshots):
                snap_path = snapshots[idx]
                print(f"\n📂 Loading snapshot: {snap_path.name}")
                if trainer.load_from_snapshot(snap_path):
                    print("✅ Snapshot loaded successfully.")
                    print("You can now start training (Option 1) or resume (Option 7) to use this state.")
                    # We need to update the global trainer state if we want to resume training from here
                    # For now, just telling the user it's loaded.
                    return
        print("Invalid choice.")

def configure_schedule():
    """Sub-menu for training schedule configuration."""
    print("\n" + "-"*40)
    print(" 📅 CONFIGURE TRAINING SCHEDULE")
    print("-"*40)
    print(" 1. Auto (Phase 1 -> Phase 2 at Epoch 50)")
    print(" 2. Manual (Force Phase 1, 2, or 3)")
    print(" 3. Alternate (VAE and Drift every N epochs)")
    print(" 4. Three Phase (VAE -> Drift -> Both)")
    print(" 5. Back to Main Menu")
    print("-"*40)
    
    choice = input("\nSelect an option: ").strip()
    if choice in ['1', '2', '3', '4']:
        config.SCHEDULE_MANUALLY_SET = True
        
    if choice == '1':
        config.TRAINING_SCHEDULE['mode'] = 'auto'
    elif choice == '2':
        config.TRAINING_SCHEDULE['mode'] = 'manual'
        ph = input("Enter phase (1: VAE, 2: Drift, 3: Both): ").strip()
        config.TRAINING_SCHEDULE['force_phase'] = int(ph)
    elif choice == '3':
        config.TRAINING_SCHEDULE['mode'] = 'alternate'
        freq = input("Enter frequency (epochs per switch): ").strip()
        config.TRAINING_SCHEDULE['alternate_freq'] = int(freq)
    elif choice == '4':
        config.TRAINING_SCHEDULE['mode'] = 'three_phase'
    elif choice == '5':
        return
    
    print(f"✅ Training schedule updated to: {config.TRAINING_SCHEDULE['mode']}")

if __name__ == "__main__":
    # Argument mapping for flexibility
    args = sys.argv[1:]
    
    # DEFAULT: If no args provided, start training directly
    if not args or "--training" in args or "--train" in args:
        run_headless_training()
    elif "--menu" in args:
        run_menu()
    elif "--gui" in args:
        print("🖥️ Launching Tkinter GUI...")
        import appmain_tk
        appmain_tk.main()
    elif "--streamlit" in args:
        print("📱 Launching Streamlit Server...")
        os.system("streamlit run app_streamlit.py")
    elif "--inference" in args:
        print("🔮 Launching Inference Interface...")
        import inference
        import torch
        # Initialize hardware
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config.DEVICE = torch.device("mps")
        else:
            config.DEVICE = torch.device("cpu")
        inference.run_inference()
    else:
        print(f"❓ Unknown arguments: {args}")
        print("Usage:")
        print("  python main.py              (Default: Start Training)")
        print("  python main.py --menu       (Show Interactive Menu)")
        print("  python main.py --inference  (Start Inference Interface)")
        print("  python main.py --gui        (Launch Desktop GUI)")
        print("  python main.py --streamlit  (Launch Web Dashboard)")
        sys.exit(1)
