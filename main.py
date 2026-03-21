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
from app_context import AppContext
from app_processor import TrainingProcessor
import config

def setup_terminal_logging(context):
    """Bridge processing logs to terminal and context queue."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Terminal handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    config.logger.handlers.clear()
    config.logger.addHandler(ch)
    config.logger.setLevel(logging.INFO)

def run_headless_training():
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

    engine.start_training(on_epoch_done=on_epoch_done)
    
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

if __name__ == "__main__":
    # Check if we should launch GUI or stay in terminal
    if "--gui" in sys.argv:
        print("🖥️ Launching Tkinter GUI...")
        import appmain_tk
        appmain_tk.main()
    elif "--streamlit" in sys.argv:
        print("📱 Launching Streamlit Server...")
        os.system("streamlit run app_streamlit.py")
    else:
        run_headless_training()
