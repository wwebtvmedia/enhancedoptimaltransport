import streamlit as st
import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import queue
import threading
import logging
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils
from datetime import datetime
import importlib

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import training
import data_management as dm
import models

# ============================================================
# Streamlit Page Configuration
# ============================================================
st.set_page_config(
    page_title="Schrödinger Bridge Trainer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Google Material-like styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 4px;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 4px;
    }
    .css-12w0qpk {
        padding-top: 2rem;
    }
    div.stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

# ============================================================
# Helper Classes & Functions
# ============================================================
class StreamlitLogHandler(logging.Handler):
    def __init__(self, q):
        super().__init__()
        self.q = q
    def emit(self, record):
        self.q.put(self.format(record))

def load_config_to_state():
    for param in dir(config):
        if param.isupper() and not param.startswith('_'):
            if param not in st.session_state:
                st.session_state[f"cfg_{param}"] = getattr(config, param)

def save_state_to_config():
    for param in dir(config):
        if param.isupper() and not param.startswith('_'):
            key = f"cfg_{param}"
            if key in st.session_state:
                setattr(config, param, st.session_state[key])

# ============================================================
# Background Training Loop
# ============================================================
def run_training_thread(log_queue, stop_event):
    # Add streamlit logging handler
    handler = StreamlitLogHandler(log_queue)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    config.logger.addHandler(handler)
    
    try:
        # Initialize device
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
        else:
            config.DEVICE = torch.device("cpu")
            
        loader = dm.load_data()
        trainer = training.EnhancedLabelTrainer(loader)
        
        # Load latest checkpoint if exists
        latest = config.DIRS["ckpt"] / "latest.pt"
        if latest.exists():
            trainer.load_checkpoint()
            log_queue.put("🔄 Loaded existing checkpoint")

        total_epochs = config.EPOCHS
        for epoch in range(trainer.epoch, total_epochs):
            if st.session_state.stop_training:
                log_queue.put("⏹️ Training stopped by user")
                break
                
            trainer.epoch = epoch
            epoch_losses = trainer.train_epoch()
            
            # Send results to main thread
            log_queue.put({
                "type": "epoch_done",
                "epoch": epoch,
                "losses": epoch_losses
            })
            
            if (epoch+1) % 5 == 0:
                trainer.save_checkpoint()
                
            if (epoch+1) % 10 == 0:
                trainer.generate_samples()
                log_queue.put({"type": "update_gallery"})

        log_queue.put("✅ Training finished")
    except Exception as e:
        log_queue.put(f"❌ Error: {str(e)}")
    finally:
        config.logger.removeHandler(handler)
        st.session_state.training_running = False

# ============================================================
# Sidebar: Configuration
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    load_config_to_state()
    
    with st.expander("📁 Paths & Dimensions", expanded=False):
        st.session_state.cfg_IMG_SIZE = st.selectbox("Image Size", [32, 64, 96, 128], index=1 if config.IMG_SIZE==64 else 0)
        st.session_state.cfg_LATENT_CHANNELS = st.number_input("Latent Channels", value=config.LATENT_CHANNELS)
        st.session_state.cfg_BATCH_SIZE = st.number_input("Batch Size", value=config.BATCH_SIZE)

    with st.expander("⚡ Hyperparameters", expanded=True):
        st.session_state.cfg_LR = st.number_input("Learning Rate", value=config.LR, format="%.5f")
        st.session_state.cfg_EPOCHS = st.number_input("Total Epochs", value=config.EPOCHS)
        st.session_state.cfg_KL_WEIGHT = st.number_input("KL Weight", value=config.KL_WEIGHT, format="%.6f")
        st.session_state.cfg_RECON_WEIGHT = st.number_input("Recon Weight", value=config.RECON_WEIGHT, format="%.2f")
        st.session_state.cfg_DRIFT_WEIGHT = st.number_input("Drift Weight", value=config.DRIFT_WEIGHT, format="%.2f")

    if st.button("💾 Save & Apply", use_container_width=True):
        save_state_to_config()
        st.success("Config updated!")

# ============================================================
# Main UI Tabs
# ============================================================
st.title("Schrödinger Bridge Trainer")
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Training", "📚 Data Viewer", "🖼️ Gallery", "📊 Metrics"])

# --- TAB 1: Training ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        if not st.session_state.training_running:
            if st.button("▶️ Start Training", type="primary", use_container_width=True):
                st.session_state.training_running = True
                st.session_state.stop_training = False
                st.session_state.metrics_history = []
                # Start background thread
                thread = threading.Thread(target=run_training_thread, 
                                         args=(st.session_state.log_queue, None),
                                         daemon=True)
                thread.start()
                st.rerun()
        else:
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                st.session_state.stop_training = True
                st.info("Stopping after current epoch...")

        # Stats Cards
        if st.session_state.metrics_history:
            last_metrics = st.session_state.metrics_history[-1]
            st.metric("Current Epoch", f"{st.session_state.current_epoch + 1}")
            st.metric("Total Loss", f"{last_metrics.get('total', 0):.4f}")
            st.progress((st.session_state.current_epoch + 1) / config.EPOCHS)

    with col2:
        st.subheader("Live Log")
        log_container = st.empty()
        
        # Display logs from queue
        all_logs = []
        # In a real Streamlit app, we'd use a loop or st.empty for live updates
        # Since Streamlit reruns, we'll store logs in session state
        if 'log_buffer' not in st.session_state:
            st.session_state.log_buffer = []
            
        while not st.session_state.log_queue.empty():
            item = st.session_state.log_queue.get()
            if isinstance(item, dict):
                if item["type"] == "epoch_done":
                    st.session_state.current_epoch = item["epoch"]
                    st.session_state.metrics_history.append(item["losses"])
                elif item["type"] == "update_gallery":
                    st.toast("🖼️ New gallery samples generated!")
            else:
                st.session_state.log_buffer.append(item)
        
        # Show last 20 logs
        log_text = "\n".join(st.session_state.log_buffer[-20:])
        log_container.code(log_text if log_text else "Ready to start...")

# --- TAB 2: Data Viewer ---
with tab2:
    st.subheader("Dataset Preview")
    if st.button("🔄 Refresh Data Preview"):
        with st.spinner("Loading dataset..."):
            loader = dm.load_data()
            batch = next(iter(loader))
            images = batch['image'] if isinstance(batch, dict) else batch[0]
            # Create grid
            grid = vutils.make_grid(images[:16], nrow=4, normalize=True, value_range=(-1, 1))
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            st.image(im, caption="Random batch from training data", use_container_width=True)

# --- TAB 3: Gallery ---
with tab3:
    st.subheader("Latest Generations")
    samples_dir = config.DIRS["samples"]
    png_files = sorted(list(samples_dir.glob("gen_*.png")), key=os.path.getmtime, reverse=True)
    
    if png_files:
        selected_sample = st.selectbox("Select Sample", [f.name for f in png_files[:10]])
        img_path = samples_dir / selected_sample
        st.image(str(img_path), caption=f"Generated: {selected_sample}", use_container_width=True)
    else:
        st.info("No generated samples yet. Start training to see results.")

# --- TAB 4: Metrics ---
with tab4:
    st.subheader("Training Progress")
    if st.session_state.metrics_history:
        df = pd.DataFrame(st.session_state.metrics_history)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("### Total Loss")
            st.line_chart(df[['total']])
        with col_m2:
            st.write("### Reconstruction vs KL")
            st.line_chart(df[['recon', 'kl']])
            
        st.write("### Detailed Metrics Table")
        st.dataframe(df.tail(10))
    else:
        st.info("No metrics recorded yet.")

# Auto-refresh if training is running
if st.session_state.training_running:
    time.sleep(1)
    st.rerun()
