import streamlit as st
import os
import sys
import torch
import pandas as pd
import time
from PIL import Image
import torchvision.utils as vutils
from pathlib import Path

# MCP Imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import data_management as dm
from app_context import AppContext
from app_processor import TrainingProcessor
from appmain_display import Colors, parse_training_log, get_param_description, get_latest_log_path

# ============================================================
# Page Setup
# ============================================================
st.set_page_config(
    page_title="Schrödinger Bridge Trainer",
    page_icon="📊",
    layout="wide"
)

# Initialize MCP Layers in Session State
if 'ctx' not in st.session_state:
    st.session_state.ctx = AppContext()
if 'engine' not in st.session_state:
    st.session_state.engine = TrainingProcessor(st.session_state.ctx)
    st.session_state.engine.initialize_hardware()

ctx = st.session_state.ctx
engine = st.session_state.engine

# Google Material Styling
st.markdown(f"""
    <style>
    .main {{ background-color: {Colors.BG_DARK}; }}
    div.stButton > button:first-child {{
        background-color: {Colors.ACCENT};
        color: white;
        border-radius: 4px;
        border: none;
    }}
    .metric-card {{
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid {Colors.BORDER};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar - Configuration (MCP Protocol)
# ============================================================
with st.sidebar:
    st.title("⚙️ Config")
    st.info(f"Hardware: {ctx.device_info}")
    
    with st.expander("🚀 Hyperparameters", expanded=True):
        for p in ["LR", "EPOCHS", "BATCH_SIZE"]:
            val = st.number_input(p, value=getattr(config, p), help=get_param_description(p))
            setattr(config, p, val)
            
    with st.expander("⚖️ Loss Weights", expanded=False):
        for p in ["KL_WEIGHT", "RECON_WEIGHT", "DIVERSITY_WEIGHT"]:
            val = st.number_input(p, value=getattr(config, p), format="%.6f", help=get_param_description(p))
            setattr(config, p, val)

    if st.button("💾 Apply & Save", use_container_width=True):
        st.success("Configuration updated in memory!")

# ============================================================
# Main UI
# ============================================================
st.title("Schrödinger Bridge Trainer")
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Training", "📚 Training Data", "🖼️ Gallery", "📊 Curves & Logs"])

# --- TAB 1: Training (Engine Control) ---
with tab1:
    col_ctrl, col_stats = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Controls")
        if not ctx.is_training:
            if st.button("▶️ Start Training", type="primary", use_container_width=True):
                engine.start_training()
                st.rerun()
        else:
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                engine.stop_training()
                st.info("Stopping after current epoch...")
        
        if ctx.is_training:
            st.write(f"### Epoch {ctx.current_epoch + 1}")
            st.progress((ctx.current_epoch + 1) / config.EPOCHS)

    with col_stats:
        st.subheader("Live Status")
        if ctx.is_training:
            st.success("Engine is running...")
            # Display latest log message if any
            if not ctx.log_queue.empty():
                st.code(ctx.log_queue.get())
        else:
            st.info("System idle. Ready to start.")

# --- TAB 2: Training Data (Restored) ---
with tab2:
    st.subheader("Dataset Inspector")
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.write("View the actual images the model is learning from.")
        if st.button("🔄 Refresh Batch", use_container_width=True):
            with st.spinner("Fetching batch..."):
                loader = dm.load_data()
                batch = next(iter(loader))
                imgs = batch['image'] if isinstance(batch, dict) else batch[0]
                grid = vutils.make_grid(imgs[:16], nrow=4, normalize=True, value_range=(-1, 1))
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                st.session_state.data_preview = Image.fromarray(ndarr)
    
    with col_d2:
        if 'data_preview' in st.session_state:
            st.image(st.session_state.data_preview, caption="Training Data Batch (96x96)", use_container_width=True)

# --- TAB 3: Gallery ---
with tab3:
    st.subheader("Latest Generated Samples")
    samples_dir = config.DIRS["samples"]
    png_files = sorted(list(samples_dir.glob("gen_*.png")), key=os.path.getmtime, reverse=True)
    if png_files:
        st.image(str(png_files[0]), caption=f"Latest Result: {png_files[0].name}", use_container_width=True)
    else:
        st.warning("No samples generated yet.")

# --- TAB 4: Curves & Logs (Restored) ---
with tab4:
    st.subheader("Historical Training Curves")
    latest_log = get_latest_log_path(config.DIRS["logs"])
    
    if latest_log:
        st.caption(f"Analyzing log: {latest_log.name}")
        metrics = parse_training_log(str(latest_log))
        
        if metrics:
            df = pd.DataFrame.from_dict(metrics, orient='index')
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("### Loss Convergence")
                # Filter to show main losses
                loss_cols = [c for c in ['total', 'recon', 'kl', 'diversity'] if c in df.columns]
                st.line_chart(df[loss_cols])
            
            with c2:
                st.write("### Quality Metrics (SNR)")
                if 'snr' in df.columns:
                    st.line_chart(df[['snr']])
                
            st.write("### Raw Epoch Data")
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.error("Could not parse metrics from the latest log.")
    else:
        st.info("No log files found to visualize.")

# Auto-refresh loop
if ctx.is_training:
    time.sleep(2)
    st.rerun()
