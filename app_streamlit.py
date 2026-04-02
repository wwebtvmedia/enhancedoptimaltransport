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
    
    with st.expander("📁 Dataset Settings", expanded=True):
        dataset_name = st.selectbox("Dataset", ["STL10", "CIFAR10", "CUSTOM"], 
                                   index=["STL10", "CIFAR10", "CUSTOM"].index(config.DATASET_NAME))
        dataset_path = st.text_input("Dataset Path", value=str(config.DATASET_PATH))
        img_size = st.number_input("Model Input Size (IMG_SIZE)", value=config.IMG_SIZE, step=8)
        gen_size = st.number_input("Generation Size (GEN_SIZE)", value=config.GEN_SIZE, step=8)
        
        # Apply updates
        config.DATASET_NAME = dataset_name
        config.DATASET_PATH = Path(dataset_path)
        if img_size != config.IMG_SIZE:
            config.IMG_SIZE = img_size
            config.LATENT_H = img_size // 8
            config.LATENT_W = img_size // 8
            config.LATENT_DIM = config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W
        config.GEN_SIZE = gen_size

    with st.expander("🚀 Hyperparameters", expanded=False):
        config.LR = st.number_input("Learning Rate", value=config.LR, format="%.6f", help=get_param_description("LR"))
        config.EPOCHS = st.number_input("Epochs", value=config.EPOCHS, help=get_param_description("EPOCHS"))
        config.BATCH_SIZE = st.number_input("Batch Size", value=config.BATCH_SIZE, help=get_param_description("BATCH_SIZE"))
        config.CST_COEF_GAUSSIAN_PRIO = st.number_input("Gaussian Prior Std", value=config.CST_COEF_GAUSSIAN_PRIO, format="%.2f", help=get_param_description("CST_COEF_GAUSSIAN_PRIO"))
            
    with st.expander("⚖️ Loss Weights", expanded=False):
        config.KL_WEIGHT = st.number_input("KL Weight", value=config.KL_WEIGHT, format="%.6f", help=get_param_description("KL_WEIGHT"))
        config.RECON_WEIGHT = st.number_input("Recon Weight", value=config.RECON_WEIGHT, format="%.2f", help=get_param_description("RECON_WEIGHT"))
        config.DIVERSITY_WEIGHT = st.number_input("Diversity Weight", value=config.DIVERSITY_WEIGHT, format="%.2f", help=get_param_description("DIVERSITY_WEIGHT"))

    with st.expander("📅 Training Schedule", expanded=False):
        mode = st.selectbox("Schedule Mode", ["auto", "manual", "three_phase", "alternate"], 
                           index=["auto", "manual", "three_phase", "alternate"].index(config.TRAINING_SCHEDULE['mode']))
        
        force_phase = config.TRAINING_SCHEDULE['force_phase'] or 1
        new_force_phase = st.radio("Manual Phase (if manual mode)", [1, 2, 3], 
                                  index=[1, 2, 3].index(force_phase),
                                  format_func=lambda x: f"Phase {x} ({['VAE', 'Drift', 'Both'][x-1]})")
        
        config.TRAINING_SCHEDULE['mode'] = mode
        config.TRAINING_SCHEDULE['force_phase'] = new_force_phase
        
        st.caption(f"Current Phase in Engine: {getattr(engine.trainer, 'phase', 'N/A') if engine.trainer else 'N/A'}")

    if st.button("💾 Apply & Save", width='stretch'):
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
            if st.button("▶️ Start Training", type="primary", width='stretch'):
                engine.start_training()
                st.rerun()
        else:
            if st.button("⏹️ Stop Training", type="secondary", width='stretch'):
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
        if st.button("🔄 Refresh Batch", width='stretch'):
            with st.spinner("Fetching batch..."):
                loader = dm.load_data()
                batch = next(iter(loader))
                imgs = batch['image'] if isinstance(batch, dict) else batch[0]
                grid = vutils.make_grid(imgs[:16], nrow=4, normalize=True, value_range=(-1, 1))
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                st.session_state.data_preview = Image.fromarray(ndarr)
    
    with col_d2:
        if 'data_preview' in st.session_state:
            st.image(st.session_state.data_preview, caption="Training Data Batch (96x96)", width='stretch')

# --- TAB 3: Gallery ---
with tab3:
    st.subheader("Latest Generated Samples")
    samples_dir = config.DIRS["samples"]
    png_files = sorted(list(samples_dir.glob("gen_*.png")), key=os.path.getmtime, reverse=True)
    if png_files:
        st.image(str(png_files[0]), caption=f"Latest Result: {png_files[0].name}", width='stretch')
    else:
        st.warning("No samples generated yet.")

# --- TAB 4: Curves & Logs (Restored) ---
with tab4:
    st.subheader("Historical Training Curves")
    
    # Log File Selection
    log_dir = config.DIRS["logs"]
    log_files = sorted(list(log_dir.glob("train_*.log")), key=os.path.getmtime, reverse=True)
    
    if log_files:
        log_names = [f.name for f in log_files]
        selected_log_name = st.selectbox("Select Log File to Analyze", log_names, index=0)
        selected_log_path = log_dir / selected_log_name
        
        st.caption(f"Analyzing log: {selected_log_path.name}")
        metrics = parse_training_log(str(selected_log_path))
        
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
            st.dataframe(df.tail(10), width='stretch')
        else:
            st.error("Could not parse metrics from the latest log.")
    else:
        st.info("No log files found to visualize.")

# Auto-refresh loop
if ctx.is_training:
    time.sleep(2)
    st.rerun()
