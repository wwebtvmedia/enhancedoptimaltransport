import streamlit as st
import os
import sys
import torch
import pandas as pd
import time
import json
import glob
from PIL import Image
import torchvision.utils as vutils
from pathlib import Path
import random

# MCP Imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from app_context import AppContext

# --- SWARM PROTOCOL UTILS ---
SWARM_DIR = Path("enhanced_label_sb/swarm")
SWARM_DIR.mkdir(parents=True, exist_ok=True)

def get_swarm_status():
    nodes = []
    for f in glob.glob(str(SWARM_DIR / "node_*.json")):
        try:
            with open(f, 'r') as j:
                nodes.append(json.load(j))
        except:
            continue
    return nodes

def update_node_status(node_id, metrics):
    status = {
        "id": node_id,
        "last_seen": time.time(),
        "epoch": metrics.get('epoch', 0),
        "total_loss": metrics.get('total', 0),
        "ssim": metrics.get('ssim_loss', 0),
        "mu_std": metrics.get('mu_std', 0),
        "score": metrics.get('composite_score', -100),
        "phase": metrics.get('phase', 1),
        "params": {
            "dw": config.DRIFT_WEIGHT,
            "cfg": config.CFG_SCALE,
            "sw": config.SSIM_WEIGHT
        }
    }
    with open(SWARM_DIR / f"node_{node_id}.json", 'w') as f:
        json.dump(status, f)

# --- UI LAYOUT ---
st.set_page_config(page_title="SB Swarm Intelligence", layout="wide", page_icon="🐝")

st.markdown("""
    <style>
    .node-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        background: #1e1e1e;
        margin-bottom: 10px;
    }
    .kpi-pos { color: #00ff00; }
    .kpi-neg { color: #ff4b4b; }
    </style>
""", unsafe_allow_state_html=True)

st.title("🐝 Schrödinger Bridge Swarm Dashboard")
st.sidebar.header("Swarm Control")

# Initialize Context
if 'ctx' not in st.session_state:
    st.session_state.ctx = AppContext()
ctx = st.session_state.ctx

# Header Metrics
nodes = get_swarm_status()
active_nodes = [n for n in nodes if time.time() - n['last_seen'] < 60]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Nodes", len(active_nodes))
if active_nodes:
    avg_score = sum(n['score'] for n in active_nodes) / len(active_nodes)
    avg_ssim = sum(n['ssim'] for n in active_nodes) / len(active_nodes)
    col2.metric("Swarm Score", f"{avg_score:.2f}")
    col3.metric("Avg SSIM", f"{avg_ssim:.3f}")
    col4.metric("Diversity Frontier", f"{sum(n['mu_std'] for n in active_nodes)/len(active_nodes):.2f}")

# Main Tabs
tabs = st.tabs(["🌎 Swarm Overview", "🖼️ Collective Gallery", "📈 Performance Heatmap", "⚙️ Global Config"])

with tabs[0]:
    st.subheader("Active Member Nodes")
    if not active_nodes:
        st.info("No active swarm nodes detected. Start nodes using `python main.py --training`.")
    
    # Show Nodes in a Grid
    cols = st.columns(3)
    for i, node in enumerate(active_nodes):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="node-card">
                <h4>Node: {node['id'][:8]}...</h4>
                <p><b>Phase:</b> {node['phase']} | <b>Epoch:</b> {node['epoch']}</p>
                <p><b>SSIM:</b> <span class="{'kpi-pos' if node['ssim'] < 0.2 else 'kpi-neg'}">{node['ssim']:.4f}</span></p>
                <p><b>Score:</b> {node['score']:.2f}</p>
                <hr>
                <small>DW: {node['params']['dw']:.2f} | CFG: {node['params']['cfg']:.1f} | SW: {node['params']['sw']:.1f}</small>
            </div>
            """, unsafe_allow_state_html=True)

with tabs[1]:
    st.subheader("Latest Generations across Swarm")
    sample_files = sorted(glob.glob("enhanced_label_sb/samples/gen_epoch*.png"), reverse=True)[:12]
    if sample_files:
        idx = st.slider("Timeline", 0, len(sample_files)-1, 0)
        st.image(sample_files[idx], use_column_width=True, caption=f"Collective Progress - Snapshot {idx}")
    else:
        st.write("Waiting for swarm samples...")

with tabs[2]:
    st.subheader("Swarm Loss Convergence")
    # Aggregate data for a combined chart
    history_files = glob.glob("enhanced_label_sb/metrics/history_*.json")
    all_data = []
    for f in history_files:
        try:
            with open(f, 'r') as j:
                d = json.load(j)
                node_id = Path(f).stem.replace("history_", "")
                df = pd.DataFrame(d)
                df['node'] = node_id
                all_data.append(df)
        except: continue
    
    if all_data:
        combined_df = pd.concat(all_data)
        st.line_chart(combined_df.pivot(columns='node', values='drift'))
    else:
        st.info("Insufficient data for heatmap.")

with tabs[3]:
    st.subheader("Global Parameter Broadcast")
    st.write("Send commands to all nodes in the swarm.")
    
    new_cfg = st.slider("Broadcast CFG Scale", 1.0, 15.0, float(config.CFG_SCALE))
    new_dw = st.slider("Broadcast Drift Weight", 0.1, 5.0, float(config.DRIFT_WEIGHT))
    
    if st.button("🚀 Push to All Nodes"):
        # Protocol: Write a global override file
        cmd = {"cfg": new_cfg, "dw": new_dw, "timestamp": time.time()}
        with open(SWARM_DIR / "global_override.json", 'w') as f:
            json.dump(cmd, f)
        st.success(f"Broadcasted CFG={new_cfg} and DW={new_dw} to the swarm.")

# Auto-refresh
time.sleep(2)
st.rerun()
