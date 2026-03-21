# ============================================================================
# SHARED DISPLAY AND LOGIC UTILITIES FOR SCHRÖDINGER BRIDGE GUI
# ============================================================================

import re
import os
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# ============================================================
# Google Material Design Colors - Light Theme (Shared)
# ============================================================
class Colors:
    BG_DARK = "#f8f9fa"      # Google Gray 50
    BG_MEDIUM = "#ffffff"    # Pure White
    BG_LIGHT = "#ffffff"     # White
    FG = "#202124"           # Google Gray 900 (Text)
    FG_SECONDARY = "#5f6368" # Google Gray 700 (Secondary Text)
    ACCENT = "#1a73e8"       # Google Blue
    ACCENT2 = "#a142f4"      # Google Purple
    SUCCESS = "#1e8e3e"      # Google Green
    WARNING = "#f9ab00"      # Google Yellow
    ERROR = "#d93025"        # Google Red
    BORDER = "#dadce0"       # Google Gray 300
    CARD_SHADOW = "#e8eaed"  # Subtle shadow color

# ============================================================
# Shared Parameter Descriptions
# ============================================================
def get_param_description(param: str) -> str:
    """Get description for parameter tooltip/help"""
    descriptions = {
        "IMG_SIZE": "Input image size (pixels). Default 96x96 for STL-10.",
        "LATENT_CHANNELS": "Number of channels in latent space. Standard is 4.",
        "NUM_CLASSES": "Number of classes for label conditioning (e.g., 10 for STL-10).",
        "LR": "Learning rate for the optimizers.",
        "EPOCHS": "Total number of training epochs.",
        "BATCH_SIZE": "Batch size (auto-adjusted for hardware).",
        "KL_WEIGHT": "Weight for KL divergence loss. Crucial for latent space structure.",
        "RECON_WEIGHT": "Weight for image reconstruction loss (MSE + SSIM + Perc).",
        "DRIFT_WEIGHT": "Weight for drift network training loss.",
        "DIVERSITY_WEIGHT": "Weight for latent channel diversity loss (prevents collapse).",
        "CONSISTENCY_WEIGHT": "Weight for reference encoder consistency during Phase 2.",
        "USE_OU_BRIDGE": "Whether to use Ornstein-Uhlenbeck reference process.",
        "USE_FOURIER_FEATURES": "Add Fourier features to encoder input for detail.",
        "TARGET_SNR": "Target Signal-to-Noise ratio for quality tracking.",
        "LATENT_SCALE": "Multiplier for latent space range.",
        "FREE_BITS": "KL divergence threshold per sample to avoid collapse.",
    }
    return descriptions.get(param, f"Configuration parameter: {param}")

# ============================================================
# Shared Log Parsing Logic
# ============================================================
def parse_training_log(filename: str) -> Dict[int, Dict[str, Any]]:
    """Parse a training log file and return a dictionary of metrics per epoch."""
    metrics = {}
    
    if not os.path.exists(filename):
        return metrics

    # Improved Regex for multi-word loss names
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) complete:')
    loss_pattern = re.compile(r'  ([\w\s]+ loss): ([\d\.eE+-]+)')
    snr_pattern = re.compile(r'  SNR: ([\d\.]+)dB')
    latent_std_pattern = re.compile(r'  Latent std: ([\d\.]+)')

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        current_epoch = None
        for line in lines:
            m = epoch_pattern.search(line)
            if m:
                current_epoch = int(m.group(1)) - 1 # Zero-indexed for consistency
                metrics[current_epoch] = {}
                continue

            if current_epoch is not None:
                m = loss_pattern.search(line)
                if m:
                    # Clean key name: 'Total loss' -> 'total'
                    key = m.group(1).lower().replace(' loss', '').strip()
                    val = float(m.group(2))
                    metrics[current_epoch][key] = val
                
                m = snr_pattern.search(line)
                if m:
                    metrics[current_epoch]['snr'] = float(m.group(1))
                
                m = latent_std_pattern.search(line)
                if m:
                    metrics[current_epoch]['latent_std'] = float(m.group(1))
                    
    except Exception as e:
        print(f"Log parsing error: {e}")
        
    return metrics

# ============================================================
# Shared Metrics Formatting
# ============================================================
def format_metrics_text(epoch: int, loss_dict: Dict[str, Any]) -> str:
    """Format metrics dictionary into a readable multi-line string."""
    text = f"📊 Epoch {epoch + 1}\n"
    text += "═" * 40 + "\n"
    
    # Sort keys to keep display consistent
    priority_keys = ['total', 'recon', 'kl', 'diversity', 'drift', 'snr', 'latent_std']
    other_keys = sorted([k for k in loss_dict.keys() if k not in priority_keys])
    
    for key in priority_keys + other_keys:
        if key in loss_dict:
            value = loss_dict[key]
            if isinstance(value, (int, float)):
                if key == 'snr':
                    text += f"📈 {key:20s}: {value:8.2f} dB\n"
                elif key in ['kl', 'diversity']:
                    text += f"📊 {key:20s}: {value:8.6f}\n"
                else:
                    text += f"📉 {key:20s}: {value:8.4f}\n"
                    
    return text

def get_latest_log_path(log_dir: Path) -> Path:
    """Find the most recent log file in the directory."""
    log_files = list(log_dir.glob("train_*.log"))
    if not log_files:
        return None
    return max(log_files, key=lambda p: p.stat().st_mtime)
