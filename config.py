# ============================================================================
# CENTRALIZED CONFIGURATION FOR SCHRÖDINGER BRIDGE
# ============================================================================

import os
import torch
import logging
from pathlib import Path
from datetime import datetime

# ============================================================
# MODEL ARCHITECTURE CONSTANTS
# ============================================================
IMG_SIZE = 64
LATENT_CHANNELS = 4
LATENT_H = IMG_SIZE // 8
LATENT_W = IMG_SIZE // 8
LATENT_DIM = LATENT_CHANNELS * LATENT_H * LATENT_W

NUM_CLASSES = 6000
LABEL_EMB_DIM = 128

# ============================================================
# TRAINING HYPERPARAMETERS (will be adjusted by device)
# ============================================================
BATCH_SIZE = 64
LR = 2e-4
EPOCHS = 200
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Loss weights
KL_WEIGHT = 0.001
RECON_WEIGHT = 1.0
DRIFT_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.01
CONSISTENCY_WEIGHT = 0.5

# Fourier feature positional encoding (for VAE encoder)
USE_FOURIER_FEATURES = True          # set to False to disable
FOURIER_FREQS = [1, 2, 4, 8, 16]     # frequencies (multiples of π)

# ============================================================
# TRAINING PHASE CONTROL
# ============================================================
SWITCH_EPOCH = 50  # Default epoch to switch from VAE to Drift
TRAINING_SCHEDULE = {
    'mode': 'auto',  # 'auto', 'manual', 'custom', 'alternate'
    'force_phase': None,  # None, 1 (VAE only), 2 (Drift only)
    'custom_schedule': {},  # {epoch: phase} for custom schedules
    'switch_epoch': SWITCH_EPOCH,
    'alternate_freq': 5,
}

# ============================================================
# INFERENCE PARAMETERS
# ============================================================
INFERENCE_TEMPERATURE = 0.8
DEFAULT_STEPS = 50
DEFAULT_SEED = 42

# ============================================================
# ORNSTEIN-UHLENBECK REFERENCE CONFIGURATION
# ============================================================
USE_OU_BRIDGE = False
OU_THETA = 1.0
OU_SIGMA = 1.4142135623730951  # sqrt(2)

# ============================================================
# FEATURE TOGGLES
# ============================================================
USE_PERCENTILE = True
USE_SNAPSHOTS = True
USE_KPI_TRACKING = True
USE_AMP = False  # Will be enabled for CUDA only

# ============================================================
# SNAPSHOT CONFIGURATION
# ============================================================
SNAPSHOT_INTERVAL = 20
SNAPSHOT_KEEP = 5

# ============================================================
# KPI TRACKING
# ============================================================
KPI_WINDOW_SIZE = 100
EARLY_STOP_PATIENCE = 15
TARGET_SNR = 30.0
REVERT_THRESHOLD = 2.5

# ============================================================
# DIRECTORY STRUCTURE
# ============================================================
BASE_DIR = Path("enhanced_label_sb")
DIRS = {
    "ckpt": BASE_DIR / "checkpoints",
    "logs": BASE_DIR / "logs",
    "samples": BASE_DIR / "samples",
    "snaps": BASE_DIR / "snapshots",
    "onnx": BASE_DIR / "onnx",
    "metrics": BASE_DIR / "metrics",
    "inference": BASE_DIR / "inference",
    "best": BASE_DIR / "best_models"
}

# Create directories
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE CONFIGURATION (will be set by main)
# ============================================================
DEVICE = torch.device('cpu')
DTYPE = torch.float32

# ============================================================
# LOGGER SETUP
# ============================================================
def setup_logger(name="EnhancedTrainer"):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        log_path = DIRS["logs"] / f"train_{int(datetime.now().timestamp())}.log"
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

logger = setup_logger()