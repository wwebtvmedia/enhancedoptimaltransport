# ============================================================================
# CONFIGURATION CONSTANTS – ALL SETTINGS IN ONE PLACE
# ============================================================================

import os
import time
import numpy as np
import torch
from pathlib import Path
import logging

# ============================================================================
# PATHS
# ============================================================================
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

# ============================================================================
# MODEL DIMENSIONS
# ============================================================================
IMG_SIZE = 64
LATENT_CHANNELS = 4
LATENT_H = IMG_SIZE // 8
LATENT_W = IMG_SIZE // 8
LATENT_DIM = LATENT_CHANNELS * LATENT_H * LATENT_W

# ============================================================================
# LABEL CONDITIONING
# ============================================================================
NUM_CLASSES = 6000
LABEL_EMB_DIM = 128

# ============================================================================
# TRAINING HYPERPARAMETERS (base values, may be adjusted per device)
# ============================================================================
LR = 2e-4
EPOCHS = 200
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ============================================================================
# LOSS WEIGHTS
# ============================================================================
KL_WEIGHT = 0.00001
RECON_WEIGHT = 1.0
DRIFT_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.1
CONSISTENCY_WEIGHT = 0.5
REVERT_THRESHOLD = 2.5

# ============================================================================
# VAE SPECIFIC
# ============================================================================
LATENT_SCALE = 0.3
FREE_BITS = 1.0
DIVERSITY_TARGET_STD = 0.1
DIVERSITY_BALANCE_WEIGHT = 0.2
KL_ANNEALING_EPOCHS = 30
LOGVAR_CLAMP_MIN = -4
LOGVAR_CLAMP_MAX = 4
MU_NOISE_SCALE = 0.01

# ============================================================================
# CHANNEL DROPOUT SETTINGS
# ============================================================================
CHANNEL_DROPOUT_PROB = 0.2          # Probability of applying dropout
CHANNEL_DROPOUT_SURVIVAL = 0.8      # Probability of channel surviving when dropout applied
# ============================================================================
# DRIFT NETWORK SPECIFIC
# ============================================================================
# Optimizer multipliers
DRIFT_LR_MULTIPLIER = 2.0                # Drift LR = LR * DRIFT_LR_MULTIPLIER
DRIFT_GRAD_CLIP_FACTOR = 0.5             # Drift grad clip = GRAD_CLIP * DRIFT_GRAD_CLIP_FACTOR

# Phase transition learning rate factors
PHASE2_VAE_LR_FACTOR = 0.1                # Phase 2 VAE LR = LR * PHASE2_VAE_LR_FACTOR
PHASE3_VAE_LR_FACTOR = 0.05               # Phase 3 VAE LR = LR * PHASE3_VAE_LR_FACTOR

# Temperature annealing (Phase 2/3)
TEMPERATURE_START = 1.0                    # Initial temperature (at start of Phase 2)
TEMPERATURE_END = 0.3                      # Final temperature (at end of training)

# Target noise for drift training
DRIFT_TARGET_NOISE_SCALE = 0.01

# Time weighting factor for drift loss
TIME_WEIGHT_FACTOR = 2.0                   # time_weights = 1 + TIME_WEIGHT_FACTOR * t

# ============================================================================
# INFERENCE
# ============================================================================
DEFAULT_STEPS = 50
DEFAULT_SEED = 42
INFERENCE_TEMPERATURE = 0.8
LANGEVIN_STEP_SIZE = 0.1
LANGEVIN_SCORE_SCALE = 1.0

# ============================================================================
# FOURIER FEATURES
# ============================================================================
USE_FOURIER_FEATURES = False
FOURIER_FREQS = [1, 2, 4, 8, 16]

# ============================================================================
# ENHANCED FEATURES
# ============================================================================
USE_PERCENTILE = True
USE_SNAPSHOTS = True
USE_KPI_TRACKING = True
TARGET_SNR = 30.0
SNAPSHOT_INTERVAL = 20
SNAPSHOT_KEEP = 5
KPI_WINDOW_SIZE = 100
EARLY_STOP_PATIENCE = 15

# ============================================================================
# OU BRIDGE AND AMP
# ============================================================================
USE_OU_BRIDGE = False
OU_THETA = 1.0
OU_SIGMA = np.sqrt(2)
USE_AMP = False

# ============================================================================
# THREE-PHASE TRAINING SCHEDULE
# ============================================================================
# Adaptive switch points based on total epochs
PHASE1_EPOCHS = max(50, int(EPOCHS / 6))    # End of Phase 1 (VAE only)
PHASE2_EPOCHS = max(50, int(EPOCHS / 2))    # End of Phase 2 (Drift only)
# Phase 3 runs from PHASE2_EPOCHS to EPOCHS (Both train)

# ============================================================================
# TRAINING SCHEDULE DICTIONARY
# ============================================================================
TRAINING_SCHEDULE = {
    'mode': 'auto',                          # 'auto', 'manual', 'custom', 'alternate', 'three_phase'
    'force_phase': None,                      # 1 (VAE), 2 (Drift), 3 (Both) for manual mode
    'custom_schedule': {},                    # {epoch: phase}
    'switch_epoch': 50,                       # For auto mode (single switch)
    'switch_epoch_1': PHASE1_EPOCHS,           # For three_phase mode
    'switch_epoch_2': PHASE2_EPOCHS,           # For three_phase mode
    'vae_epochs': list(range(0, 50)),
    'drift_epochs': list(range(50, 200)),
    'alternate_freq': 5,
}

# ============================================================================
# DEVICE CONFIGURATION (will be set at runtime)
# ============================================================================
DEVICE = None
DTYPE = torch.float32
BATCH_SIZE = 64  # Will be adjusted per device
AMP_AVAILABLE = False

# ============================================================================
# LOGGER SETUP
# ============================================================================
def setup_logger():
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("EnhancedTrainer")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create log directory if it doesn't exist
    DIRS["logs"].mkdir(parents=True, exist_ok=True)
    log_path = DIRS["logs"] / f"train_{int(time.time())}.log"
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logger()