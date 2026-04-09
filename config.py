# ============================================================================
# CONFIGURATION CONSTANTS - ALL SETTINGS IN ONE PLACE
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
# DATASET CONFIGURATION
# ============================================================================
DATASET_NAME = "STL10"        # Options: "STL10", "CIFAR10", "CUSTOM", "MULTI"
DATASET_PATH = Path("./data") # Path to download or find custom data
USE_MULTI_DATASET = True      # Enable training on multiple datasets
DATASETS = ["STL10", "CIFAR10"] # Datasets to use when MULTI is enabled

# ============================================================================
# LABEL AND CONTEXT CONDITIONING
# ============================================================================
NUM_CLASSES = 10              # Max classes across datasets (shared mapping)
LABEL_EMB_DIM = 128
USE_CONTEXT = True            # Enable additional contextual input
CONTEXT_DIM = 64              # Dimension of continuous context vector
NUM_SOURCES = 2               # Number of unique data sources for context embedding
IMG_SIZE = 96                # Internal processing size (standardized)
GEN_SIZE = 96                # Final output generation size (can be different)
LATENT_CHANNELS = 8          # Increased from 4 for better reconstruction quality
LATENT_H = IMG_SIZE // 16    # 6x6 for 96x96 images (4 downsampling stages)
LATENT_W = IMG_SIZE // 16
LATENT_DIM = LATENT_CHANNELS * LATENT_H * LATENT_W
COMPRESSION_RATIO = (IMG_SIZE * IMG_SIZE * 3) / LATENT_DIM

# ============================================================================
# LABEL CONDITIONING
# ============================================================================
NUM_CLASSES = 10
LABEL_EMB_DIM = 128

# ============================================================================
# TRAINING HYPERPARAMETERS (base values, may be adjusted per device)
# ============================================================================
LR = 2e-4
EPOCHS = 600
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ============================================================================
# LOSS WEIGHTS (ENHANCED FOR QUALITY)
# ============================================================================
KL_WEIGHT = 0.003              # Increased from 0.001 for better latent structure
RECON_WEIGHT = 7.0             # Increased from 2.5 for sharper reconstruction
SUBPIXEL_INITIAL_MIX = -1.0    # Initial logit for hybrid upsampling (sigmoid(-1.0) ≈ 0.27 subpixel)
DRIFT_WEIGHT = 1.0             # Increased from 0.5 for better trajectory learning
DIVERSITY_WEIGHT = 1.5         # Increased from 1.0 to prevent channel collapse
CONSISTENCY_WEIGHT = 1.0       # Increased from 0.5 for better anchor stability
PHASE3_RECON_SCALE = 0.5       # Keep decoder sharp in Phase 3

# NEW: Quality-focused loss weights
PERCEPTUAL_WEIGHT = 0.5        # VGG feature matching
LPIPS_WEIGHT = 0.3             # Learned perceptual similarity (if available)
EDGE_WEIGHT = 0.2              # Edge preservation loss
TV_WEIGHT = 0.05               # Total Variation (TV) weight for smoothness (small by default)

# ============================================================================
# INFERENCE (ENHANCED FOR QUALITY)
# ============================================================================
DEFAULT_STEPS = 100            # Increased from 50 for smoother trajectories
DEFAULT_SEED = 42
INFERENCE_TEMPERATURE = 0.6    # Lower from 0.8 for sharper samples
DEFAULT_LANGEVIN_STEPS = 0     # Disabled by default (was causing instability)
LANGEVIN_STEP_SIZE = 0.02      # Smaller from 0.05 for stability
LANGEVIN_SCORE_SCALE = 1.2     # Reset from 1.5 to safer value
LANGEVIN_DECAY = 0.95          # Slower decay for better convergence

# ============================================================================
# VAE SPECIFIC (ENHANCED)
# ============================================================================
LATENT_SCALE = 1.0
FREE_BITS = 1.0
DIVERSITY_TARGET_STD = 0.8
DIVERSITY_MAX_STD = 2.0
DIVERSITY_LOW_PENALTY = 2.0
DIVERSITY_HIGH_PENALTY = 0.5
DIVERSITY_BALANCE_WEIGHT = 0.4
DIVERSITY_ADAPTIVE = True
DIVERSITY_TARGET_START = 0.3
DIVERSITY_TARGET_END = 1.0
DIVERSITY_ADAPT_EPOCHS = 50
KL_ANNEALING_EPOCHS = 30
LOGVAR_CLAMP_MIN = -4
LOGVAR_CLAMP_MAX = 4
MU_NOISE_SCALE = 0.01
CST_COEF_GAUSSIAN_PRIO = 0.8        # Standard deviation for the Gaussian prior in latent space

# NEW: Enhanced decoder settings
DECODER_ATTENTION_LAYERS = [16, 32]  # Resolutions to add self-attention
DECODER_UPSAMPLE_STAGES = 4          # Increased from 3

# ============================================================================
# CHANNEL DROPOUT SETTINGS
# ============================================================================
CHANNEL_DROPOUT_PROB = 0.2          # Probability of applying dropout
CHANNEL_DROPOUT_SURVIVAL = 0.8      # Probability of channel surviving when dropout applied

# ============================================================================
# CLASSIFIER-FREE GUIDANCE (CFG)
# ============================================================================
LABEL_DROPOUT_PROB = 0.1            # Probability of dropping label during training
CFG_SCALE = 1.0                     # Scale for classifier-free guidance (1.0 = disabled)

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
CHECKPOINT_INTERVAL = 5
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
SCHEDULE_MANUALLY_SET = False
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
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = None
DTYPE = torch.float32
BATCH_SIZE = 32
AMP_AVAILABLE = False

def initialize_hardware():
    """Determines best available hardware and updates global config."""
    global DEVICE, AMP_AVAILABLE, BATCH_SIZE
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        AMP_AVAILABLE = True
        info = f"🎮 CUDA: {torch.cuda.get_device_name(0)}"
        BATCH_SIZE = 64
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = torch.device("xpu")
        AMP_AVAILABLE = True
        info = f"🔵 Intel Arc: {torch.xpu.get_device_name(0)}"
        BATCH_SIZE = 64
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        AMP_AVAILABLE = False
        info = "🍎 Apple Silicon (MPS)"
        BATCH_SIZE = 32
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                DEVICE = torch_directml.device()
                AMP_AVAILABLE = False
                info = "🎮 DirectML (AMD/Intel)"
                BATCH_SIZE = 32
            else:
                raise ImportError
        except ImportError:
            DEVICE = torch.device("cpu")
            AMP_AVAILABLE = False
            info = "💻 CPU (Slow)"
            BATCH_SIZE = 16
            
    return info

# ============================================================================
# LOGGER SETUP
# ============================================================================
def setup_logger():
    """Setup logger with file and console handlers."""
    logger = logging.getLogger("EnhancedTrainer")
    logger.setLevel(logging.INFO)
    
    # Check if we already have handlers to avoid duplicate logging or clearing active ones
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    if has_file_handler and has_stream_handler:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    
    if not has_stream_handler:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    if not has_file_handler:
        # Create log directory if it doesn't exist
        DIRS["logs"].mkdir(parents=True, exist_ok=True)
        log_path = DIRS["logs"] / f"train_{int(time.time())}.log"
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

logger = setup_logger()