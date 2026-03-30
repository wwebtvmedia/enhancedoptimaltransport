# ============================================================================
# ENHANCED TRAINER FOR SCHRÖDINGER BRIDGE
# ============================================================================

import os
import math
import time
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import config
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
import scipy.linalg  # Added for matrix operations if needed
from typing import Optional, List, Dict, Union, Tuple, Any
from torchvision.transforms.functional import rgb_to_grayscale

# Optional imports
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Import modules
import config
import data_management as dm
import models

warnings.filterwarnings('ignore')

# ============================================================
# ORNSTEIN-UHLENBECK REFERENCE PROCESS
# ============================================================
class OUReference:
    """Ornstein-Uhlenbeck reference process for Schrödinger Bridge."""
    
    def __init__(self, theta=1.0, sigma=np.sqrt(2)):
        self.theta = theta
        self.sigma = sigma
        
    def stationary_variance(self):
        """Variance of the stationary distribution N(0, sigma^2/(2*theta))."""
        return self.sigma**2 / (2 * self.theta)
    
    def transition_kernel(self, z0, t, dt):
        """Mean and variance of transition from z0 at time t to t+dt."""
        exp_neg_theta_dt = torch.exp(-self.theta * dt)
        mean = z0 * exp_neg_theta_dt
        var = (self.sigma**2 / (2 * self.theta)) * (1 - exp_neg_theta_dt**2)
        return mean, var
    
    def bridge_sample(self, z0, z1, t):
        """
        Sample from the exact OU bridge between z0 at t=0 and z1 at t=1.
        Returns mean and variance of the bridge at time t (0 <= t <= 1).
        
        Using exponential forms for numerical stability.
        """
        exp_neg_theta_t = torch.exp(-self.theta * t)
        exp_neg_theta_1_t = torch.exp(-self.theta * (1 - t))
        exp_neg_theta = torch.exp(-self.theta)
        
        # Mean
        numerator = (exp_neg_theta_t * (1 - exp_neg_theta_1_t**2) * z0 + 
                     (1 - exp_neg_theta_t**2) * exp_neg_theta_1_t * z1)
        denominator = 1 - exp_neg_theta**2
        mean = numerator / denominator
        
        # Variance
        var = (self.sigma**2 / (2 * self.theta)) * \
              ((1 - exp_neg_theta_t**2) * (1 - exp_neg_theta_1_t**2)) / (1 - exp_neg_theta**2)
        var = var.clamp(min=0)
        
        return mean, var

    def bridge_velocity(self, z0, z1, t):
        """
        Compute the velocity (time derivative of the mean) of the OU bridge.
        d/dt E[zt | z0, z1]
        """
        exp_neg_theta_t = torch.exp(-self.theta * t)
        exp_neg_theta_1_t = torch.exp(-self.theta * (1 - t))
        exp_neg_theta = torch.exp(-self.theta)
        
        # d/dt sinh(theta(1-t)) = -theta cosh(theta(1-t))
        # d/dt sinh(theta t) = theta cosh(theta t)
        
        # Mean formula: mu(t) = (sinh(theta(1-t)) z0 + sinh(theta t) z1) / sinh(theta)
        # We'll use the exponential form for consistency with bridge_sample
        # sinh(x) = (exp(x) - exp(-x)) / 2
        
        # Let's derive it directly from the mean formula used in bridge_sample:
        # mean = (exp(-theta*t) * (1 - exp(-theta*(1-t))^2) * z0 + (1 - exp(-theta*t)^2) * exp(-theta*(1-t)) * z1) / (1 - exp(-theta)^2)
        
        dt = 1e-4
        mean_plus, _ = self.bridge_sample(z0, z1, t + dt)
        mean_minus, _ = self.bridge_sample(z0, z1, (t - dt).clamp(min=0))
        return (mean_plus - mean_minus) / (2 * dt)

# ============================================================
# UTILITIES
# ============================================================
def calc_snr(real: torch.Tensor, recon: torch.Tensor) -> float:
    """Calculate Signal-to-Noise Ratio."""
    mse = F.mse_loss(recon, real)
    if mse == 0: 
        return 100.0
    return 10 * torch.log10(1.0 / (mse + 1e-8)).item()
    
def kl_divergence_spatial(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.sum(kl, dim=[1, 2, 3])
    kl = torch.max(kl, torch.full_like(kl, config.FREE_BITS))
    return torch.mean(kl)

def composite_score(loss_dict: Dict, phase: int) -> float:
    """Compute a composite score for model selection."""
    score = 0.0
    if phase == 1:
        # VAE phase: higher SNR, lower KL, higher diversity are good
        if 'snr' in loss_dict:
            score += loss_dict['snr'] / config.TARGET_SNR
        if 'kl' in loss_dict:
            score -= loss_dict['kl'] * 10
        if 'diversity' in loss_dict:
            score += loss_dict['diversity'] * 100
        if 'min_channel_std' in loss_dict:
            score += loss_dict['min_channel_std'] * 20
    else:
        # Drift phase: lower drift error is better
        if 'drift' in loss_dict:
            score -= loss_dict['drift'] * 10
        if 'consistency' in loss_dict:
            score -= loss_dict['consistency'] * 10
    return score

def set_training_phase(epoch: int) -> int:
    """Global function to set training phase for each epoch."""
    mode = config.TRAINING_SCHEDULE['mode']
    
    if mode == 'manual':
        phase = config.TRAINING_SCHEDULE['force_phase']
        return 1 if phase is None else phase
    
    elif mode == 'custom':
        return config.TRAINING_SCHEDULE['custom_schedule'].get(epoch, 1)
    
    elif mode == 'alternate':
        alt_freq = config.TRAINING_SCHEDULE.get('alternate_freq', 5)
        return 1 if (epoch // alt_freq) % 2 == 0 else 2
    
    elif mode == 'three_phase':
        e1 = config.TRAINING_SCHEDULE['switch_epoch_1']
        e2 = config.TRAINING_SCHEDULE['switch_epoch_2']
        if epoch < e1:
            return 1
        elif epoch < e2:
            return 2
        else:
            return 3
    
    else:  # 'auto' mode (single switch)
        return 1 if epoch < config.TRAINING_SCHEDULE['switch_epoch'] else 2

def configure_training_schedule(
    mode: str = 'auto',
    vae_epochs: Optional[List[int]] = None,
    drift_epochs: Optional[List[int]] = None,
    switch_epoch: int = 50,
    alternate_freq: int = 5,
    custom_schedule: Optional[Dict] = None,
    switch_epoch_1: Optional[int] = None,
    switch_epoch_2: Optional[int] = None,
) -> Dict:
    """Configure the global training schedule."""
    if mode == 'vae_only':
        config.TRAINING_SCHEDULE['mode'] = 'manual'
        config.TRAINING_SCHEDULE['force_phase'] = 1
    elif mode == 'drift_only':
        config.TRAINING_SCHEDULE['mode'] = 'manual'
        config.TRAINING_SCHEDULE['force_phase'] = 2
    elif mode == 'auto':
        config.TRAINING_SCHEDULE['mode'] = 'auto'
        config.TRAINING_SCHEDULE['switch_epoch'] = switch_epoch
    elif mode == 'alternate':
        config.TRAINING_SCHEDULE['mode'] = 'alternate'
        config.TRAINING_SCHEDULE['alternate_freq'] = alternate_freq
    elif mode == 'custom':
        config.TRAINING_SCHEDULE['mode'] = 'custom'
        config.TRAINING_SCHEDULE['custom_schedule'] = custom_schedule or {}
    elif mode == 'three_phase':
        config.TRAINING_SCHEDULE['mode'] = 'three_phase'
        config.TRAINING_SCHEDULE['switch_epoch_1'] = switch_epoch_1 or config.PHASE1_EPOCHS
        config.TRAINING_SCHEDULE['switch_epoch_2'] = switch_epoch_2 or config.PHASE2_EPOCHS
    elif mode == 'manual':
        config.TRAINING_SCHEDULE['mode'] = 'manual'
    
    if vae_epochs is not None:
        config.TRAINING_SCHEDULE['vae_epochs'] = vae_epochs
    if drift_epochs is not None:
        config.TRAINING_SCHEDULE['drift_epochs'] = drift_epochs
    
    config.logger.info(f"Training schedule configured: mode={config.TRAINING_SCHEDULE['mode']}")
    return config.TRAINING_SCHEDULE

# ============================================================
# KPI TRACKER
# ============================================================
class KPITracker:
    """Track key performance indicators during training."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.composite_scores = []
        
    def update(self, metrics_dict: Dict) -> None:
        for key, value in metrics_dict.items():
            if value is not None:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key].pop(0)
        
        # Track composite score if available
        if 'composite_score' in metrics_dict:
            self.composite_scores.append(metrics_dict['composite_score'])
            if len(self.composite_scores) > self.window_size:
                self.composite_scores.pop(0)
                
    def compute_convergence(self) -> Dict:
        """Compute convergence metrics."""
        convergence = {}
        for metric_name, values in self.metrics.items():
            if len(values) >= 10:
                window = min(20, len(values))
                ma = np.mean(values[-window:])
                convergence[f'{metric_name}_ma'] = ma
                std = np.std(values[-window:]) if len(values) >= window else 0
                convergence[f'{metric_name}_std'] = std
                if len(values) >= 5:
                    x = np.arange(len(values[-10:]))
                    y = values[-10:]
                    slope, _, _, _, _ = stats.linregress(x, y)
                    convergence[f'{metric_name}_trend'] = slope
        
        if 'loss' in self.metrics and len(self.metrics['loss']) >= 10:
            loss_values = self.metrics['loss'][-20:]
            loss_std = np.std(loss_values)
            convergence['convergence_score'] = 1.0 / (1.0 + loss_std)
        return convergence
        
    def should_stop(self, patience: int = config.EARLY_STOP_PATIENCE, min_delta: float = 1e-4, phase: int = 1) -> bool:
        """Determine if training should stop early."""
        if phase == 1:
            return False
        if 'loss' not in self.metrics or len(self.metrics['loss']) < patience * 2:
            return False
        if len(self.metrics['loss']) < 30:
            return False
        
        recent_losses = self.metrics['loss'][-patience:]
        is_increasing = all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1))
        
        if is_increasing:
            best_loss_in_window = min(recent_losses)
            current_loss = recent_losses[-1]
            if current_loss - best_loss_in_window > min_delta:
                config.logger.info(f"Early stopping triggered: loss increased for {patience} epochs")
                return True
        return False

# ============================================================
# ENHANCED TRAINER
# ============================================================
class EnhancedLabelTrainer:
    """Main trainer class for multimodal Schrödinger Bridge."""
    
    def __init__(self, loader):
        self.loader = loader

        self.vae = models.MultimodalVAE().to(config.DEVICE)
        self.drift = models.MultimodalDrift().to(config.DEVICE)
        
        # Initialize text encoder based on configuration
        if config.USE_CLIP_STYLE:
            self.text_encoder = models.CLIPTextEncoder(
                vocab_size=config.CLIP_VOCAB_SIZE,
                embed_dim=config.TEXT_EMBEDDING_DIM,
                max_length=config.MAX_TEXT_LENGTH,
                num_layers=config.CLIP_TEXT_ENCODER_LAYERS,
                num_heads=config.CLIP_TEXT_ENCODER_HEADS,
                use_clip_style=True
            ).to(config.DEVICE)
        else:
            self.text_encoder = models.TextEncoder(vocab_size=config.NUM_CLASSES).to(config.DEVICE)
        
        # Initialize image projection head if enabled
        if config.USE_IMAGE_PROJECTION:
            self.image_projection = models.ImageProjection(
                latent_dim=config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W,
                embed_dim=config.IMAGE_PROJECTION_DIM
            ).to(config.DEVICE)
        else:
            self.image_projection = None
        
        # Initialize contrastive loss if enabled
        if config.USE_CONTRASTIVE_LOSS:
            self.contrastive_loss = models.CLIPContrastiveLoss(
                temperature=config.CONTRASTIVE_TEMPERATURE,
                learnable_temperature=config.LEARNABLE_TEMPERATURE
            ).to(config.DEVICE)
        else:
            self.contrastive_loss = None
        
        # Optimizers
        self.opt_vae = optim.AdamW(self.vae.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        self.opt_text = optim.AdamW(self.text_encoder.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        
        # Add image projection optimizer if enabled
        if self.image_projection is not None:
            self.opt_image_proj = optim.AdamW(self.image_projection.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        else:
            self.opt_image_proj = None

        # Drift optimizer with multiplier from config
        self.opt_drift = optim.AdamW(
            self.drift.parameters(),
            lr=config.LR * config.DRIFT_LR_MULTIPLIER,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_vae, T_max=config.EPOCHS, eta_min=config.LR*0.01
        )
        self.scheduler_drift = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_drift, T_max=config.EPOCHS, eta_min=config.LR*0.01
        )
        
        self.kpi_tracker = KPITracker(window_size=config.KPI_WINDOW_SIZE)
          
        if config.USE_SNAPSHOTS:
            self.snapshot_vae = dm.SnapshotManager(self.vae, self.opt_vae, name="vae", 
                                                interval=config.SNAPSHOT_INTERVAL, 
                                                keep=config.SNAPSHOT_KEEP)
            self.snapshot_drift = dm.SnapshotManager(self.drift, self.opt_drift, name="drift",
                                                interval=config.SNAPSHOT_INTERVAL,
                                                keep=config.SNAPSHOT_KEEP)
        else:
            self.snapshot_vae = None
            self.snapshot_drift = None

        self.epoch = 0
        self.step = 0
        self.phase = 1
        self.best_loss = float('inf')
        self.best_composite_score = float('-inf')
        self.debug_counter = 0
        self.debug_interval = 50
        
        # Reference for Phase 2
        self.vae_ref = None
        
        # OU reference process
        self.ou_ref = OUReference(theta=config.OU_THETA, sigma=config.OU_SIGMA) if config.USE_OU_BRIDGE else None
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and config.DEVICE.type == 'cuda' else None
        
        config.logger.info(f"Models initialized:")
        config.logger.info(f"  VAE params: {sum(p.numel() for p in self.vae.parameters()):,}")
        config.logger.info(f"  Drift params: {sum(p.numel() for p in self.drift.parameters()):,}")
        if config.USE_OU_BRIDGE:
            config.logger.info(f"  Using OU bridge reference (theta={config.OU_THETA})")

        # ImageNet Mean and Std (for 0-1 normalized images)
        # ImageNet Mean and Std (for 0-1 normalized images) - Moved to device once
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(config.DEVICE)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(config.DEVICE)

        # Pre-compute SSIM Gaussian Window
        window_size = 11
        sigma = 1.5
        gauss = torch.arange(window_size, dtype=torch.float32, device=config.DEVICE)
        gauss = torch.exp(-(gauss - window_size//2)**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        window = window[None, None, :, :]
        # Cache it for the 3 channels
        self.ssim_window = window.expand(3, -1, -1, -1).contiguous()


    def diagnose_latent_collapse(self, mu, logvar, epoch):
        """Diagnose and fix latent space collapse issues."""
        with torch.no_grad():
            # Calculate statistics
            latent_std = torch.exp(0.5 * logvar).mean().item()
            mu_std = mu.std().item()
            channel_means = mu.mean(dim=[0, 2, 3])
            channel_stds = mu.std(dim=[0, 2, 3])
            
            # Check for collapse
            if epoch > 10 and latent_std < 0.3:
                config.logger.warning(f"⚠️ Latent collapse detected! std={latent_std:.3f}")
                
                # Option 1: Increase KL weight temporarily
                old_kl = config.KL_WEIGHT
                config.KL_WEIGHT = min(0.05, config.KL_WEIGHT * 2)  # Double but cap at 0.05
                config.logger.info(f"  Temporarily increasing KL weight: {old_kl} -> {config.KL_WEIGHT}")
                
                # Option 2: Add noise to encourage exploration
                if hasattr(self, 'vae') and hasattr(self.vae, 'z_logvar'):
                    # Add small noise to mu during training
                    self.vae.mu_noise_scale = 0.05  # Will need to add this attribute
                
            return latent_std, mu_std, channel_stds

    def perceptual_loss(self, recon, target):
        """Simple perceptual loss using pretrained VGG features with correct ImageNet normalization"""
        if not hasattr(self, 'vgg'):
            try:
                import torchvision.models as tv_models
                self.vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(config.DEVICE)
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.vgg.eval()
            except Exception as e:
                # Cache failure to avoid repeated attempts
                self.vgg = None
                return torch.tensor(0.0, device=config.DEVICE)
        
        if self.vgg is None:
            return torch.tensor(0.0, device=config.DEVICE)
    
        # Using pre-allocated device tensors
        mean = self.vgg_mean
        std = self.vgg_std

        # 1. Map from [-1, 1] to [0, 1]
        recon_01 = (recon + 1) / 2
        target_01 = (target + 1) / 2
        
        # 2. Apply ImageNet normalization - added .contiguous() for MPS stability
        recon_norm = ((recon_01 - mean) / std).contiguous()
        target_norm = ((target_01 - mean) / std).contiguous()
        
        # 3. Get features
        recon_feat = self.vgg(recon_norm)
        target_feat = self.vgg(target_norm)
        
        return F.mse_loss(recon_feat, target_feat)


    def _switch_to_phase(self, new_phase: int):
        """Handle transition between training phases."""
        if new_phase == 1:
            # Phase 1: VAE only
            self.vae.train()
            self.drift.eval()
            
            # Ensure all VAE params are trainable
            for param in self.vae.parameters():
                param.requires_grad = True
                
            # Update existing optimizer's LR dynamically
            new_lr = config.LR
            for param_group in self.opt_vae.param_groups:
                param_group['lr'] = new_lr
                
            self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_vae, T_max=config.EPOCHS, eta_min=config.LR * 0.01
            )
            self.vae_ref = None  # no anchor needed

        elif new_phase == 2:
            # Phase 2: Drift only, freeze decoder, unfreeze encoder, create anchor
            self.vae_ref = models.MultimodalVAE().to(config.DEVICE)
            self.vae_ref.load_state_dict(self.vae.state_dict())
            self.vae_ref.eval()
            for param in self.vae_ref.parameters():
                param.requires_grad = False

            # Unfreeze encoder parts only, explicitly zero-out grad for frozen params
            unfrozen_count = 0
            for name, param in self.vae.named_parameters():
                if any(k in name for k in ['enc_', 'context_encoder', 'z_mean', 'z_logvar']):
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                else:
                    param.requires_grad = False
                    param.grad = None  # Ensures optimizer skips this parameter
            config.logger.info(f"Phase 2: Unfrozen {unfrozen_count:,} encoder params. Anchor set.")

            # Update existing optimizer's LR dynamically
            new_lr = config.LR * config.PHASE2_VAE_LR_FACTOR
            for param_group in self.opt_vae.param_groups:
                param_group['lr'] = new_lr
                
            self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_vae, T_max=config.EPOCHS - self.epoch, eta_min=config.LR * 0.005
            )
            # Clear optimizer state to flush momentum from phase 1 for frozen parameters
            self.opt_vae.state.clear()
            # Store the epoch when Phase 2 started
            self.phase2_start_epoch = self.epoch

        elif new_phase == 3:
            # Phase 3: Both trainable – unfreeze decoder as well
            for name, param in self.vae.named_parameters():
                param.requires_grad = True
            
            # Ensure reference anchor exists if we skipped Phase 2
            if not hasattr(self, 'vae_ref') or self.vae_ref is None:
                self.vae_ref = models.MultimodalVAE().to(config.DEVICE)
                self.vae_ref.load_state_dict(self.vae.state_dict())
                self.vae_ref.eval()
                for param in self.vae_ref.parameters():
                    param.requires_grad = False
                config.logger.info("Phase 3: Reference anchor created (transitioned from Phase 1).")

            config.logger.info("Phase 3: Unfroze all VAE parameters (encoder + decoder).")

            # Update existing optimizer's LR dynamically
            new_lr = config.LR * config.PHASE3_VAE_LR_FACTOR
            for param_group in self.opt_vae.param_groups:
                param_group['lr'] = new_lr
                
            self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
                self.opt_vae, T_max=config.EPOCHS - self.epoch, eta_min=config.LR * 0.001
            )
            # Drift optimizer unchanged

        # Ensure correct train/eval modes
        if new_phase == 1:
            self.vae.train()
            self.drift.eval()
        elif new_phase == 2:
            self.vae.train()   # Let's put VAE in train mode so that BatchNorm etc update.
            self.drift.train()
        elif new_phase == 3:
            self.vae.train()
            self.drift.train()
   
    def get_training_phase(self, epoch):
        phase = set_training_phase(epoch)
        if phase != self.phase:
            config.logger.info(f" Phase changed from {self.phase} to {phase} at epoch {epoch+1}")
            self._switch_to_phase(phase)
            self.phase = phase
        return phase

    def ssim_loss(self, x, y):
        """
        Compute SSIM loss between two image batches.
        Input: x, y in range [-1, 1] (as produced by tanh).
        Returns: 1 - mean SSIM over batch.
        """
        # Map from [-1,1] to [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        
        # Constants from the SSIM paper
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Gaussian window parameters
        window_size = 11
        
        # Use pre-computed window
        window = self.ssim_window
        
        # Pad to keep spatial size
        pad = window_size // 2
        
        # Compute means, variances, covariances
        mu_x = F.conv2d(x, window, padding=pad, groups=x.size(1))
        mu_y = F.conv2d(y, window, padding=pad, groups=x.size(1))
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=x.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=x.size(1)) - mu_xy
        
        # Numerical stability
        sigma_x_sq = torch.clamp(sigma_x_sq, min=0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0)
        
        # SSIM map per channel
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        # Average over spatial dimensions and channels -> per-image SSIM
        ssim_per_image = ssim_map.mean(dim=[1, 2, 3])
        
        # Return 1 - mean SSIM over batch (loss)
        return 1 - ssim_per_image.mean()

    def compute_loss(self, batch: Dict, phase: int = 1, batch_idx: int = 0) -> Dict:
        """Compute loss for current batch based on training phase."""
        # Extract images and context
        images = batch['image'].to(config.DEVICE)
        labels = batch.get('label', None)
        if labels is not None:
            labels = labels.to(config.DEVICE)
        
        # Multimodal Text Encoding
        text_emb = None
        if config.USE_MULTIMODAL and 'text_tokens' in batch:
            tokens = batch['text_tokens'].to(config.DEVICE)
            text_emb = self.text_encoder(tokens)
        
        # Phase 1: Train VAE + Text Encoder + Context Decoder
        if phase == 1:
            # Full forward pass (Replaces the redundant separate encode call)
            recon, mu, logvar = self.vae(images, labels, text_emb)
            
            # CLIP-style contrastive loss calculation (using 'mu' from VAE pass)
            contrastive_loss_val = torch.tensor(0.0, device=config.DEVICE)
            if (config.USE_CONTRASTIVE_LOSS and self.contrastive_loss is not None and
                text_emb is not None):
                
                # Project image latent to shared embedding space
                # We detach 'mu' to keep VAE encoding gradient-isolated from contrastive loss
                # (matching previous behavior) unless we want them to learn together.
                mu_for_contrastive = mu.detach()
                
                if self.image_projection is not None:
                    image_emb = self.image_projection(mu_for_contrastive)
                else:
                    # Use flattened mu as image embedding
                    image_emb = mu_for_contrastive.flatten(1)
                    # Project to same dimension as text embedding if needed
                    if image_emb.size(1) != text_emb.size(1):
                        if not hasattr(self, 'temp_projection'):
                            self.temp_projection = nn.Linear(image_emb.size(1), text_emb.size(1)).to(config.DEVICE)
                        image_emb = self.temp_projection(image_emb)
                
                # Compute contrastive loss
                contrastive_loss_val = self.contrastive_loss(image_emb, text_emb)

            # 1. Classification & Text Branch
            pred_text_emb, pred_logits = self.vae.context_decoder(mu)
            cls_loss = F.cross_entropy(pred_logits, labels) if labels is not None else torch.tensor(0.0, device=config.DEVICE)
            
            # 2. Text Alignment (Mapping text vectors to the same latent space)
            latent_alignment_loss = torch.tensor(0.0, device=config.DEVICE)
            if text_emb is not None:
                z_txt = self.vae.encode_text(text_emb)
                # MSE Alignment between Image Latent (mu) and Text Latent (z_txt)
                latent_alignment_loss = F.mse_loss(mu, z_txt.detach()) # Align img to txt
                # Also decode from text latent to ensure reconstructability
                recon_from_txt = self.vae.decode(z_txt, labels, text_emb)
                recon_loss_txt = F.l1_loss(recon_from_txt, images) * 0.1 # Small consistency bonus
                latent_alignment_loss = latent_alignment_loss + recon_loss_txt

            # Periodically diagnose and correct latent collapse
            if batch_idx % self.debug_interval == 0:
                self.diagnose_latent_collapse(mu, logvar, self.epoch)
            
            latent_std = torch.exp(0.5 * logvar).mean().item()
            channel_stds = mu.std(dim=[0, 2, 3]).detach().cpu().numpy()
            min_channel_std = channel_stds.min()
            
            raw_l1 = F.l1_loss(recon, images)
            raw_mse = F.mse_loss(recon, images)
            raw_kl = kl_divergence_spatial(mu, logvar)
            
            current_kl_weight = config.KL_WEIGHT * 10.0 if latent_std < 0.3 else config.KL_WEIGHT
            diversity_loss = self.vae.diversity_loss if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)
            kl_annealing = min(1.0, self.epoch / config.KL_ANNEALING_EPOCHS)
            kl_loss = raw_kl * current_kl_weight * kl_annealing

            recon_loss = (raw_l1 * config.RECON_WEIGHT + 
                         config.PERCEPTUAL_WEIGHT * self.perceptual_loss(recon, images))
            
            # Total Loss with Classification, Text-Alignment, and Contrastive Loss
            text_align_weight = getattr(config, 'TEXT_ALIGN_WEIGHT', 1.0)
            contrastive_weight = getattr(config, 'CONTRASTIVE_WEIGHT', 0.1)
            
            total_loss = (recon_loss + kl_loss +
                         (diversity_loss * config.DIVERSITY_WEIGHT) +
                         (cls_loss * 0.5) +
                         (latent_alignment_loss * text_align_weight) +
                         (contrastive_loss_val * contrastive_weight))
            
            snr = calc_snr(images, recon)
            
            loss_dict = {
                'total': total_loss,
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'diversity': diversity_loss.item(),
                'cls': cls_loss.item(),
                'align': latent_alignment_loss.item(),
                'contrastive': contrastive_loss_val.item() if contrastive_loss_val > 0 else 0.0,
                'snr': snr,
                'latent_std': latent_std,
                'min_channel_std': min_channel_std,
            }

        else:  # Drift training (phase 2 or 3)
            with torch.no_grad():
                mu_ref, _ = self.vae_ref.encode(images, labels, text_emb)
            
            mu, logvar = self.vae.encode(images, labels, text_emb)
            consistency_loss = F.mse_loss(mu, mu_ref)

            temperature = config.TEMPERATURE_START + (config.TEMPERATURE_END - config.TEMPERATURE_START) * (self.epoch / config.EPOCHS)
            z1 = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * temperature
            
            t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            z0 = torch.randn_like(z1) * 0.8
            
            if config.USE_OU_BRIDGE and self.ou_ref is not None:
                mean, var = self.ou_ref.bridge_sample(z0, z1, t)
                zt = mean + torch.sqrt(var + 1e-8) * torch.randn_like(mean)
                target = self.ou_ref.bridge_velocity(z0, z1, t)
            else:
                t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
                zt = (1 - t_reshaped) * z0 + t_reshaped * z1
                target = z1 - z0
            
            # Classifier-Free Guidance dropout
            if self.drift.training and torch.rand(1).item() < config.LABEL_DROPOUT_PROB:
                train_labels, train_text = None, None
            else:
                train_labels, train_text = labels, text_emb

            pred = self.drift(zt, t, train_labels, train_text)
            
            t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
            time_weights = 1.0 + config.TIME_WEIGHT_FACTOR * t_reshaped
            drift_loss = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * config.DRIFT_WEIGHT

            drift_start_epoch = getattr(self, 'phase2_start_epoch', 50)
            consistency_decay = max(0.1, 1.0 - (self.epoch - drift_start_epoch) / (config.EPOCHS - drift_start_epoch))
            
            total_loss = drift_loss + (consistency_loss * config.CONSISTENCY_WEIGHT * consistency_decay)
            
            loss_dict = {'total': total_loss, 'drift': drift_loss.item(), 'consistency': consistency_loss.item(), 'temperature': temperature}
            
            if phase == 3:
                recon_p3 = self.vae.decode(mu, labels, text_emb)
                recon_loss_p3 = F.l1_loss(recon_p3, images) * config.RECON_WEIGHT * config.PHASE3_RECON_SCALE
                loss_dict['total'] = loss_dict['total'] + recon_loss_p3
                loss_dict['recon_p3'] = recon_loss_p3.item()
        
        # Add composite score
        loss_dict['composite_score'] = composite_score(loss_dict, phase)
        return loss_dict


    def _initialize_vgg(self):
        """Ensure VGG is initialized for perceptual loss or FID."""
        if not hasattr(self, 'vgg'):
            try:
                import torchvision.models as tv_models
                self.vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(config.DEVICE)
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.vgg.eval()
                config.logger.info("VGG16 initialized for quality assessment.")
            except Exception as e:
                config.logger.error(f"Failed to initialize VGG: {e}")
                return False
        return True

    def calculate_fid_batch(self, real_features, fake_features):
        """Calculate FID score using scipy for matrix square root."""
        mu1 = real_features.mean(dim=0).cpu().numpy()
        sigma1 = torch.cov(real_features.T).cpu().numpy()
        mu2 = fake_features.mean(dim=0).cpu().numpy()
        sigma2 = torch.cov(fake_features.T).cpu().numpy()
        
        diff = mu1 - mu2
        
        # Add regularization to prevent singularity with small batches
        reg = 1e-5 * np.eye(sigma1.shape[0])
        sigma1 += reg
        sigma2 += reg
        
        # Matrix square root of (sigma1 @ sigma2)
        covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Handle imaginary parts from numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def train_epoch(self) -> Dict:
        
        """Train for one epoch."""
        losses = defaultdict(float)
        snr_values = []
        latent_std_values = []
        channel_std_values = []
        
        current_epoch = self.epoch + 1
        phase = self.get_training_phase(current_epoch)
        mode = "VAE" if phase == 1 else "Drift"
        self.debug_counter = 0
        
        # Update epoch counter in VAE for adaptive diversity loss
        self.vae.current_epoch = current_epoch 

        # Set train/eval modes
        if phase == 1:
            self.vae.train()
            self.drift.eval()
        else:
            self.vae.train()
            self.drift.train()
        
        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(self.loader, desc=f"Epoch {current_epoch} ({mode})")
        else:
            pbar = self.loader
            config.logger.info(f"Epoch {current_epoch} ({mode})")
        
        batch_count = 0
        for batch_idx, batch in enumerate(pbar):
            try:
                # Prepare batch
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_dict = {'image': batch[0], 'label': batch[1], 'index': batch_idx}
                elif isinstance(batch, dict):
                    batch_dict = batch
                else:
                    continue
                
                # Forward pass with AMP if enabled
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.compute_loss(batch_dict, phase=phase, batch_idx=batch_idx)
                else:
                    loss_dict = self.compute_loss(batch_dict, phase=phase, batch_idx=batch_idx)
                
                # Check for NaN/Inf
                if not isinstance(loss_dict, dict) or 'total' not in loss_dict:
                    continue
                
                if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
                    config.logger.error(f" NaN/Inf detected at batch {batch_idx}!")
                    
                    # Try to revert
                    reverted = False
                    if phase == 1 and self.snapshot_vae:
                        config.logger.warning("Attempting to revert VAE to last good snapshot...")
                        reverted = self.snapshot_vae.revert()
                        if reverted:
                            config.logger.info("VAE reverted successfully. Skipping batch and continuing...")
                            continue
                    elif phase >= 2 and self.snapshot_drift:
                        config.logger.warning("Attempting to revert Drift to last good snapshot...")
                        reverted = self.snapshot_drift.revert()
                        if reverted:
                            config.logger.info("Drift reverted successfully. Skipping batch and continuing...")
                            continue
                    
                    if not reverted:
                        config.logger.error("No snapshot available to revert to! Stopping training...")
                        break
                                
                # Debug logging
                if batch_idx % 50 == 0 and batch_idx > 0:
                    self._debug_training_loop(batch_dict, loss_dict)

                if batch_idx % 100 == 0 and phase == 1:
                    # Check for collapse across batches
                    if len(latent_std_values) > 50:
                        avg_std = np.mean(latent_std_values[-50:])
                        if avg_std < 0.4 and self.epoch > 10:
                            config.logger.warning(f" Sustained low latent std: {avg_std:.3f}")
                            config.logger.warning("  Consider increasing KL_WEIGHT or reducing LATENT_SCALE")

                # Backward pass
                if phase == 1:
                    # Zero gradients for all optimizers
                    self.opt_vae.zero_grad()
                    self.opt_text.zero_grad()
                    if self.opt_image_proj is not None:
                        self.opt_image_proj.zero_grad()
                    
                    # Backward pass
                    loss_dict['total'].backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                    torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), config.GRAD_CLIP)
                    if self.image_projection is not None:
                        torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), config.GRAD_CLIP)
                    
                    # Optimizer steps
                    self.opt_vae.step()
                    self.opt_text.step()
                    if self.opt_image_proj is not None:
                        self.opt_image_proj.step()
                    
                    if 'snr' in loss_dict:
                        snr_values.append(loss_dict['snr'])
                    if 'latent_std' in loss_dict:
                        latent_std_values.append(loss_dict['latent_std'])
                    if 'min_channel_std' in loss_dict:
                        channel_std_values.append(loss_dict['min_channel_std'])
                elif phase == 2:
                    self.opt_drift.zero_grad()
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_dict['total']).backward()
                        self.scaler.unscale_(self.opt_drift)
                        # Use config factor for drift grad clip
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.scaler.step(self.opt_drift)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.opt_drift.step()
                elif phase == 3:
                    # In Phase 3, we update EVERYTHING
                    # Zero gradients for all optimizers
                    self.opt_vae.zero_grad()
                    self.opt_drift.zero_grad()
                    self.opt_text.zero_grad()
                    if self.opt_image_proj is not None:
                        self.opt_image_proj.zero_grad()
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_dict['total']).backward()
                        self.scaler.unscale_(self.opt_vae)
                        self.scaler.unscale_(self.opt_drift)
                        self.scaler.unscale_(self.opt_text)
                        if self.opt_image_proj is not None:
                            self.scaler.unscale_(self.opt_image_proj)
                        
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), config.GRAD_CLIP)
                        if self.image_projection is not None:
                            torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), config.GRAD_CLIP)
                        
                        self.scaler.step(self.opt_vae)
                        self.scaler.step(self.opt_drift)
                        self.scaler.step(self.opt_text)
                        if self.opt_image_proj is not None:
                            self.scaler.step(self.opt_image_proj)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), config.GRAD_CLIP)
                        if self.image_projection is not None:
                            torch.nn.utils.clip_grad_norm_(self.image_projection.parameters(), config.GRAD_CLIP)
                        
                        self.opt_vae.step()
                        self.opt_drift.step()
                        self.opt_text.step()
                        if self.opt_image_proj is not None:
                            self.opt_image_proj.step()
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in ['total', 'raw_mse', 'raw_kl', 'snr', 'latent_std', 
                                  'min_channel_std', 'max_channel_std', 'channel_stds', 
                                  'temperature', 'composite_score']:
                        if isinstance(value, (int, float)):
                            losses[key] += value
                        elif isinstance(value, torch.Tensor):
                            losses[key] += value.item()

                self.step += 1
                self.debug_counter += 1
                batch_count += 1
                
                # Update progress bar
                if TQDM_AVAILABLE and batch_idx % 10 == 0:
                    current_loss_val = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
                    avg_snr = np.mean(snr_values[-10:]) if snr_values else 0
                    postfix = {'loss': f"{current_loss_val:.4f}"}
                    if snr_values:
                        postfix['snr'] = f"{avg_snr:.1f}dB"
                    if channel_std_values:
                        postfix['min_std'] = f"{channel_std_values[-1]:.3f}"
                    pbar.set_postfix(postfix)
                    
            except Exception as e:
                config.logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
                        
                        
        # Periodic FID calculation (every 20 epochs)
        if (self.epoch + 1) % 20 == 0 and phase == 1:
            try:
                if self._initialize_vgg():
                    # Collect more samples for more stable FID
                    with torch.no_grad():
                        fid_samples = 32 # Increased from 16
                        batch = next(iter(self.loader))
                        real_images = batch['image'][:fid_samples].to(config.DEVICE)
                        real_labels = batch['label'][:fid_samples].to(config.DEVICE)
                        
                        # Correct VGG normalization
                        real_norm = ((real_images + 1) / 2 - self.vgg_mean) / self.vgg_std
                        real_feat = self.vgg(real_norm)
                        
                        gen_images, _, _ = self.vae(real_images, real_labels)
                        gen_norm = ((gen_images + 1) / 2 - self.vgg_mean) / self.vgg_std
                        gen_feat = self.vgg(gen_norm)
                        
                        fid_score = self.calculate_fid_batch(real_feat.flatten(1), gen_feat.flatten(1))
                        losses['fid'] = fid_score
                        config.logger.info(f"FID Score (approx): {fid_score:.2f}")
            except Exception as e:
                config.logger.warning(f"FID calculation failed: {e}")

        # Compute average losses
        if batch_count > 0:
            avg_losses = {}
            for key, value in losses.items():
                avg_losses[key] = value / batch_count

            if avg_losses:
                avg_losses['total'] = sum(avg_losses.values())
            else:
                avg_losses['total'] = 0.0            
            
            if snr_values:
                avg_losses['snr'] = np.mean(snr_values)
            if latent_std_values:
                avg_losses['latent_std'] = np.mean(latent_std_values)
            if channel_std_values:
                avg_losses['min_channel_std'] = np.mean(channel_std_values)
        else:
            avg_losses = {'total': float('inf')}
            config.logger.error("No batches were successfully processed in this epoch!")
        
        # Log results
        config.logger.info(f"Epoch {current_epoch}/{config.EPOCHS} complete:")
        config.logger.info(f"  Total loss: {avg_losses.get('total', 0):.4f}")
        
        if phase == 1:
            self.scheduler_vae.step()
            if self.snapshot_vae and self.snapshot_vae.should_save(self.epoch):
                loss_value = avg_losses.get('total', float('inf'))
                if loss_value != float('inf'):
                    self.snapshot_vae.save_snapshot(
                        epoch=self.epoch + 1,
                        loss=loss_value
                    )
            config.logger.info(f"  Recon loss: {avg_losses.get('recon', 0):.4f}")
            config.logger.info(f"  KL loss: {avg_losses.get('kl', 0):.6f}")
            config.logger.info(f"  Diversity loss: {avg_losses.get('diversity', 0):.6f}")
            config.logger.info(f"  Latent std: {avg_losses.get('latent_std', 0):.3f}")
            if 'snr' in avg_losses:
                config.logger.info(f"  SNR: {avg_losses['snr']:.2f}dB")
        else:
            self.scheduler_drift.step()
            if self.snapshot_drift and self.snapshot_drift.should_save(self.epoch):
                loss_value = avg_losses.get('drift', avg_losses.get('total', float('inf')))
                if loss_value != float('inf'):
                    self.snapshot_drift.save_snapshot(
                        epoch=self.epoch + 1,
                        loss=loss_value
                    )
            config.logger.info(f"  Drift loss: {avg_losses.get('drift', 0):.4f}")
        
        return avg_losses

    def _debug_training_loop(self, batch: Dict, loss_dict: Dict) -> None:
        """Debug helper for training loop."""
        config.logger.info("=== DEBUG INFO ===")
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                config.logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                config.logger.info(f"  {key}: {value.item():.4f}")

    def save_checkpoint(self, is_best: bool = False, is_best_overall: bool = False) -> Path:
        """Save training checkpoint."""
        return dm.save_checkpoint(self, is_best, is_best_overall)

    def load_checkpoint(self, path: Optional[Path] = None) -> bool:
        """Load training checkpoint."""
        return dm.load_checkpoint(self, path)

    def load_for_inference(self, path: Optional[Path] = None) -> bool:
        """Load model for inference."""
        return dm.load_for_inference(self, path)

    def load_from_snapshot(self, snapshot_path: Path, load_vae: bool = True, 
                        load_drift: bool = True, phase: Optional[int] = None) -> bool:
        """Load model from a snapshot file."""
        if not os.path.exists(snapshot_path):
            config.logger.error(f"Snapshot not found: {snapshot_path}")
            return False
        
        try:
            snapshot = torch.load(snapshot_path, map_location=config.DEVICE, weights_only=False)
            
            # Load VAE if requested and available
            if load_vae:
                if 'model_state' in snapshot:
                    self.vae.load_state_dict(snapshot['model_state'])
                    if 'optimizer_state' in snapshot:
                        self.opt_vae.load_state_dict(snapshot['optimizer_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'vae' and 'model_state' in snapshot:
                    self.vae.load_state_dict(snapshot['model_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                else:
                    config.logger.warning("No VAE state found in snapshot")
            
            # Load Drift if requested and available
            if load_drift:
                if 'drift_state' in snapshot:
                    self.drift.load_state_dict(snapshot['drift_state'])
                    if 'opt_drift_state' in snapshot:
                        self.opt_drift.load_state_dict(snapshot['opt_drift_state'])
                    config.logger.info(f"✅ Loaded Drift from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'drift' and 'drift_state' in snapshot:
                    self.drift.load_state_dict(snapshot['drift_state'])
                    config.logger.info(f"✅ Loaded Drift from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                else:
                    config.logger.warning("No Drift state found in snapshot")
            
            if phase is not None and phase >= 2 and (not hasattr(self, 'vae_ref') or self.vae_ref is None):
                self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
                self.vae_ref.load_state_dict(self.vae.state_dict())
                self.vae_ref.eval()
                for param in self.vae_ref.parameters():
                    param.requires_grad = False
                config.logger.info("Reference anchor created from loaded snapshot.")
                
            # Set phase if specified
            if phase is not None:
                self.phase = phase
                config.logger.info(f"Set phase to {phase}")
            
            # Set epoch from snapshot
            if 'epoch' in snapshot:
                self.epoch = snapshot['epoch']
            
            return True
            
        except Exception as e:
            config.logger.error(f"Failed to load snapshot: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def generate_samples(self, labels=None, text_emb=None, num_samples=8, temperature=None, method='heun',
                         langevin_steps=None, langevin_step_size=None, langevin_score_scale=None,
                         cfg_scale=None):
        """
        Generate samples with multimodal conditioning.

        Args:
            labels: List of class labels
            text_emb: Text embeddings [B, D] or [B, S, D]
            num_samples: Number of samples to generate
            temperature: Ignored (kept for API compatibility)
            method: 'euler', 'heun', or 'rk4'
            langevin_steps: Number of Langevin refinement steps
            langevin_step_size: Step size for Langevin dynamics
            langevin_score_scale: Scaling factor for the approximate score
            cfg_scale: Scale for classifier-free guidance
        """
        if cfg_scale is None:
            cfg_scale = getattr(config, 'CFG_SCALE', 1.0)
        if langevin_steps is None:
            langevin_steps = getattr(config, 'DEFAULT_LANGEVIN_STEPS', 0)
        if langevin_step_size is None:
            langevin_step_size = config.LANGEVIN_STEP_SIZE
        if langevin_score_scale is None:
            langevin_score_scale = config.LANGEVIN_SCORE_SCALE

        self.vae.eval()
        self.drift.eval()
        
        if labels is None and text_emb is None:
            labels = [i % 10 for i in range(num_samples)]
        
        with torch.no_grad():
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=config.DEVICE) if labels is not None else None
            text_tensor = text_emb.to(config.DEVICE) if text_emb is not None else None
            
            # Use provided num_samples or derive from context
            if labels_tensor is not None:
                num_samples = labels_tensor.shape[0]
            elif text_tensor is not None:
                num_samples = text_tensor.shape[0]

            # Prior distribution: standard normal in latent space
            z = torch.randn(num_samples, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE)

            config.logger.info(f"Initial z std: {z.std():.4f}")
            
            steps = config.DEFAULT_STEPS
            dt = 1.0 / steps
            
            # ----- ODE integration -----
            for i in range(steps):
                t_cur = torch.full((num_samples, 1), i * dt, device=config.DEVICE)

                if method == 'euler':
                    drift = self.drift(z, t_cur, labels_tensor, text_tensor, cfg_scale=cfg_scale)
                    z = z + drift * dt
                elif method == 'heun':
                    k1 = self.drift(z, t_cur, labels_tensor, text_tensor, cfg_scale=cfg_scale)
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_pred = z + dt * k1
                    k2 = self.drift(z_pred, t_next, labels_tensor, text_tensor, cfg_scale=cfg_scale)
                    z = z + (dt / 2.0) * (k1 + k2)
                
                z = torch.clamp(z, -10, 10)
                                
                if i % 20 == 0:
                    config.logger.info(f"Step {i:3d}, t={i*dt:.3f}, z std: {z.std():.4f}")
            
            # Decode
            images = self.vae.decode(z, labels_tensor, text_tensor)
            images = torch.clamp(images, -1, 1)
            
            # Save results
            images_display = (images + 1) / 2
            grid = vutils.make_grid(images_display, nrow=4, padding=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            grid_path = config.DIRS["samples"] / f"gen_multimodal_{timestamp}.png"
            vutils.save_image(grid, grid_path)
            
            config.logger.info(f"Saved multimodal samples to: {grid_path}")
            return grid_path

    def list_available_snapshots(self) -> List[Path]:
        """List all available snapshots."""
        snap_files = list(config.DIRS["snaps"].glob("*_snapshot_epoch_*.pt"))
        snap_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return snap_files

    def inspect_snapshot(self, snapshot_path: Path) -> Dict:
        """Inspect snapshot contents without loading."""
        try:
            snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            info = {
                'path': snapshot_path,
                'epoch': snapshot.get('epoch', 'unknown'),
                'loss': snapshot.get('loss', 'N/A'),
                'timestamp': snapshot.get('timestamp', 'unknown'),
                'model_type': snapshot.get('model_type', 'unknown'),
                'has_vae': 'model_state' in snapshot or snapshot.get('model_type') == 'vae',
                'has_drift': 'drift_state' in snapshot or snapshot.get('model_type') == 'drift',
            }
            return info
        except Exception as e:
            config.logger.error(f"Failed to inspect snapshot: {e}")
            return {'path': snapshot_path, 'error': str(e)}


    def export_onnx(self) -> None:
        if not ONNX_AVAILABLE:
            config.logger.warning("ONNX export requires onnx and onnxruntime packages")
            return
        
        # Helper to set export mode on PercentileRescale modules
        def set_export_mode(module, mode=True):
            if hasattr(module, '_set_export_mode'):
                module._set_export_mode(mode)
        
        # Helper to merge .onnx.data files back into the .onnx file
        def merge_external_data(model_path):
            try:
                import onnx
                from pathlib import Path
                model_path = Path(model_path)
                if not model_path.exists():
                    return
                
                # Load model and external data automatically
                model = onnx.load(str(model_path))
                # Save model as a single file (default behavior when not specifying external data location)
                onnx.save(model, str(model_path))
                
                # Remove the now redundant .data file
                data_path = model_path.with_suffix(model_path.suffix + ".data")
                if data_path.exists():
                    data_path.unlink()
                    config.logger.info(f"Merged and removed external data: {data_path.name}")
            except Exception as e:
                config.logger.warning(f"Could not merge external data for {model_path.name}: {e}")

        # Wrapper class to export ONLY the decoder (generator) part of the VAE
        class VAEMultimodalGenerator(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, z, labels=None, text_emb=None):
                return self.vae.decode(z, labels, text_emb)

        # New: Wrapper for VAE Encoder and Classifier
        class VAEMultimodalEncoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, x, labels=None, text_emb=None):
                mu, logvar = self.vae.encode(x, labels, text_emb)
                # Also return predicted logits for Image-to-Text
                _, logits = self.vae.context_decoder(mu)
                return mu, logits

        try:
            # --- Set export mode for VAE and Drift ---
            self.vae.eval()
            self.drift.eval()
            self.vae.apply(lambda m: set_export_mode(m, True))
            self.drift.apply(lambda m: set_export_mode(m, True))
            
            dummy_z = torch.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE)
            dummy_img = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=config.DEVICE)
            dummy_label = torch.tensor([0], device=config.DEVICE, dtype=torch.long)
            dummy_text = torch.zeros(1, config.TEXT_EMBEDDING_DIM, device=config.DEVICE)

            # Export Encoder
            enc_path = config.DIRS["onnx"] / "encoder.onnx"
            vae_enc = VAEMultimodalEncoder(self.vae)
            with torch.no_grad():
                torch.onnx.export(
                    vae_enc,
                    (dummy_img, dummy_label, dummy_text),
                    str(enc_path),
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['image', 'label', 'text_emb'],
                    output_names=['mu', 'logits'],
                    dynamic_axes={
                        'image': {0: 'batch_size'},
                        'label': {0: 'batch_size'},
                        'text_emb': {0: 'batch_size'},
                        'mu': {0: 'batch_size'},
                        'logits': {0: 'batch_size'}
                    }
                )
            config.logger.info(f"Encoder exported to {enc_path}")

            # Export Generator (Decoder only)
            gen_path = config.DIRS["onnx"] / "generator.onnx"
            vae_gen = VAEMultimodalGenerator(self.vae)
            
            with torch.no_grad():
                torch.onnx.export(
                    vae_gen,
                    (dummy_z, dummy_label, dummy_text),
                    str(gen_path),
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['z', 'label', 'text_emb'],
                    output_names=['reconstruction'],
                    dynamic_axes={
                        'z': {0: 'batch_size'},
                        'label': {0: 'batch_size'},
                        'text_emb': {0: 'batch_size'},
                        'reconstruction': {0: 'batch_size'}
                    }
                )
            merge_external_data(gen_path)
            config.logger.info(f"Generator exported to {gen_path}")
            
            # Export Drift
            dummy_t = torch.tensor([[0.5]], device=config.DEVICE)
            
            drift_path = config.DIRS["onnx"] / "drift.onnx"
            
            with torch.no_grad():
                torch.onnx.export(
                    self.drift,
                    (dummy_z, dummy_t, dummy_label, dummy_text),
                    str(drift_path),
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['z', 't', 'label', 'text_emb'],
                    output_names=['drift'],
                    dynamic_axes={
                        'z': {0: 'batch_size'},
                        't': {0: 'batch_size'},
                        'label': {0: 'batch_size'},
                        'text_emb': {0: 'batch_size'},
                        'drift': {0: 'batch_size'}
                    }
                )
            merge_external_data(drift_path)
            config.logger.info(f"Drift exported to {drift_path}")

            # Export Text Encoder
            text_path = config.DIRS["onnx"] / "text_encoder.onnx"
            dummy_tokens = torch.tensor([0], device=config.DEVICE, dtype=torch.long)
            
            with torch.no_grad():
                torch.onnx.export(
                    self.text_encoder,
                    (dummy_tokens,),
                    str(text_path),
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['tokens'],
                    output_names=['text_emb'],
                    dynamic_axes={
                        'tokens': {0: 'batch_size'},
                        'text_emb': {0: 'batch_size'}
                    }
                )
            config.logger.info(f"Text Encoder exported to {text_path}")

            # --- Auto-configure the HTML file to match the current dimensions ---
            html_path = Path("onnx_generate_image.html")
            if html_path.exists():
                try:
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    import re
                    # Replace LATENT_SHAPE array (handles let or const, and varied spacing)
                    html_content = re.sub(
                        r'(let|const)\s+LATENT_SHAPE\s*=\s*\[1,\s*\d+,\s*\d+,\s*\d+\];',
                        f'\\1 LATENT_SHAPE = [1, {config.LATENT_CHANNELS}, {config.LATENT_H}, {config.LATENT_W}];',
                        html_content
                    )
                    # Replace IMG_SIZE constant
                    html_content = re.sub(
                        r'(let|const)\s+IMG_SIZE\s*=\s*\d+;',
                        f'\\1 IMG_SIZE = {config.IMG_SIZE};',
                        html_content
                    )

                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    config.logger.info(f"Updated {html_path.name} with new dimensions (IMG_SIZE={config.IMG_SIZE}).")
                except Exception as e:
                    config.logger.warning(f"Could not auto-update HTML file dimensions: {e}")

            # --- Reset export mode ---
            self.vae.apply(lambda m: set_export_mode(m, False))
            self.drift.apply(lambda m: set_export_mode(m, False))
            
        except Exception as e:
            config.logger.error(f"ONNX export failed: {e}")


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(num_epochs: int = config.EPOCHS, resume_from_snapshot: Optional[Path] = None) -> None:
    """Main training loop."""
    loader = dm.load_data()
    trainer = EnhancedLabelTrainer(loader)
    
    if resume_from_snapshot:
        print(f"\n Resuming from snapshot: {resume_from_snapshot}")
        trainer.load_from_snapshot(resume_from_snapshot)
    else:
        latest_checkpoint = config.DIRS["ckpt"] / "latest.pt"
        if latest_checkpoint.exists():
            resume = input("\n Found existing checkpoint. Resume training? (y/n): ").strip().lower()
            if resume == 'y':
                trainer.load_checkpoint()
            else:
                print("Starting fresh training...")
    
    config.logger.info(f"Starting training for {num_epochs} epochs")
    config.logger.info(f"Training schedule mode: {config.TRAINING_SCHEDULE['mode']}")
    
    for epoch in range(trainer.epoch, num_epochs):
        trainer.epoch = epoch
        epoch_losses = trainer.train_epoch()
        
        if config.USE_KPI_TRACKING:
            kpi_update = {
                'latent_std': epoch_losses.get('latent_std', 0),
                'lr_vae': trainer.opt_vae.param_groups[0]['lr'],
                'lr_drift': trainer.opt_drift.param_groups[0]['lr']
            }
            if 'snr' in epoch_losses:
                kpi_update['snr'] = epoch_losses['snr']
            if trainer.phase == 1:
                kpi_update['loss'] = epoch_losses.get('total', 0)
                kpi_update['recon_loss'] = epoch_losses.get('recon', 0)
                kpi_update['kl_loss'] = epoch_losses.get('kl', 0)
                kpi_update['diversity_loss'] = epoch_losses.get('diversity', 0)
                kpi_update['min_channel_std'] = epoch_losses.get('min_channel_std', 0)
            else:
                kpi_update['loss'] = epoch_losses.get('drift', 0)
                kpi_update['drift_loss'] = epoch_losses.get('drift', 0)
            
            # Compute composite score
            comp_score = composite_score(epoch_losses, trainer.phase)
            kpi_update['composite_score'] = comp_score
            trainer.kpi_tracker.update(kpi_update)
            
            # Log convergence metrics periodically (e.g., every 5 epochs)
            if (epoch + 1) % 5 == 0:
                conv_stats = trainer.kpi_tracker.compute_convergence()
                if 'loss_trend' in conv_stats:
                    trend_symbol = "\\" if conv_stats['loss_trend'] < 0 else "//"
                    config.logger.info(
                        f"Convergence Stats {trend_symbol} | Loss trend: {conv_stats['loss_trend']:.6f} | "
                        f"Stability Score: {conv_stats.get('convergence_score', 0):.4f}"
                    )

            # Check for new best composite score
            if comp_score > trainer.best_composite_score:
                trainer.best_composite_score = comp_score
                trainer.save_checkpoint(is_best=False, is_best_overall=True)
        
        # Check for best loss
        if trainer.phase == 1:
            current_total_loss = epoch_losses.get('total', float('inf'))
        else:
            current_total_loss = epoch_losses.get('drift', float('inf'))
        
        if current_total_loss < trainer.best_loss and current_total_loss != float('inf'):
            trainer.best_loss = current_total_loss
            trainer.save_checkpoint(is_best=True, is_best_overall=False)
        elif (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(is_best=False, is_best_overall=False)
        
        # Generate samples periodically
        if (epoch + 1) % 10 == 0 and current_total_loss != float('inf'):
            config.logger.info("Generating samples...")
            trainer.generate_samples()
        
        # Check early stopping
        if config.USE_KPI_TRACKING and trainer.phase == 2:
            if trainer.kpi_tracker.should_stop(phase=trainer.phase):
                config.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    config.logger.info(f"Training complete! Best loss: {trainer.best_loss:.4f}")
    config.logger.info(f"Best composite score: {trainer.best_composite_score:.4f}")
    
    if ONNX_AVAILABLE:
        trainer.export_onnx()
    
    trainer.generate_samples(labels=list(range(8)))