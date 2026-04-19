# ============================================================================
# CONTRASTIVE LOSS FOR MODALITY ALIGNMENT (NEW)
# ============================================================================
import torch.nn as nn
import torch.nn.functional as F
import config

class ContrastiveLoss(nn.Module):
    """InfoNCE loss for image-text alignment."""
    def __init__(self, temperature=config.CONTRASTIVE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, image_emb, text_emb):
        # Normalize embeddings to unit hypersphere
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Compute cosine similarity matrix: [B, B]
        logits = torch.matmul(image_emb, text_emb.T) / self.temperature
        
        # Labels are the diagonal (matching indices)
        batch_size = image_emb.size(0)
        labels = torch.arange(batch_size, device=image_emb.device)
        
        # Symmetric loss: image-to-text and text-to-image
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

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
import torch.amp
import torchvision.utils as vutils
import config
import data_management
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
import lora

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

def total_variation_loss(img: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    Compute Total Variation (TV) loss to encourage spatial smoothness.
    Penalizes high-frequency noise and sharp discontinuities.
    """
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).mean()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).mean()
    return weight * (tv_h + tv_w)

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
            score -= loss_dict['diversity'] * 100
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
            phase = 1
        elif epoch < e2:
            phase = 2
        else:
            phase = 3
        # config.logger.debug(f"Three-phase logic: epoch={epoch}, e1={e1}, e2={e2} -> phase={phase}")
        return phase
    
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
    """Main trainer class for label-conditioned Schrödinger Bridge."""
    
    def __init__(self, loader):
        self.loader = loader

        self.vae = models.LabelConditionedVAE().to(config.DEVICE)
        self.drift = models.LabelConditionedDrift().to(config.DEVICE)
        
        # Apply LoRA if enabled in config
        if config.USE_LORA:
            import lora
            n_vae = lora.apply_lora(self.vae, r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, 
                                    lora_dropout=config.LORA_DROPOUT, target_modules=config.LORA_TARGET_MODULES)
            n_drift = lora.apply_lora(self.drift, r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, 
                                      lora_dropout=config.LORA_DROPOUT, target_modules=config.LORA_TARGET_MODULES)
            
            tr_v, tot_v = lora.count_lora_params(self.vae)
            tr_d, tot_d = lora.count_lora_params(self.drift)
            config.logger.info(f"🚀 LoRA enabled: VAE ({n_vae} layers, {tr_v/tot_v:.1%} trainable), "
                               f"Drift ({n_drift} layers, {tr_d/tot_d:.1%} trainable)")

        # Only optimize parameters with requires_grad=True (filters frozen base weights)
        self.opt_vae = optim.AdamW(filter(lambda p: p.requires_grad, self.vae.parameters()), 
                                   lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        # Drift optimizer with multiplier from config
        self.opt_drift = optim.AdamW(
            filter(lambda p: p.requires_grad, self.drift.parameters()),
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

        # Multimodal Alignment Loss
        self.contrastive_criterion = ContrastiveLoss().to(config.DEVICE)
        
        self.epoch = 0
        self.step = 0
        self.phase = 1
        self.best_loss = float('inf')
        self.best_composite_score = float('-inf')
        self.debug_counter = 0
        self.debug_interval = 10
        self.phase2_start_epoch = None
        
        # Reference for Phase 2
        self.vae_ref = None
        
        # OU reference process
        self.ou_ref = OUReference(theta=config.OU_THETA, sigma=config.OU_SIGMA) if config.USE_OU_BRIDGE else None
        
        # AMP Scaler (Cross-backend support)
        self.scaler = None
        if config.USE_AMP and config.AMP_AVAILABLE:
            if config.DEVICE.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            elif config.DEVICE.type == 'xpu':
                # Intel XPU uses the same GradScaler API or its own depending on torch version
                try:
                    self.scaler = torch.xpu.amp.GradScaler()
                except (AttributeError, ImportError):
                    self.scaler = torch.cuda.amp.GradScaler() # Fallback
        
        config.logger.info(f"💓 Epoch 0 | Batch 0/{len(self.loader)}")
        config.logger.info(f"Models initialized:")
        config.logger.info(f"  VAE params: {sum(p.numel() for p in self.vae.parameters()):,}")
        config.logger.info(f"  Drift params: {sum(p.numel() for p in self.drift.parameters()):,}")
        if config.USE_OU_BRIDGE:
            config.logger.info(f"  Using OU bridge reference (theta={config.OU_THETA})")

        # ImageNet Mean and Std (for 0-1 normalized images) - Registered as buffers
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # Pre-compute SSIM Gaussian Window
        window_size = 11
        sigma = 1.5
        gauss = torch.arange(window_size, dtype=torch.float32)
        gauss = torch.exp(-(gauss - window_size//2)**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] * gauss[None, :]
        window = window[None, None, :, :]
        # Cache it for the 3 channels - Registered as buffer
        self.register_buffer('ssim_window', window.expand(3, -1, -1, -1).contiguous())


    def register_buffer(self, name, tensor):
        """Helper to register a buffer on the trainer (since it's not an nn.Module)."""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor.to(config.DEVICE)
        setattr(self, name, self._buffers[name])

    def to_device(self, device):
        """Move all buffers to device."""
        if hasattr(self, '_buffers'):
            for name in self._buffers:
                self._buffers[name] = self._buffers[name].to(device)
                setattr(self, name, self._buffers[name])
        self.vae.to(device)
        self.drift.to(device)
        if hasattr(self, 'vgg'):
            self.vgg.to(device)
        return self


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
                import torchvision.models as models
                self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(config.DEVICE)
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.vgg.eval()
            except:
                return torch.tensor(0.0, device=config.DEVICE)
    
        # Ensure mean/std match the precision of recon (Crucial for AMP)
        mean = self.vgg_mean.to(recon.dtype)
        std = self.vgg_std.to(recon.dtype)

        # 1. Map from [-1, 1] to [0, 1]
        recon_01 = (recon + 1) / 2
        target_01 = (target + 1) / 2
        
        # 2. Apply ImageNet normalization - added .contiguous() for MPS stability
        recon_norm = ((recon_01 - mean) / std).contiguous()
        target_norm = ((target_01 - mean) / std).contiguous()
        
        # 3. Get features
        # Ensure VGG also operates in the same precision as inputs
        self.vgg.to(recon.dtype)
        recon_feat = self.vgg(recon_norm)
        with torch.no_grad():
            target_feat = self.vgg(target_norm)
        
        return F.mse_loss(recon_feat, target_feat)


    def _switch_to_phase(self, new_phase: int):
        """Handle transition between training phases."""
        if new_phase == 1:
            # Phase 1: VAE only
            self.vae.train()
            self.drift.eval()
            
            # Ensure VAE params are trainable (respecting LoRA)
            for name, param in self.vae.named_parameters():
                if config.USE_LORA:
                    param.requires_grad = ("lora_" in name)
                else:
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
            self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
            # Use flexible_load to ensure LoRA mapping if needed
            dm.flexible_load(self.vae_ref, self.vae.state_dict())
            self.vae_ref.eval()
            for param in self.vae_ref.parameters():
                param.requires_grad = False

            # Unfreeze encoder parts only, explicitly zero-out grad for frozen params
            unfrozen_count = 0
            for name, param in self.vae.named_parameters():
                # Check if it's an encoder part
                is_encoder = any(k in name for k in ['enc_', 'label_emb', 'z_mean', 'z_logvar', 'source_emb', 'cond_proj'])
                
                if is_encoder:
                    if config.USE_LORA:
                        param.requires_grad = ("lora_" in name)
                    else:
                        param.requires_grad = True
                    
                    if param.requires_grad:
                        unfrozen_count += param.numel()
                else:
                    param.requires_grad = False
                    param.grad = None
            
            config.logger.info(f"Phase 2: Unfrozen {unfrozen_count:,} encoder trainable params. Anchor set.")

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
                if config.USE_LORA:
                    param.requires_grad = ("lora_" in name)
                else:
                    param.requires_grad = True
            
            # Ensure reference anchor exists if we skipped Phase 2
            if not hasattr(self, 'vae_ref') or self.vae_ref is None:
                self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
                dm.flexible_load(self.vae_ref, self.vae.state_dict())
                self.vae_ref.eval()
                for param in self.vae_ref.parameters():
                    param.requires_grad = False
                config.logger.info("Phase 3: Reference anchor created (transitioned from Phase 1).")

            config.logger.info("Phase 3: Unfroze all trainable VAE parameters (encoder + decoder).")

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
        mu_y = F.conv2d(y, window, padding=pad, groups=y.size(1))
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=x.size(1)) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=y.size(1)) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=y.size(1)) - mu_xy
        
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
        # Extract images, labels and optional text_bytes
        if isinstance(batch, dict):
            images = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            text_bytes = batch.get('text_bytes', None)
            if text_bytes is not None:
                text_bytes = text_bytes.to(config.DEVICE)
            source_id = batch.get('source_id', None)
            if source_id is not None:
                source_id = source_id.to(config.DEVICE)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            images = batch[0].to(config.DEVICE)
            labels = batch[1].to(config.DEVICE)
            text_bytes = None
            source_id = None
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        if phase == 1:
            # Phase 1: Train VAE + Neural Tokenizer Alignment
            recon, mu, logvar = self.vae(images, labels, text_bytes=text_bytes, source_id=source_id)
            
            # --- Multimodal Contrastive Alignment (NEW) ---
            contrastive_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.USE_NEURAL_TOKENIZER and text_bytes is not None and config.USE_PROJECTION_HEADS:
                # 1. Get text embedding: [B, 512]
                text_emb = self.vae.text_encoder(text_bytes)
                
                # 2. Project image latent to shared space: [B, 1152] -> [B, 512]
                # Flatten mu: [B, 8, 12, 12] -> [B, 1152]
                z_flat = mu.flatten(start_dim=1)
                image_emb = self.vae.image_proj(z_flat)
                
                # 3. Compute InfoNCE loss
                contrastive_loss = self.contrastive_criterion(image_emb, text_emb)
            
            # Compute VAE metrics
            latent_std = torch.exp(0.5 * logvar).mean().item()
            channel_stds = mu.std(dim=[0, 2, 3]).detach().cpu().to(torch.float32).numpy()
            min_channel_std = channel_stds.min()
            
            # Adaptive KL weight based on channel usage
            raw_l1 = F.l1_loss(recon, images)
            raw_mse = F.mse_loss(recon, images) 
            raw_kl = kl_divergence_spatial(mu, logvar)
            
            current_kl_weight = config.KL_WEIGHT
            diversity_loss = self.vae.diversity_loss if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)

            kl_annealing = min(1.0, self.epoch / config.KL_ANNEALING_EPOCHS)
            kl_loss = raw_kl * current_kl_weight * kl_annealing 

            recon_loss = (raw_l1 * config.RECON_WEIGHT + 
                         config.PERCEPTUAL_WEIGHT * self.perceptual_loss(recon, images))
            
            # Add SSIM loss for structural integrity
            if config.SSIM_WEIGHT > 0:
                ssim_loss = self.ssim_loss(recon, images) * config.SSIM_WEIGHT
            else:
                ssim_loss = torch.tensor(0.0, device=config.DEVICE)

            # --- Texture and Sharpness Enhancements ---
            # 1. Edge Loss (Sobel-like gradient matching)
            recon_grad_x = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1])
            recon_grad_y = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :])
            image_grad_x = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
            image_grad_y = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
            edge_loss = (F.mse_loss(recon_grad_x, image_grad_x) + F.mse_loss(recon_grad_y, image_grad_y)) * config.EDGE_WEIGHT

            # 2. Total Variation (TV) Loss for denoising
            tv_loss = total_variation_loss(recon, weight=config.TV_WEIGHT)

            # Combined VAE loss
            total_loss = recon_loss + kl_loss + ssim_loss + edge_loss + tv_loss + diversity_loss * config.DIVERSITY_WEIGHT
            
            # Add Contrastive loss with annealing
            if config.USE_NEURAL_TOKENIZER:
                # Anneal contrastive weight to focus on alignment early, then generation
                c_weight = config.CONTRASTIVE_WEIGHT * min(1.0, self.epoch / 10.0)
                total_loss = total_loss + contrastive_loss * c_weight
            
            snr = calc_snr(images, recon)
            
            loss_dict = {
                'total': total_loss,
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'diversity': diversity_loss.item(),
                'contrastive': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0,
                'ssim_loss': ssim_loss.item(),
                'edge_loss': edge_loss.item(),
                'tv_loss': tv_loss.item(),
                'snr': snr,
                'latent_std': latent_std,
            }
            
            # Log gradient magnitude every 10 epochs for sharpness monitoring
            if self.epoch % 10 == 0 and phase == 1 and batch_idx % 100 == 0:
                with torch.no_grad():
                    # Check image sharpness via gradients
                    grad_x = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]).mean().item()
                    grad_y = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]).mean().item()
                    config.logger.info(f"Image gradient magnitude - X: {grad_x:.4f}, Y: {grad_y:.4f}")
                    
                    # Check latent statistics
                    latent_std_channel = mu.std(dim=[0, 2, 3])
                    config.logger.info(f"Channel utilization - min: {latent_std_channel.min().item():.4f}, max: {latent_std_channel.max().item():.4f}")

        else:  # Drift training (phase 2 or 3)
            # Ensure vae_ref exists (extra safety check)
            if not hasattr(self, 'vae_ref') or self.vae_ref is None:
                 config.logger.warning("vae_ref was None during Phase 2/3. Initializing now.")
                 self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
                 self.vae_ref.load_state_dict(self.vae.state_dict())
                 self.vae_ref.eval()
                 for param in self.vae_ref.parameters():
                     param.requires_grad = False

            # Get the epoch when drift training started
            drift_start_epoch = getattr(self, 'phase2_start_epoch', None)
            if drift_start_epoch is None:
                drift_start_epoch = config.TRAINING_SCHEDULE.get('switch_epoch', 50)
            
            with torch.no_grad():
                mu_ref, _ = self.vae_ref.encode(images, labels, text_bytes=text_bytes, source_id=source_id)   # always use frozen anchor
            mu, logvar = self.vae.encode(images, labels, text_bytes=text_bytes, source_id=source_id)
            
            # --- GLOBAL SPATIAL HARMONIZER ---
            # Use 'mu' as the reference spatial shape
            target_shape = mu.shape[2:]
            if mu_ref.shape[2:] != target_shape:
                mu_ref = F.interpolate(mu_ref, size=target_shape, mode='bilinear', align_corners=False)
            if logvar.shape[2:] != target_shape:
                logvar = F.interpolate(logvar, size=target_shape, mode='bilinear', align_corners=False)
                
            std_from_logvar = torch.exp(0.5 * logvar).mean().item()
            consistency_loss = F.mse_loss(mu, mu_ref)
            mu_global_std = mu.std().item()

            # Temperature annealing using config values
            temperature = config.TEMPERATURE_START + (config.TEMPERATURE_END - config.TEMPERATURE_START) * (self.epoch / config.EPOCHS)
            
            # Ensure logvar/mu match exactly for z1
            mu_for_drift = mu.detach()
            logvar_for_drift = logvar.detach()
            
            z1_noise = torch.exp(0.5 * logvar_for_drift) * torch.randn_like(logvar_for_drift) * temperature
            z1 = mu_for_drift + z1_noise
            z_global_std = z1.std().item()
            
            # Sample time with beta distribution after sufficient training
            if self.epoch > config.TRAINING_SCHEDULE.get('switch_epoch', 50) + config.EPOCHS // 6:  # After 1/6 of drift training
                # Move parameters to device for beta distribution
                alpha = torch.tensor([2.0], device=config.DEVICE)
                beta = torch.tensor([2.0], device=config.DEVICE)
                beta_dist = torch.distributions.Beta(alpha, beta)
                t = beta_dist.sample((images.shape[0],)).reshape(-1, 1)
            else:
                t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            
            # Start from noise with std defined in config
            z0 = torch.randn_like(z1) * config.CST_COEF_GAUSSIAN_PRIO
            
            # Sample intermediate latent using either linear interpolation or OU bridge
            if config.USE_OU_BRIDGE and self.ou_ref is not None:
                mean, var = self.ou_ref.bridge_sample(z0, z1, t)
                # Harmonize bridge outputs
                if mean.shape[2:] != target_shape:
                    mean = F.interpolate(mean, size=target_shape, mode='bilinear', align_corners=False)
                if var.shape[2:] != target_shape:
                    var = F.interpolate(var, size=target_shape, mode='bilinear', align_corners=False)
                zt = mean + torch.sqrt(var + 1e-8) * torch.randn_like(mean)
                target = self.ou_ref.bridge_velocity(z0, z1, t)
            else:
                t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
                zt = (1 - t_reshaped) * z0 + t_reshaped * z1
                target = z1 - z0
            
            # Final verification of bridge output alignment
            if zt.shape[2:] != target_shape:
                 zt = F.interpolate(zt, size=target_shape, mode='bilinear', align_corners=False)
            if target.shape[2:] != target_shape:
                 target = F.interpolate(target, size=target_shape, mode='bilinear', align_corners=False)
            
            # Add noise to targets only (not to state) – scale from config
            if self.drift.training:
                t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
                noise_scale = config.DRIFT_TARGET_NOISE_SCALE * (1 - t_reshaped)
                target = target + torch.randn_like(target) * noise_scale
            
            # Classifier-Free Guidance: Randomly drop labels (set to NULL index) during training
            if self.drift.training and torch.rand(1).item() < config.LABEL_DROPOUT_PROB:
                train_labels = torch.full_like(labels, config.NUM_CLASSES - 1)
                train_text_bytes = None # No text for unconditional branch
            else:
                train_labels = labels
                train_text_bytes = text_bytes

            pred = self.drift(zt, t, train_labels, text_bytes=train_text_bytes, source_id=source_id)
            
            # Ensure prediction and target match perfectly for loss
            if pred.shape[2:] != target_shape:
                pred = F.interpolate(pred, size=target_shape, mode='bilinear', align_corners=False)
            if target.shape[2:] != target_shape:
                target = F.interpolate(target, size=target_shape, mode='bilinear', align_corners=False)

            # Time-weighted loss using config factor
            t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
            time_weights = 1.0 + config.TIME_WEIGHT_FACTOR * t_reshaped
            drift_loss_base = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * config.DRIFT_WEIGHT

            consistency_decay = max(0.1, 1.0 - (self.epoch - drift_start_epoch) / (config.EPOCHS - drift_start_epoch))
            
            # --- MONITORING METRICS FOR APP LAYER (Phase 2/3) ---
            with torch.no_grad():
                div_loss = self.vae._channel_diversity_loss(mu).item()
                # Use mu for structural checking in drift phase
                recon_for_ssim = self.vae.decode(mu, labels, text_bytes=text_bytes, source_id=source_id)
                ssim_val = self.ssim_loss(recon_for_ssim, images).item()

            # PHASE 3 ENHANCEMENT: Also train the VAE to reconstruct from the latent mean
            if phase == 3:
                # Use mu (the clean latent) for VAE stability in Phase 3
                # This ensures the decoder stays sharp without being confused by bridge noise
                recon_p3 = recon_for_ssim # already computed above
                
                # Scale recon loss down in Phase 3 so it doesn't overwhelm the Drift training
                p3_scale = getattr(config, 'PHASE3_RECON_SCALE', 0.1)
                recon_loss_p3 = F.l1_loss(recon_p3, images) * config.RECON_WEIGHT * p3_scale
                
                total_loss = drift_loss_base + (consistency_loss * config.CONSISTENCY_WEIGHT * consistency_decay) + recon_loss_p3
                loss_dict = {
                    'total': total_loss,
                    'drift': drift_loss_base.item(),
                    'consistency': consistency_loss.item(),
                    'recon_p3': recon_loss_p3.item(),
                    'diversity': div_loss,
                    'ssim_loss': ssim_val,
                    'mu_std': mu_global_std,
                    'z_std': z_global_std,
                    'temperature': temperature
                }
            else:
                total_loss = drift_loss_base + (consistency_loss * config.CONSISTENCY_WEIGHT * consistency_decay)
                loss_dict = {
                    'total': total_loss,
                    'drift': drift_loss_base.item(),
                    'consistency': consistency_loss.item(),
                    'diversity': div_loss,
                    'ssim_loss': ssim_val,
                    'mu_std': mu_global_std,
                    'z_std': z_global_std,
                    'temperature': temperature
                }
        
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
        """Calculate FID score on GPU using float64 for stability."""
        # Ensure correct shape [N, D]
        if real_features.dim() > 2:
            real_features = real_features.flatten(1)
        if fake_features.dim() > 2:
            fake_features = fake_features.flatten(1)

        # Stay on GPU but use high precision for the sensitive math
        f_real = real_features.to(torch.float64)
        f_fake = fake_features.to(torch.float64)

        mu1 = f_real.mean(dim=0)
        mu2 = f_fake.mean(dim=0)
        
        # torch.cov expects variables as rows, observations as columns [D, N]
        sigma1 = torch.cov(f_real.T)
        sigma2 = torch.cov(f_fake.T)

        diff = mu1 - mu2
        
        # Add regularization to prevent singularity
        reg = torch.eye(sigma1.shape[0], device=config.DEVICE, dtype=torch.float64) * 1e-6
        sigma1 += reg
        sigma2 += reg
        
        # Matrix square root via SVD: sqrt(sigma1 @ sigma2)
        # For FID, we need Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))
        # A more stable way on GPU:
        try:
            # We use the property that Tr(A + B - 2*sqrt(A@B)) is equivalent to 
            # the squared Fröbenius norm of (sqrt(A) - sqrt(B)) if they commute, 
            # but generally we use the standard formula.
            cov_prod = sigma1 @ sigma2
            
            # Use SVD for matrix square root on GPU
            u, s, v = torch.linalg.svd(cov_prod)
            covmean = u @ torch.diag(torch.sqrt(s)) @ v
            
            fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid.real.item())
        except Exception as e:
            config.logger.warning(f"GPU FID math failed (SVD): {e}")
            return 0.0

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
                
                # Heartbeat for Colab visibility
                if batch_idx % 10 == 0:
                    config.logger.info(f"💓 Epoch {current_epoch} ({mode}) | Batch {batch_idx}/{len(self.loader)}")
                
                # Forward pass with AMP if enabled (Cross-backend autocast)
                if self.scaler is not None:
                    device_type = config.DEVICE.type
                    # Standard CUDA/XPU autocast supports dtype argument
                    with torch.amp.autocast(device_type=device_type, enabled=config.USE_AMP, dtype=config.DTYPE_AMP):
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
                    self.opt_vae.zero_grad()
                    if self.scaler is not None:
                        self.scaler.scale(loss_dict['total']).backward()
                        self.scaler.unscale_(self.opt_vae)
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        self.scaler.step(self.opt_vae)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        self.opt_vae.step()
                    
                    if 'snr' in loss_dict:
                        snr_values.append(loss_dict['snr'])
                    if 'latent_std' in loss_dict:
                        latent_std_values.append(loss_dict['latent_std'])
                    if 'min_channel_std' in loss_dict:
                        channel_std_values.append(loss_dict['min_channel_std'])
                elif phase == 2:
                    self.opt_vae.zero_grad()
                    self.opt_drift.zero_grad()
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_dict['total']).backward()
                        self.scaler.unscale_(self.opt_vae)
                        self.scaler.unscale_(self.opt_drift)
                        # Clip both
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.scaler.step(self.opt_vae)
                        self.scaler.step(self.opt_drift)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.opt_vae.step()
                        self.opt_drift.step()
                elif phase == 3:
                    # In Phase 3, we update BOTH
                    self.opt_vae.zero_grad()
                    self.opt_drift.zero_grad()
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_dict['total']).backward()
                        self.scaler.unscale_(self.opt_vae)
                        self.scaler.unscale_(self.opt_drift)
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.scaler.step(self.opt_vae)
                        self.scaler.step(self.opt_drift)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR)
                        self.opt_vae.step()
                        self.opt_drift.step()
                
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
                import traceback
                config.logger.error(f"Error processing batch {batch_idx}: {e}")
                config.logger.error(traceback.format_exc())
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
                        source_id = batch.get('source_id', None)
                        if source_id is not None:
                            source_id = source_id[:fid_samples].to(config.DEVICE)
                        
                        # Correct VGG normalization
                        vgg_dtype = next(self.vgg.parameters()).dtype
                        real_norm = (((real_images + 1) / 2 - self.vgg_mean) / self.vgg_std).to(vgg_dtype)
                        real_feat = self.vgg(real_norm)

                        gen_images, _, _ = self.vae(real_images, real_labels, source_id)
                        gen_norm = (((gen_images + 1) / 2 - self.vgg_mean) / self.vgg_std).to(vgg_dtype)
                        gen_feat = self.vgg(gen_norm)
                        
                        # Apply Global Average Pooling to reduce dimension for stability and speed
                        # Shape change: [B, 256, 12, 12] -> [B, 256, 1, 1] -> [B, 256]
                        real_feat_pooled = F.adaptive_avg_pool2d(real_feat, (1, 1)).flatten(1)
                        gen_feat_pooled = F.adaptive_avg_pool2d(gen_feat, (1, 1)).flatten(1)
                        
                        fid_score = self.calculate_fid_batch(real_feat_pooled, gen_feat_pooled)
                        losses['fid'] = fid_score
                        config.logger.info(f"FID Score (approx): {fid_score:.2f}")
            except Exception as e:
                config.logger.warning(f"FID calculation failed: {e}")

        # Compute average losses
        if batch_count > 0:
            avg_losses = {}
            for key, value in losses.items():
                avg_losses[key] = value / batch_count

            # Use sum of individual average losses for total
            avg_losses['total'] = sum(v for k, v in avg_losses.items() if k != 'total')

            if snr_values:
                avg_losses['snr'] = np.mean(snr_values)
            if latent_std_values:
                avg_losses['latent_std'] = np.mean(latent_std_values)
            if channel_std_values:
                avg_losses['min_channel_std'] = np.mean(channel_std_values)
        else:
            avg_losses = {'total': 1e9}
            config.logger.error("No batches were successfully processed in this epoch!")

        # Final safety check against NaN/Inf in avg_losses
        if torch.tensor(avg_losses.get('total', 0)).isnan() or torch.tensor(avg_losses.get('total', 0)).isinf():
            config.logger.error("⚠️ Average total loss is NaN/Inf! Overriding with high value to prevent corrupted checkpoint.")
            avg_losses['total'] = 1e9

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
            config.logger.info(f"  SSIM loss: {avg_losses.get('ssim_loss', 0):.4f}")
            config.logger.info(f"  Edge loss: {avg_losses.get('edge_loss', 0):.4f}")
            config.logger.info(f"  TV loss: {avg_losses.get('tv_loss', 0):.4f}")
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
                    data_management.flexible_load(self.vae, snapshot['model_state'])
                    if 'optimizer_state' in snapshot:
                        self.opt_vae.load_state_dict(snapshot['optimizer_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'vae' and 'model_state' in snapshot:
                    data_management.flexible_load(self.vae, snapshot['model_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                else:
                    config.logger.warning("No VAE state found in snapshot")
            
            # Load Drift if requested and available
            if load_drift:
                if 'drift_state' in snapshot:
                    data_management.flexible_load(self.drift, snapshot['drift_state'])
                    if 'opt_drift_state' in snapshot:
                        self.opt_drift.load_state_dict(snapshot['opt_drift_state'])
                    config.logger.info(f"✅ Loaded Drift from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'drift' and 'drift_state' in snapshot:
                    data_management.flexible_load(self.drift, snapshot['drift_state'])
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
        
    def generate_reconstructions(self, batch: Optional[Dict] = None):
        """Save real images and their reconstructions to check VAE quality."""
        self.vae.eval()
        
        # Use provided batch or grab one from loader
        if batch is None:
            batch = next(iter(self.loader))
            
        images = batch['image'][:8].to(config.DEVICE)
        labels = batch['label'][:8].to(config.DEVICE)
        text_bytes = batch.get('text_bytes', None)
        if text_bytes is not None:
            text_bytes = text_bytes[:8].to(config.DEVICE)
        source_id = batch.get('source_id', None)
        if source_id is not None:
            source_id = source_id[:8].to(config.DEVICE)
            
        with torch.no_grad():
            recon, _, _ = self.vae(images, labels, text_bytes=text_bytes, source_id=source_id)
            
        # Combine into a single grid: top row real, bottom row recon
        combined = torch.cat([images, recon], dim=0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = config.DIRS["samples"] / f"recon_epoch{self.epoch+1}_{timestamp}.png"
        
        dm.save_image_grid(combined, save_path, nrow=8)
        config.logger.info(f"VAE reconstructions saved to {save_path}")

    def generate_samples(self, labels=None, num_samples=8, temperature=None, method='heun',
                         langevin_steps=None, langevin_step_size=None, langevin_score_scale=None,
                         cfg_scale=None, source_id=None):
        """
        Generate samples with label conditioning and optional context.

        Args:
            labels: List of class labels
            num_samples: Number of samples to generate
            temperature: Ignored (kept for API compatibility)
            method: 'euler', 'heun', or 'rk4'
            langevin_steps: Number of Langevin refinement steps (None = use config default)
            langevin_step_size: Step size for Langevin dynamics (default from config)
            langevin_score_scale: Scaling factor for the approximate score (default from config)
            cfg_scale: Scale for classifier-free guidance (None = use config default)
            source_id: Optional dataset source ID
        """
        if cfg_scale is None:
            cfg_scale = getattr(config, 'CFG_SCALE', 1.0)
        if langevin_steps is None:            langevin_steps = getattr(config, 'DEFAULT_LANGEVIN_STEPS', 0)
        if langevin_step_size is None:
            langevin_step_size = config.LANGEVIN_STEP_SIZE
        if langevin_score_scale is None:
            langevin_score_scale = config.LANGEVIN_SCORE_SCALE

        self.vae.eval()
        self.drift.eval()
        
        if labels is not None:
            num_samples = len(labels)
        else:
            labels = [i % 10 for i in range(num_samples)]
        
        with torch.no_grad():
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=config.DEVICE)
            
            # Neural Tokenizer support
            if config.USE_NEURAL_TOKENIZER:
                text_bytes_list = []
                for l in labels:
                    desc = dm.CLASS_DESCRIPTIONS[l] if l < 10 else f"class_{l}"
                    text_bytes_list.append(dm.text_to_bytes(desc))
                text_bytes_tensor = torch.tensor(text_bytes_list, device=config.DEVICE)
            else:
                text_bytes_tensor = None

            # Source ID context
            if source_id is None:
                s_id = torch.zeros(labels_tensor.shape[0], dtype=torch.long, device=config.DEVICE)
            else:
                s_id = torch.full((labels_tensor.shape[0],), source_id, dtype=torch.long, device=config.DEVICE)
                        
            # Start from pure noise using the prior standard deviation from config
            z = torch.randn(num_samples, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE) * config.CST_COEF_GAUSSIAN_PRIO

            config.logger.info(f"Initial z - min: {z.min():.3f}, max: {z.max():.3f}, mean: {z.mean():.3f}, std: {z.std():.3f}")
            
            steps = config.DEFAULT_STEPS
            dt = 1.0 / steps
            
            # ----- ODE integration -----
            for i in range(steps):
                t_cur = torch.full((num_samples, 1), i * dt, device=config.DEVICE)
                
                # Predict drift with context
                drift = self.drift(z, t_cur, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                
                # Monitor drift magnitude (adaptive clipping is inside the drift network)
                drift_norm = drift.flatten(start_dim=1).norm(p=2, dim=1).mean().item()

                if method == 'euler':
                    z = z + drift * dt
                elif method == 'heun':
                    k1 = drift
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_pred = z + dt * k1
                    k2 = self.drift(z_pred, t_next, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z = z + (dt / 2.0) * (k1 + k2)
                elif method == 'rk4':
                    k1 = drift
                    t_half = torch.full((num_samples, 1), (i + 0.5) * dt, device=config.DEVICE)
                    z_half = z + 0.5 * dt * k1
                    k2 = self.drift(z_half, t_half, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z_half2 = z + 0.5 * dt * k2
                    k3 = self.drift(z_half2, t_half, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_next = z + dt * k3
                    k4 = self.drift(z_next, t_next, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
                z = torch.clamp(z, -10, 10)   # gentle clamping
                
                if i % 10 == 0:
                    config.logger.info(f"Step {i:3d}, t={i*dt:.3f}, drift norm: {drift_norm:.4f}, z std: {z.std():.4f}")
                                
                if torch.isnan(z).any():
                    config.logger.error(f"NaN detected at step {i}!")
                    break
            
            # ----- Refined Langevin Refinement -----
            if langevin_steps > 0:
                config.logger.info(f"Starting {langevin_steps} Refined Langevin steps...")
                t_one = torch.full((num_samples, 1), 1.0, device=config.DEVICE)
                
                # Adaptive step size: start strong, end gentle
                base_lr = langevin_step_size
                
                for step in range(langevin_steps):
                    # 1. Compute Score Proxy 
                    # Using drift at t=1 is theoretically the 'terminal' velocity
                    drift_at_end = self.drift(z, t_one, labels_tensor, text_bytes=text_bytes_tensor, source_id=s_id)
                    
                    # 2. Add Annealed Noise
                    # We decay the noise scale as we converge to the manifold
                    step_ratio = step / langevin_steps
                    current_noise_scale = np.sqrt(2 * base_lr) * (1 - 0.5 * step_ratio)
                    noise = torch.randn_like(z) * current_noise_scale
                    
                    # 3. Stochastic Gradient Update (MALA style)
                    # We treat the drift as the gradient of the log-probability
                    z = z.detach() + 0.5 * base_lr * langevin_score_scale * drift_at_end + noise
                    
                    # 4. Gentle Manifold Constraint
                    # Prevents latents from escaping the range expected by the VAE decoder
                    z = torch.clamp(z, -config.DIVERSITY_MAX_STD * 2, config.DIVERSITY_MAX_STD * 2)
                    
                    if (step + 1) % 5 == 0:
                        config.logger.info(f" Langevin Step {step+1}: z_std={z.std():.4f}")

                z = z.detach() # Final cleanup
            config.logger.info(f"Refinement complete: Final z_std={z.std():.4f}")
            
            # Decode
            self.vae.set_force_active(True)
            images = self.vae.decode(z, labels_tensor, text_bytes=text_bytes_tensor, source_id=s_id)
            self.vae.set_force_active(False)
            images = torch.clamp(images, -1, 1)
            
            config.logger.info(f"Generated images - min: {images.min():.3f}, max: {images.max():.3f}, mean: {images.mean():.3f}")
            
            # Save images
            images_display = (images + 1) / 2
            images_display = torch.clamp(images_display, 0, 1)
            
            grid = vutils.make_grid(images_display, nrow=4, padding=2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            grid_path = config.DIRS["samples"] / f"gen_epoch{self.epoch+1}_{timestamp}.png"
            vutils.save_image(grid, grid_path)
            
            for idx, img in enumerate(images_display):
                individual_path = config.DIRS["samples"] / f"gen_{idx}_label{labels[idx]}_epoch{self.epoch+1}.png"
                vutils.save_image(img, individual_path)
            
            debug_path = config.DIRS["samples"] / f"raw_epoch{self.epoch+1}_{timestamp}.pt"
            torch.save({
                'z': z.cpu(),
                'images': images.cpu(),
                'labels': labels
            }, debug_path)
            
            config.logger.info(f"Generated {num_samples} samples for labels {labels}")
            config.logger.info(f"Images saved to: {grid_path}")
            
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
        
        # Helper to bake spectral norm into weights
        def bake_spectral_norm(model):
            import copy
            from torch.nn.utils import remove_spectral_norm
            model_copy = copy.deepcopy(model)
            for m in model_copy.modules():
                try:
                    remove_spectral_norm(m)
                except (ValueError, AttributeError):
                    pass
            return model_copy

        # Wrapper class to export ONLY the decoder (generator) part of the VAE
        class VAEGenerator(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, z, labels):
                return self.vae.decode(z, labels, None)

        try:
            self.vae.eval()
            self.drift.eval()
            # Set internal export flags
            self.vae.apply(lambda m: set_export_mode(m, True))
            self.drift.apply(lambda m: set_export_mode(m, True))
            
            vae_for_export = bake_spectral_norm(self.vae)
            drift_for_export = bake_spectral_norm(self.drift)

            # --- Export Generator ---
            dummy_z = torch.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE)
            dummy_label = torch.tensor([0], device=config.DEVICE, dtype=torch.long)
            
            gen_path = config.DIRS["onnx"] / "generator.onnx"
            vae_gen = VAEGenerator(vae_for_export)
            
            config.logger.info("Exporting Generator (Simplified)...")
            with torch.no_grad():
                torch.onnx.export(
                    vae_gen,
                    (dummy_z, dummy_label),
                    str(gen_path),
                    export_params=True,
                    opset_version=15,
                    do_constant_folding=True,
                    input_names=['z', 'label'],
                    output_names=['reconstruction'],
                    dynamic_axes={
                        'z': {0: 'batch_size'},
                        'label': {0: 'batch_size'},
                        'reconstruction': {0: 'batch_size'}
                    }
                )
            
            # --- Export Drift ---
            dummy_t = torch.tensor([[0.5]], device=config.DEVICE)
            drift_path = config.DIRS["onnx"] / "drift.onnx"
            
            config.logger.info("Exporting Drift (Simplified)...")
            # Create a simple wrapper for drift too to strip source_id
            class DriftWrapper(torch.nn.Module):
                def __init__(self, drift):
                    super().__init__()
                    self.drift = drift
                def forward(self, z, t, label):
                    return self.drift(z, t, label, None, None)

            drift_gen = DriftWrapper(drift_for_export)

            with torch.no_grad():
                torch.onnx.export(
                    drift_gen,
                    (dummy_z, dummy_t, dummy_label),
                    str(drift_path),
                    export_params=True,
                    opset_version=15, 
                    do_constant_folding=True,
                    input_names=['z', 't', 'label'],
                    output_names=['drift'],
                    dynamic_axes={
                        'z': {0: 'batch_size'},
                        't': {0: 'batch_size'},
                        'label': {0: 'batch_size'},
                        'drift': {0: 'batch_size'}
                    }
                )
            
            config.logger.info(f"✅ ONNX Export Successful: {gen_path}, {drift_path}")

            # --- Auto-configure the HTML file to match the current dimensions ---
            html_path = Path("onnx_generate_image.html")
            if html_path.exists():
                try:
                    import re
                    with open(html_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Correct Latent Shape
                    new_latent = f"[1, {config.LATENT_CHANNELS}, {config.LATENT_H}, {config.LATENT_W}]"
                    content = re.sub(r'LATENT_SHAPE\s*=\s*\[1,\s*\d+,\s*\d+,\s*\d+\]', f'LATENT_SHAPE = {new_latent}', content)
                    
                    # Correct Image Size
                    content = re.sub(r'(let|const)\s+IMG_SIZE\s*=\s*\d+', f'let IMG_SIZE = {config.IMG_SIZE}', content)
                    
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    config.logger.info(f"Updated {html_path.name} with {config.LATENT_H}x{config.LATENT_W} dimensions.")
                except Exception as e:
                    config.logger.warning(f"Could not auto-update HTML: {e}")

        except Exception as e:
            config.logger.error(f"ONNX export failed: {e}")
        finally:
            # Revert internal flags
            self.vae.apply(lambda m: set_export_mode(m, False))
            self.drift.apply(lambda m: set_export_mode(m, False))



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
        elif (epoch + 1) % config.CHECKPOINT_INTERVAL == 0:
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
