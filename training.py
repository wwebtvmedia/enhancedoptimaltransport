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
    """Main trainer class for label-conditioned Schrödinger Bridge."""
    
    def __init__(self, loader):
        self.loader = loader

        self.vae = models.LabelConditionedVAE().to(config.DEVICE)
        self.drift = models.LabelConditionedDrift().to(config.DEVICE)
        self.opt_vae = optim.AdamW(self.vae.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

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
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.vgg_std =torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
                
                # ImageNet normalization constants
                # Ensure mean/std are on the same device as inputs
                mean = self.vgg_mean.to(recon.device)
                std = self.vgg_std.to(recon.device)
            except:
                return torch.tensor(0.0, device=config.DEVICE)
    
         # Move mean/std to the device of the input (handles CPU/GPU automatically)
        mean = self.vgg_mean.to(recon.device)
        std = self.vgg_std.to(recon.device)

        # 1. Map from [-1, 1] to [0, 1]
        recon_01 = (recon + 1) / 2
        target_01 = (target + 1) / 2
        
        # 2. Apply ImageNet normalization
        recon_norm = (recon_01 - mean) / std
        target_norm = (target_01 - mean) / std
        
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
            self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
            self.vae_ref.load_state_dict(self.vae.state_dict())
            self.vae_ref.eval()
            for param in self.vae_ref.parameters():
                param.requires_grad = False

            # Unfreeze encoder parts only, explicitly zero-out grad for frozen params
            unfrozen_count = 0
            for name, param in self.vae.named_parameters():
                if any(k in name for k in ['enc_', 'label_emb', 'z_mean', 'z_logvar']):
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
            # Store the epoch when Phase 2 started
            self.phase2_start_epoch = self.epoch

        elif new_phase == 3:
            # Phase 3: Both trainable – unfreeze decoder as well
            for name, param in self.vae.named_parameters():
                param.requires_grad = True
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

    def ssim_loss(self,x, y):

        C1 = (0.01 * 1.0)**2 
        C2 = (0.03 * 1.0)**2

        mu_x, mu_y = x.mean([-2,-1], keepdim=True), y.mean([-2,-1], keepdim=True)
        var_x = ((x - mu_x)**2).mean([-2,-1], keepdim=True)
        var_y = ((y - mu_y)**2).mean([-2,-1], keepdim=True)
        cov_xy = ((x - mu_x)*(y - mu_y)).mean([-2,-1], keepdim=True)

        ssim = (2*mu_x*mu_y + C1)*(2*cov_xy + C2) / ((mu_x**2 + mu_y**2 + C1)*(var_x + var_y + C2))

        return 1 - ssim.mean()

    def compute_loss(self, batch: Dict, phase: int = 1) -> Dict:
        """Compute loss for current batch based on training phase."""
        # Extract images and labels
        if isinstance(batch, dict):
            images = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            images = batch[0].to(config.DEVICE)
            labels = batch[1].to(config.DEVICE)
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        # Debug logging
        if self.debug_counter % self.debug_interval == 0:
            config.logger.info(f"\n=== DEBUG Epoch {self.epoch+1}, Batch {self.debug_counter} ===")
            config.logger.info(f"Images - min: {images.min().item():.4f}, max: {images.max().item():.4f}, mean: {images.mean().item():.4f}")

        if phase == 1:
            # Phase 1: Train VAE
            recon, mu, logvar = self.vae(images, labels)

            # Periodically diagnose and correct latent collapse
            if self.debug_counter % self.debug_interval == 0:
                self.diagnose_latent_collapse(mu, logvar, self.epoch)

            # Compute metrics
            latent_std = torch.exp(0.5 * logvar).mean().item()
            channel_stds = mu.std(dim=[0, 2, 3]).detach().cpu().numpy()
            min_channel_std = channel_stds.min()
            
            if self.debug_counter % self.debug_interval == 0:
                config.logger.info(f"Mu std: {mu.std().item():.3f}")
                config.logger.info(f"Channel stds: {channel_stds}")
            
            # Adaptive KL weight based on channel usage
            raw_mse = F.mse_loss(recon, images)
            raw_kl = kl_divergence_spatial(mu, logvar)
            
            if latent_std < 0.3:
                current_kl_weight = config.KL_WEIGHT * 10.0
            elif min_channel_std < 0.01:
                current_kl_weight = config.KL_WEIGHT * 5.0
            else:
                current_kl_weight = config.KL_WEIGHT
            
            diversity_loss = self.vae.diversity_loss if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)

            kl_annealing = min(1.0, self.epoch / config.KL_ANNEALING_EPOCHS)
            # Add a minimum KL floor to prevent collapse
            min_kl_per_channel = 0.1  # bits per channel
            kl_per_channel = raw_kl / (config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W)
            if kl_per_channel < min_kl_per_channel and self.epoch > 5:
                # Boost KL if it's too low
                kl_multiplier = 2.0
                config.logger.debug(f"  KL too low ({kl_per_channel:.3f} < {min_kl_per_channel}), boosting")
            else:
                kl_multiplier = 1.0
                
            kl_loss = raw_kl * current_kl_weight * kl_annealing * kl_multiplier

            recon_loss = raw_mse * config.RECON_WEIGHT + config.SIM_LOST_FACTOR * self.ssim_loss(recon, images) + config.PERSP_LOST_FACTOR * self.perceptual_loss(recon, images) * 0.2
            total_loss = recon_loss + kl_loss + diversity_loss * config.DIVERSITY_WEIGHT
            
            snr = calc_snr(images, recon)
            self.debug_counter += 1
            
            loss_dict = {
                'total': total_loss,
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'diversity': diversity_loss.item(),
                'snr': snr,
                'raw_mse': raw_mse.item(),
                'raw_kl': raw_kl.item(),
                'latent_std': latent_std,
                'min_channel_std': min_channel_std,
                'max_channel_std': channel_stds.max(),
            }
            
            # Log gradient magnitude every 10 epochs for sharpness monitoring
            if self.epoch % 10 == 0 and phase == 1:
                with torch.no_grad():
                    # Check image sharpness via gradients
                    grad_x = torch.abs(recon[:, :, :, 1:] - recon[:, :, :, :-1]).mean().item()
                    grad_y = torch.abs(recon[:, :, 1:, :] - recon[:, :, :-1, :]).mean().item()
                    config.logger.info(f"Image gradient magnitude - X: {grad_x:.4f}, Y: {grad_y:.4f}")
                    
                    # Check latent statistics
                    latent_std_channel = mu.std(dim=[0, 2, 3])
                    config.logger.info(f"Channel utilization - min: {latent_std_channel.min():.4f}, max: {latent_std_channel.max():.4f}")

        else:  # Drift training (phase 2 or 3)
            # Get the epoch when drift training started
            drift_start_epoch = getattr(self, 'phase2_start_epoch', config.TRAINING_SCHEDULE.get('switch_epoch_1', config.TRAINING_SCHEDULE.get('switch_epoch', 50)))
            with torch.no_grad():
                mu_ref, _ = self.vae_ref.encode(images, labels)   # always use frozen anchor
                mu, logvar = self.vae.encode(images, labels)
                std_from_logvar = torch.exp(0.5 * logvar).mean().item()
                consistency_loss = F.mse_loss(mu, mu_ref)
                mu_global_std = mu.std().item()

                # Temperature annealing using config values
                temperature = config.TEMPERATURE_START + (config.TEMPERATURE_END - config.TEMPERATURE_START) * (self.epoch / config.EPOCHS)
                z1 = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * temperature
                z_global_std = z1.std().item()
            
            # Sample time with beta distribution after sufficient training
            if self.epoch > config.TRAINING_SCHEDULE['switch_epoch'] + config.EPOCHS // 6:  # After 1/6 of drift training
                beta_dist = torch.distributions.Beta(2, 2)
                t = beta_dist.sample((images.shape[0], 1)).to(config.DEVICE)
            else:
                t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            
            # Start from noise with std=0.8 to match training
            z0 = torch.randn_like(z1) * 0.8
            
            # Sample intermediate latent using either linear interpolation or OU bridge
            if config.USE_OU_BRIDGE and self.ou_ref is not None:
                mean, var = self.ou_ref.bridge_sample(z0, z1, t)
                zt = mean + torch.sqrt(var + 1e-8) * torch.randn_like(mean)
                target = z1 - z0
            else:
                zt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1
                target = z1 - z0
            
            # Add noise to targets only (not to state) – scale from config
            if self.drift.training:
                noise_scale = config.DRIFT_TARGET_NOISE_SCALE * (1 - t.view(-1, 1, 1, 1))
                target = target + torch.randn_like(target) * noise_scale
            
            pred = self.drift(zt, t, labels)
            
            # Time-weighted loss using config factor
            time_weights = 1.0 + config.TIME_WEIGHT_FACTOR * t.view(-1, 1, 1, 1)
            drift_loss_base = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * config.DRIFT_WEIGHT

            consistency_decay = max(0.1, 1.0 - (self.epoch - drift_start_epoch) / (config.EPOCHS - drift_start_epoch))
            # In Phase 3, we might want a different decay schedule; for now keep same.
            total_loss = drift_loss_base + (consistency_loss * config.CONSISTENCY_WEIGHT * consistency_decay)
            
            loss_dict = {
                'total': total_loss,
                'drift': drift_loss_base.item(),
                'consistency': consistency_loss.item(),
                'temperature': temperature
            }
        
        # Add composite score
        loss_dict['composite_score'] = composite_score(loss_dict, phase)
        return loss_dict

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
                if self.scaler is not None and phase in (2,3):
                    with torch.cuda.amp.autocast():
                        loss_dict = self.compute_loss(batch_dict, phase=phase)
                else:
                    loss_dict = self.compute_loss(batch_dict, phase=phase)
                
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
                    loss_dict['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                    self.opt_vae.step()
                    
                    if self.snapshot_vae:
                        self.snapshot_vae.step()
                    
                    if 'snr' in loss_dict:
                        snr_values.append(loss_dict['snr'])
                    if 'latent_std' in loss_dict:
                        latent_std_values.append(loss_dict['latent_std'])
                    if 'min_channel_std' in loss_dict:
                        channel_std_values.append(loss_dict['min_channel_std'])
                else:
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
                    
                    if self.snapshot_drift:
                        self.snapshot_drift.step()
                
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
        
    def generate_samples(self, labels=None, num_samples=8, temperature=None, method='heun',
                        langevin_steps=0, langevin_step_size=None, langevin_score_scale=None):
        """
        Generate samples with label conditioning.
        
        Args:
            labels: List of class labels
            num_samples: Number of samples to generate
            temperature: Ignored (kept for API compatibility)
            method: 'euler', 'heun', or 'rk4'
            langevin_steps: Number of Langevin refinement steps (0 = disable)
            langevin_step_size: Step size for Langevin dynamics (default from config)
            langevin_score_scale: Scaling factor for the approximate score (default from config)
        """
        if langevin_step_size is None:
            langevin_step_size = config.LANGEVIN_STEP_SIZE
        if langevin_score_scale is None:
            langevin_score_scale = config.LANGEVIN_SCORE_SCALE

        self.vae.eval()
        self.drift.eval()
        
        if labels is None:
            labels = [i % 10 for i in range(num_samples)]
        
        with torch.no_grad():
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=config.DEVICE)
                        
            try:
                # Try to get a batch from the loader to estimate latent distribution
                sample_batch = next(iter(self.loader))
                if isinstance(sample_batch, dict):
                    images = sample_batch['image'][:min(16, num_samples)].to(config.DEVICE)
                    bash_labels = sample_batch['label'][:min(16, num_samples)].to(config.DEVICE)
                else:
                    images = sample_batch[0][:min(16, num_samples)].to(config.DEVICE)
                    bash_labels = sample_batch[1][:min(16, num_samples)].to(config.DEVICE)
                
                with torch.no_grad():
                    mu, logvar = self.vae.encode(images, bash_labels)
                    z_init = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * 0.3
                    
                    # If we need more samples than we have, repeat
                    if num_samples > z_init.shape[0]:
                        repeats = (num_samples + z_init.shape[0] - 1) // z_init.shape[0]
                        z = z_init.repeat(repeats, 1, 1, 1)[:num_samples]
                    else:
                        z = z_init[:num_samples]
            except Exception as e:
                config.logger.warning(f"Could not estimate latent distribution, using random noise: {e}")
                z = torch.randn(num_samples, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE) * 0.5



            config.logger.info(f"Initial z - min: {z.min():.3f}, max: {z.max():.3f}, mean: {z.mean():.3f}, std: {z.std():.3f}")
            
            steps = config.DEFAULT_STEPS
            dt = 1.0 / steps
            
            # ----- ODE integration -----
            for i in range(steps):
                t_cur = torch.full((num_samples, 1), i * dt, device=config.DEVICE)
                used_for_norm = None

                if method == 'euler':
                    drift = self.drift(z, t_cur, labels_tensor)
                    z = z + drift * dt
                    used_for_norm = drift
                elif method == 'heun':
                    k1 = self.drift(z, t_cur, labels_tensor)
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_pred = z + dt * k1
                    k2 = self.drift(z_pred, t_next, labels_tensor)
                    z = z + (dt / 2.0) * (k1 + k2)
                    used_for_norm = k1
                elif method == 'rk4':
                    k1 = self.drift(z, t_cur, labels_tensor)
                    t_half = torch.full((num_samples, 1), (i + 0.5) * dt, device=config.DEVICE)
                    z_half = z + 0.5 * dt * k1
                    k2 = self.drift(z_half, t_half, labels_tensor)
                    z_half2 = z + 0.5 * dt * k2
                    k3 = self.drift(z_half2, t_half, labels_tensor)
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_next = z + dt * k3
                    k4 = self.drift(z_next, t_next, labels_tensor)
                    z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    used_for_norm = k1
                else:
                    raise ValueError(f"Unknown method: {method}")
    
                z = torch.clamp(z, -10, 10)   # gentle clamping
                
                if i % 10 == 0:
                    drift_norm = used_for_norm.flatten(start_dim=1).norm(p=2, dim=1).mean().item()
                    config.logger.info(f"Step {i:3d}, t={i*dt:.3f}, drift norm: {drift_norm:.4f}, z std: {z.std():.4f}")
                                
                if torch.isnan(z).any():
                    config.logger.error(f"NaN detected at step {i}!")
                    break
            
            # ----- Langevin refinement (optional) -----
            if langevin_steps > 0:
                config.logger.info(f"Starting {langevin_steps} Langevin refinement steps...")
                t_one = torch.full((num_samples, 1), 1.0, device=config.DEVICE)
                for step in range(langevin_steps):
                    noise = torch.randn_like(z) * np.sqrt(2 * langevin_step_size)
                    # Use drift at t=1 as an approximate score direction
                    drift_at_end = self.drift(z, t_one, labels_tensor)
                    z = z + 0.5 * langevin_step_size * langevin_score_scale * drift_at_end + noise
                    z = torch.clamp(z, -10, 10)
                    config.logger.info(f"Langevin step {step+1}: z std = {z.std():.4f}")
            
            config.logger.info(f"Final z - mean: {z.mean():.3f}, std: {z.std():.3f}")
            
            # Decode
            images = self.vae.decode(z, labels_tensor)
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
        
        try:
            # --- Set export mode for VAE and Drift ---
            self.vae.apply(lambda m: set_export_mode(m, True))
            self.drift.apply(lambda m: set_export_mode(m, True))
            
            # Export VAE
            dummy_img = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=config.DEVICE)
            dummy_label = torch.tensor([0], device=config.DEVICE)
            
            vae_path = config.DIRS["onnx"] / "vae.onnx"
            torch.onnx.export(
                self.vae,
                (dummy_img, dummy_label),
                vae_path,
                input_names=['image', 'label'],
                output_names=['reconstruction', 'mu', 'logvar'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'label': {0: 'batch_size'},
                    'reconstruction': {0: 'batch_size'}
                },
                opset_version=11
            )
            config.logger.info(f"VAE exported to {vae_path}")
            
            # Export Drift
            dummy_z = torch.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE)
            dummy_t = torch.tensor([[0.5]], device=config.DEVICE)
            
            drift_path = config.DIRS["onnx"] / "drift.onnx"
            torch.onnx.export(
                self.drift,
                (dummy_z, dummy_t, dummy_label),
                drift_path,
                input_names=['z', 't', 'label'],
                output_names=['drift'],
                dynamic_axes={
                    'z': {0: 'batch_size'},
                    'label': {0: 'batch_size'},
                    'drift': {0: 'batch_size'}
                },
                opset_version=11
            )
            config.logger.info(f"Drift exported to {drift_path}")
            
            # --- Reset export mode (optional) ---
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