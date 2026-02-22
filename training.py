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
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from typing import Optional, List, Dict, Union, Tuple, Any

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
# USE CENTRALIZED CONFIGURATION
# ============================================================
IMG_SIZE = config.IMG_SIZE
LATENT_CHANNELS = config.LATENT_CHANNELS
LATENT_H = config.LATENT_H
LATENT_W = config.LATENT_W
LATENT_DIM = config.LATENT_DIM

LR = config.LR
EPOCHS = config.EPOCHS
WEIGHT_DECAY = config.WEIGHT_DECAY
GRAD_CLIP = config.GRAD_CLIP

NUM_CLASSES = config.NUM_CLASSES
LABEL_EMB_DIM = config.LABEL_EMB_DIM

# Loss weights
KL_WEIGHT = config.KL_WEIGHT
RECON_WEIGHT = config.RECON_WEIGHT
DRIFT_WEIGHT = config.DRIFT_WEIGHT
DIVERSITY_WEIGHT = config.DIVERSITY_WEIGHT
CONSISTENCY_WEIGHT = config.CONSISTENCY_WEIGHT

# Feature toggles
USE_PERCENTILE = config.USE_PERCENTILE
USE_SNAPSHOTS = config.USE_SNAPSHOTS
USE_KPI_TRACKING = config.USE_KPI_TRACKING
USE_OU_BRIDGE = config.USE_OU_BRIDGE
USE_AMP = config.USE_AMP

# Inference parameters
INFERENCE_TEMPERATURE = config.INFERENCE_TEMPERATURE
DEFAULT_STEPS = config.DEFAULT_STEPS
DEFAULT_SEED = config.DEFAULT_SEED

# KPI tracking
KPI_WINDOW_SIZE = config.KPI_WINDOW_SIZE
EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
TARGET_SNR = config.TARGET_SNR
REVERT_THRESHOLD = config.REVERT_THRESHOLD

# OU reference
OU_THETA = config.OU_THETA
OU_SIGMA = config.OU_SIGMA

# Training schedule
SWITCH_EPOCH = config.SWITCH_EPOCH
TRAINING_SCHEDULE = config.TRAINING_SCHEDULE

# Paths
DIRS = config.DIRS
logger = config.logger

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

def kl_divergence_spatial(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence for spatial latent variables."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.mean(torch.sum(kl, dim=[1, 2, 3]))

def composite_score(loss_dict: Dict, phase: int) -> float:
    """Compute a composite score for model selection."""
    score = 0.0
    if phase == 1:
        # VAE phase: higher SNR, lower KL, higher diversity are good
        if 'snr' in loss_dict:
            score += loss_dict['snr'] / TARGET_SNR
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
    mode = TRAINING_SCHEDULE['mode']
    
    if mode == 'manual':
        phase = TRAINING_SCHEDULE['force_phase']
        return 1 if phase is None else phase
    elif mode == 'custom':
        return TRAINING_SCHEDULE['custom_schedule'].get(epoch, 1)
    elif mode == 'alternate':
        alternate_freq = TRAINING_SCHEDULE.get('alternate_freq', 5)
        return 1 if (epoch // alternate_freq) % 2 == 0 else 2
    else:  # 'auto' mode
        return 1 if epoch < TRAINING_SCHEDULE['switch_epoch'] else 2

def configure_training_schedule(
    mode: str = 'auto', 
    vae_epochs: Optional[List[int]] = None, 
    drift_epochs: Optional[List[int]] = None, 
    switch_epoch: int = 50,
    alternate_freq: int = 5,
    custom_schedule: Optional[Dict] = None
) -> Dict:
    """Configure the global training schedule."""
    if mode == 'vae_only':
        TRAINING_SCHEDULE['mode'] = 'manual'
        TRAINING_SCHEDULE['force_phase'] = 1
    elif mode == 'drift_only':
        TRAINING_SCHEDULE['mode'] = 'manual'
        TRAINING_SCHEDULE['force_phase'] = 2
    elif mode == 'auto':
        TRAINING_SCHEDULE['mode'] = 'auto'
        TRAINING_SCHEDULE['switch_epoch'] = switch_epoch
    elif mode == 'alternate':
        TRAINING_SCHEDULE['mode'] = 'alternate'
        TRAINING_SCHEDULE['alternate_freq'] = alternate_freq
    elif mode == 'custom':
        TRAINING_SCHEDULE['mode'] = 'custom'
        TRAINING_SCHEDULE['custom_schedule'] = custom_schedule or {}
    elif mode == 'manual':
        TRAINING_SCHEDULE['mode'] = 'manual'
    
    if vae_epochs is not None:
        TRAINING_SCHEDULE['vae_epochs'] = vae_epochs
    if drift_epochs is not None:
        TRAINING_SCHEDULE['drift_epochs'] = drift_epochs
    
    logger.info(f"Training schedule configured: mode={TRAINING_SCHEDULE['mode']}")
    return TRAINING_SCHEDULE

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
        
    def should_stop(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 1e-4, phase: int = 1) -> bool:
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
                logger.info(f"Early stopping triggered: loss increased for {patience} epochs")
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
        
        self.opt_vae = optim.AdamW(self.vae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.opt_drift = optim.AdamW(self.drift.parameters(), lr=LR * 2, weight_decay=WEIGHT_DECAY)
        
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_vae, T_max=EPOCHS, eta_min=LR*0.01
        )
        self.scheduler_drift = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_drift, T_max=EPOCHS, eta_min=LR*0.01
        )
        
        self.kpi_tracker = KPITracker(window_size=KPI_WINDOW_SIZE)
        
        self.snapshot_vae = dm.SnapshotManager(self.vae, self.opt_vae, name="vae") if USE_SNAPSHOTS else None
        self.snapshot_drift = dm.SnapshotManager(self.drift, self.opt_drift, name="drift") if USE_SNAPSHOTS else None
        
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
        self.ou_ref = OUReference(theta=OU_THETA, sigma=OU_SIGMA) if USE_OU_BRIDGE else None
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if USE_AMP and config.DEVICE.type == 'cuda' else None
        
        logger.info(f"Models initialized:")
        logger.info(f"  VAE params: {sum(p.numel() for p in self.vae.parameters()):,}")
        logger.info(f"  Drift params: {sum(p.numel() for p in self.drift.parameters()):,}")
        if USE_OU_BRIDGE:
            logger.info(f"  Using OU bridge reference (theta={OU_THETA})")

    def get_training_phase(self, epoch: int) -> int:
        """Get training phase for current epoch."""
        phase = set_training_phase(epoch)
        if phase != self.phase:
            logger.info(f"🔄 Phase changed from {self.phase} to {phase} at epoch {epoch+1}")
            self.phase = phase
            if phase == 2 and self.vae_ref is None:
                self._handle_phase_transition()
        return phase

    def _handle_phase_transition(self) -> None:
        """Handle transition from Phase 1 to Phase 2."""
        self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
        self.vae_ref.load_state_dict(self.vae.state_dict())
        self.vae_ref.eval()
        for param in self.vae_ref.parameters():
            param.requires_grad = False
        
        # Freeze most VAE layers, keep encoder trainable
        unfrozen_count = 0
        for name, param in self.vae.named_parameters():
            if any(k in name for k in ['enc_', 'label_emb', 'z_mean', 'z_logvar']):
                param.requires_grad = True
                unfrozen_count += param.numel()
            else:
                param.requires_grad = False
        
        logger.info(f"PHASE 2: Unfrozen {unfrozen_count:,} params. Consistency Anchor Set.")
        
        # Recreate optimizer for VAE with frozen layers
        self.opt_vae = optim.AdamW(
            filter(lambda p: p.requires_grad, self.vae.parameters()),
            lr=LR * 0.1,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_vae,
            T_max=EPOCHS - self.epoch,
            eta_min=LR * 0.005
        )

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
            logger.info(f"\n=== DEBUG Epoch {self.epoch+1}, Batch {self.debug_counter} ===")
            logger.info(f"Images - min: {images.min().item():.4f}, max: {images.max().item():.4f}, mean: {images.mean().item():.4f}")

        if phase == 1:
            # Phase 1: Train VAE
            recon, mu, logvar = self.vae(images, labels)
            
            # Compute metrics
            latent_std = torch.exp(0.5 * logvar).mean().item()
            channel_stds = mu.std(dim=[0, 2, 3]).detach().cpu().numpy()
            min_channel_std = channel_stds.min()
            
            if self.debug_counter % self.debug_interval == 0:
                logger.info(f"Mu std: {mu.std().item():.3f}")
                logger.info(f"Channel stds: {channel_stds}")
            
            # Adaptive KL weight based on channel usage
            raw_mse = F.mse_loss(recon, images)
            raw_kl = kl_divergence_spatial(mu, logvar)
            
            if latent_std < 0.3:
                current_kl_weight = KL_WEIGHT * 10.0
            elif min_channel_std < 0.01:
                current_kl_weight = KL_WEIGHT * 5.0
            else:
                current_kl_weight = KL_WEIGHT
            
            diversity_loss = self.vae.diversity_loss if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)
            kl_annealing = min(1.0, self.epoch / 10.0)
            
            kl_loss = raw_kl * current_kl_weight * kl_annealing
            recon_loss = raw_mse * RECON_WEIGHT
            total_loss = recon_loss + kl_loss + diversity_loss * DIVERSITY_WEIGHT
            
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
        else:
            # Phase 2: Train Drift with consistency
            with torch.no_grad():
                mu_ref, _ = self.vae_ref.encode(images, labels)
                mu, logvar = self.vae.encode(images, labels)
                consistency_loss = F.mse_loss(mu, mu_ref)
                
                # Temperature annealing based on global EPOCHS
                temperature = 0.3 + 0.7 * (1 - self.epoch / EPOCHS)
                z1 = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * temperature
            
            # Sample time with beta distribution after sufficient training
            if self.epoch > SWITCH_EPOCH + EPOCHS // 6:  # After 1/6 of drift training
                beta_dist = torch.distributions.Beta(2, 2)
                t = beta_dist.sample((images.shape[0], 1)).to(config.DEVICE)
            else:
                t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            
            # Start from noise with std=0.8 to match training
            z0 = torch.randn_like(z1) * 0.8
            
            # Sample intermediate latent using either linear interpolation or OU bridge
            if USE_OU_BRIDGE and self.ou_ref is not None:
                mean, var = self.ou_ref.bridge_sample(z0, z1, t)
                zt = mean + torch.sqrt(var + 1e-8) * torch.randn_like(mean)
                target = z1 - z0
            else:
                zt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1
                target = z1 - z0
            
            # Add noise to targets only (not to state)
            if self.drift.training:
                noise_scale = 0.01 * (1 - t.view(-1, 1, 1, 1))
                target = target + torch.randn_like(target) * noise_scale
            
            pred = self.drift(zt, t, labels)
            
            # Time-weighted loss
            time_weights = 1.0 + 2.0 * t.view(-1, 1, 1, 1)
            drift_loss_base = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * DRIFT_WEIGHT
            
            total_loss = drift_loss_base + (consistency_loss * CONSISTENCY_WEIGHT)
            
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
            self.vae.eval()
            self.drift.train()
        
        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(self.loader, desc=f"Epoch {current_epoch} ({mode})")
        else:
            pbar = self.loader
            logger.info(f"Epoch {current_epoch} ({mode})")
        
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
                if self.scaler is not None and phase == 2:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.compute_loss(batch_dict, phase=phase)
                else:
                    loss_dict = self.compute_loss(batch_dict, phase=phase)
                
                # Check for NaN/Inf
                if not isinstance(loss_dict, dict) or 'total' not in loss_dict:
                    continue
                
                if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
                    logger.error(f" NaN/Inf detected!")
                    reverted = False
                    if phase == 1 and self.snapshot_vae:
                        reverted = self.snapshot_vae.revert()
                    elif phase == 2 and self.snapshot_drift:
                        reverted = self.snapshot_drift.revert()
                    if reverted:
                        continue
                    else:
                        break
                
                # Debug logging
                if batch_idx % 50 == 0 and batch_idx > 0:
                    self._debug_training_loop(batch_dict, loss_dict)
                
                # Backward pass
                if phase == 1:
                    self.opt_vae.zero_grad()
                    loss_dict['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), GRAD_CLIP)
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
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), GRAD_CLIP * 0.5)
                        self.scaler.step(self.opt_drift)
                        self.scaler.update()
                    else:
                        loss_dict['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), GRAD_CLIP * 0.5)
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
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Update schedulers
        if phase == 1:
            self.scheduler_vae.step()
        else:
            self.scheduler_drift.step()
        
        # Compute average losses
        if batch_count > 0:
            avg_losses = {}
            for key, value in losses.items():
                avg_losses[key] = value / batch_count
            
            if snr_values:
                avg_losses['snr'] = np.mean(snr_values)
            if latent_std_values:
                avg_losses['latent_std'] = np.mean(latent_std_values)
            if channel_std_values:
                avg_losses['min_channel_std'] = np.mean(channel_std_values)
        else:
            avg_losses = {'total': float('inf')}
            logger.error("No batches were successfully processed in this epoch!")
        
        # Log results
        logger.info(f"Epoch {current_epoch}/{EPOCHS} complete:")
        logger.info(f"  Total loss: {avg_losses.get('total', 0):.4f}")
        
        if phase == 1:
            logger.info(f"  Recon loss: {avg_losses.get('recon', 0):.4f}")
            logger.info(f"  KL loss: {avg_losses.get('kl', 0):.6f}")
            logger.info(f"  Diversity loss: {avg_losses.get('diversity', 0):.6f}")
            logger.info(f"  Latent std: {avg_losses.get('latent_std', 0):.3f}")
            if 'snr' in avg_losses:
                logger.info(f"  SNR: {avg_losses['snr']:.2f}dB")
        else:
            logger.info(f"  Drift loss: {avg_losses.get('drift', 0):.4f}")
        
        return avg_losses

    def _debug_training_loop(self, batch: Dict, loss_dict: Dict) -> None:
        """Debug helper for training loop."""
        logger.info("=== DEBUG INFO ===")
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                logger.info(f"  {key}: {value.item():.4f}")

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
            logger.error(f"Snapshot not found: {snapshot_path}")
            return False
        
        try:
            snapshot = torch.load(snapshot_path, map_location=config.DEVICE, weights_only=False)
            
            if load_vae and 'model_state' in snapshot and snapshot.get('model_type') == 'vae':
                self.vae.load_state_dict(snapshot['model_state'])
                if 'optimizer_state' in snapshot:
                    self.opt_vae.load_state_dict(snapshot['optimizer_state'])
                logger.info(f"Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
            
            if load_drift and 'drift_state' in snapshot and snapshot.get('model_type') == 'drift':
                self.drift.load_state_dict(snapshot['drift_state'])
                if 'opt_drift_state' in snapshot:
                    self.opt_drift.load_state_dict(snapshot['opt_drift_state'])
                logger.info(f"Loaded Drift from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
            
            if phase is not None:
                self.phase = phase
                logger.info(f"Set phase to {phase}")
            
            if 'epoch' in snapshot:
                self.epoch = snapshot['epoch']
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return False

    def generate_samples(self, labels: Optional[List[int]] = None, 
                        num_samples: int = 8, temperature: float = 0.8) -> Path:
        """
        Generate samples using probability flow ODE.
        
        FIXES APPLIED:
        1. Match training noise scale: z0 = randn * 0.8
        2. Remove spurious Langevin noise
        3. Replace tanh with gentle clamping
        4. Use adaptive clipping based on running statistics
        5. Temperature based on global EPOCHS
        """
        self.vae.eval()
        self.drift.eval()
        
        if labels is None:
            labels = [i % 10 for i in range(num_samples)]
        
        with torch.no_grad():
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=config.DEVICE)
            
            # FIX #1: Match training scale (0.8)
            z = torch.randn(num_samples, LATENT_CHANNELS, LATENT_H, LATENT_W, device=config.DEVICE) * 0.8
            
            steps = DEFAULT_STEPS
            dt = 1.0 / steps
            
            # FIX #5: Temperature based on global EPOCHS
            if hasattr(self, 'epoch') and self.epoch > 0:
                # Match training annealing schedule
                effective_temp = 0.3 + 0.7 * (1 - self.epoch / EPOCHS)
                temperature = temperature * effective_temp
            
            for i in range(steps):
                t = torch.full((num_samples, 1), i * dt, device=config.DEVICE)
                
                # FIX #2: Removed Langevin-style noise injection
                # No noise added to state during inference
                
                # Get drift prediction
                drift = self.drift(z, t, labels_tensor)
                
                # FIX #4: Adaptive clipping based on running statistics
                adaptive_threshold = self.drift.get_adaptive_threshold(num_std=3.0)
                drift_norm = drift.flatten(start_dim=1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
                
                mask = drift_norm > adaptive_threshold
                if mask.any():
                    drift = torch.where(
                        mask,
                        drift * (adaptive_threshold / (drift_norm + 1e-8)),
                        drift
                    )
                
                # Euler step
                z = z + drift * dt
                
                # FIX #3: Gentle clamping instead of tanh
                z = torch.clamp(z, -10, 10)
                
                if torch.isnan(z).any():
                    logger.error(f"NaN detected at step {i}!")
                    break
            
            # Decode to images
            images = self.vae.decode(z, labels_tensor)
            images = torch.clamp(images, -1, 1)

            # Save results
            images_display = (images + 1) / 2
            images_display = torch.clamp(images_display, 0, 1)

            grid_path = DIRS["samples"] / f"gen_epoch{self.epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            dm.save_image_grid(images_display, grid_path, nrow=4, normalize=False)
            dm.save_individual_images(images_display, labels, self.epoch, base_dir=DIRS["samples"])
            dm.save_raw_tensors(z, images, labels, self.epoch, base_dir=DIRS["samples"])

            logger.info(f"Generated {num_samples} samples for labels {labels}")
            logger.info(f"Images saved to: {grid_path}")
            return grid_path

    def list_available_snapshots(self) -> List[Path]:
        """List all available snapshots."""
        snap_files = list(DIRS["snaps"].glob("*_snapshot_epoch_*.pt"))
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
            logger.error(f"Failed to inspect snapshot: {e}")
            return {'path': snapshot_path, 'error': str(e)}

    def export_onnx(self) -> None:
        """Export models to ONNX format."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX export requires onnx and onnxruntime packages")
            return
        
        try:
            # Export VAE
            dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=config.DEVICE)
            dummy_label = torch.tensor([0], device=config.DEVICE)
            
            vae_path = DIRS["onnx"] / "vae.onnx"
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
            logger.info(f"VAE exported to {vae_path}")
            
            # Export Drift
            dummy_z = torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, device=config.DEVICE)
            dummy_t = torch.tensor([[0.5]], device=config.DEVICE)
            
            drift_path = DIRS["onnx"] / "drift.onnx"
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
            logger.info(f"Drift exported to {drift_path}")
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(num_epochs: int = EPOCHS, resume_from_snapshot: Optional[Path] = None) -> None:
    """Main training loop."""
    loader = dm.load_data()
    trainer = EnhancedLabelTrainer(loader)
    
    if resume_from_snapshot:
        print(f"\n Resuming from snapshot: {resume_from_snapshot}")
        trainer.load_from_snapshot(resume_from_snapshot)
    else:
        latest_checkpoint = DIRS["ckpt"] / "latest.pt"
        if latest_checkpoint.exists():
            resume = input("\n Found existing checkpoint. Resume training? (y/n): ").strip().lower()
            if resume == 'y':
                trainer.load_checkpoint()
            else:
                print("Starting fresh training...")
    
    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Training schedule mode: {TRAINING_SCHEDULE['mode']}")
    
    for epoch in range(trainer.epoch, num_epochs):
        trainer.epoch = epoch
        epoch_losses = trainer.train_epoch()
        
        if USE_KPI_TRACKING:
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
            
            # Check for new best composite score
            if comp_score > trainer.best_composite_score:
                trainer.best_composite_score = comp_score
                # FIX: Added is_best parameter
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
            logger.info("Generating samples...")
            trainer.generate_samples()
        
        # Check early stopping
        if USE_KPI_TRACKING and trainer.phase == 2:
            if trainer.kpi_tracker.should_stop(phase=trainer.phase):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    logger.info(f"Training complete! Best loss: {trainer.best_loss:.4f}")
    logger.info(f"Best composite score: {trainer.best_composite_score:.4f}")
    
    if ONNX_AVAILABLE:
        trainer.export_onnx()
    
    trainer.generate_samples(labels=list(range(8)))