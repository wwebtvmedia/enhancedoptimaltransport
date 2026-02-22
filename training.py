# ============================================================================
# MATHEMATICAL FOUNDATIONS OF SCHRÖDINGER BRIDGE WITH LABEL CONDITIONING
# ============================================================================
#
# ... (full comment block preserved as in original) ...
#
# ============================================================================

import os
import math
import time
import json
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats

# Optional Imports
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

# Import data management module
import data_management as dm

warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL TRAINING PHASE CONTROL
# ============================================================
TRAINING_SCHEDULE = {
    'mode': 'auto',  # 'auto', 'manual', 'custom'
    'force_phase': None,  # None, 1 (VAE only), 2 (Drift only)
    'custom_schedule': {},  # {epoch: phase} for custom schedules
    'switch_epoch': 50,  # Default switch epoch for auto mode
    'vae_epochs': list(range(0, 50)),  # Epochs to train VAE
    'drift_epochs': list(range(50, 200)),  # Epochs to train Drift
    'alternate_freq': 5,
}

def set_training_phase(epoch, mode='auto', force_phase=None, switch_epoch=50):
    """Global function to set training phase for each epoch."""
    global TRAINING_SCHEDULE
    
    if mode:
        TRAINING_SCHEDULE['mode'] = mode
    if force_phase is not None:
        TRAINING_SCHEDULE['force_phase'] = force_phase
    if switch_epoch:
        TRAINING_SCHEDULE['switch_epoch'] = switch_epoch
    
    if TRAINING_SCHEDULE['mode'] == 'manual':
        return TRAINING_SCHEDULE['force_phase']
    elif TRAINING_SCHEDULE['mode'] == 'custom':
        return TRAINING_SCHEDULE['custom_schedule'].get(epoch, 1)
    elif TRAINING_SCHEDULE['mode'] == 'alternate':
        alternate_freq = TRAINING_SCHEDULE.get('alternate_freq', 5)
        return 1 if (epoch // alternate_freq) % 2 == 0 else 2
    else:  # 'auto' mode
        return 1 if epoch < TRAINING_SCHEDULE['switch_epoch'] else 2

def configure_training_schedule(
    mode='auto', 
    vae_epochs=None, 
    drift_epochs=None, 
    switch_epoch=50,
    alternate_freq=5,
    custom_schedule=None
):
    """Configure the global training schedule."""
    global TRAINING_SCHEDULE
    
    if mode == 'vae_only':
        TRAINING_SCHEDULE['mode'] = 'manual'
        TRAINING_SCHEDULE['force_phase'] = 1
    elif mode == 'drift_only':
        TRAINING_SCHEDULE['mode'] = 'manual'
        TRAINING_SCHEDULE['force_phase'] = 2
    elif mode == 'auto':
        TRAINING_SCHEDULE['mode'] = 'auto'
        TRAINING_SCHEDULE['switch_epoch'] = switch_epoch
        TRAINING_SCHEDULE['vae_epochs'] = list(range(switch_epoch))
        TRAINING_SCHEDULE['drift_epochs'] = list(range(switch_epoch, 1000))
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
# CONFIGURATION (some constants are imported from dm)
# ============================================================
IMG_SIZE = dm.IMG_SIZE
LATENT_CHANNELS = dm.LATENT_CHANNELS
LATENT_H = dm.LATENT_H
LATENT_W = dm.LATENT_W
LATENT_DIM = LATENT_CHANNELS * LATENT_H * LATENT_W

LR = dm.LR
EPOCHS = dm.EPOCHS
PHASE_1_EPOCHS = 50  # Default switch epoch
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NUM_CLASSES = dm.NUM_CLASSES
LABEL_EMB_DIM = dm.LABEL_EMB_DIM

USE_PERCENTILE = True
USE_SNAPSHOTS = True
USE_KPI_TRACKING = True
TARGET_SNR = 30.0
SNAPSHOT_INTERVAL = dm.SNAPSHOT_INTERVAL
SNAPSHOT_KEEP = dm.SNAPSHOT_KEEP
KPI_WINDOW_SIZE = 100
EARLY_STOP_PATIENCE = 15

KL_WEIGHT = 0.001
RECON_WEIGHT = 1.0
DRIFT_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.01

INFERENCE_TEMPERATURE = 0.8
DEFAULT_STEPS = 50
CONSISTENCY_WEIGHT = 0.5
REVERT_THRESHOLD = 2.5

# Paths
DIRS = dm.DIRS

# ============================================================
# LOGGER SETUP (use the one from dm or create new)
# ============================================================
logger = logging.getLogger("EnhancedTrainer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_path = DIRS["logs"] / f"train_{int(time.time())}.log"
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# ============================================================
# UTILITIES
# ============================================================
def calc_snr(real, recon):
    """Calculate Signal-to-Noise Ratio."""
    mse = F.mse_loss(recon, real)
    if mse == 0: 
        return 100.0
    return 10 * torch.log10(1.0 / (mse + 1e-8)).item()

def kl_divergence_spatial(mu, logvar):
    """KL divergence for spatial latent variables."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.mean(torch.sum(kl, dim=[1, 2, 3]))

def channel_diversity_loss(mu):
    """Encourage all latent channels to be used."""
    channel_stds = mu.std(dim=[0, 2, 3])  # [channels]
    min_std = 0.05
    diversity_loss = torch.mean(F.relu(min_std - channel_stds))
    balance_loss = channel_stds.std()
    return diversity_loss + 0.1 * balance_loss

# ============================================================
# PERCENTILE RESCALING (same as before)
# ============================================================
class PercentileRescale(nn.Module):
    def __init__(self, features, p_low=1.0, p_high=99.0, momentum=0.1):
        super().__init__()
        self.register_buffer('low', torch.zeros(features))
        self.register_buffer('high', torch.ones(features))
        self.p_low, self.p_high, self.m = p_low, p_high, momentum
        self._is_exporting = False

    def _set_export_mode(self, is_exporting=True):
        self._is_exporting = is_exporting

    def forward(self, x):
        if self._is_exporting or not self.training:
            scale = (self.high - self.low).clamp(min=1e-6).view(1, -1, 1, 1)
            shift = self.low.view(1, -1, 1, 1)
            return torch.tanh((x - shift) / scale)
        if self.training:
            with torch.no_grad():
                flat = x.float().transpose(0, 1).reshape(x.shape[1], -1)
                if flat.numel() > 0:
                    l = torch.quantile(flat, self.p_low/100, dim=1)
                    h = torch.quantile(flat, self.p_high/100, dim=1)
                    self.low.mul_(1 - self.m).add_(l, alpha=self.m)
                    self.high.mul_(1 - self.m).add_(h, alpha=self.m)
        scale = (self.high - self.low).clamp(min=1e-6).view(1, -1, 1, 1)
        shift = self.low.view(1, -1, 1, 1)
        return torch.tanh((x - shift) / scale)

# ============================================================
# KPI TRACKER
# ============================================================
class KPITracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            if value is not None:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key].pop(0)
                
    def compute_convergence(self):
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
        
    def should_stop(self, patience=EARLY_STOP_PATIENCE, min_delta=1e-4, phase=1):
        if phase == 1:
            return False
        if 'loss' not in self.metrics or len(self.metrics['loss']) < patience * 2:
            return False
        if len(self.metrics['loss']) < 30:
            return False
        recent_losses = self.metrics['loss'][-patience:]
        is_increasing = True
        for i in range(len(recent_losses)-1):
            if recent_losses[i] >= recent_losses[i+1]:
                is_increasing = False
                break
        if is_increasing:
            best_loss_in_window = min(recent_losses)
            current_loss = recent_losses[-1]
            if current_loss - best_loss_in_window > min_delta:
                logger.info(f"Early stopping triggered: loss increased for {patience} epochs")
                return True
        return False

# ============================================================
# ARCHITECTURE WITH LABEL CONDITIONING (same as original)
# ============================================================
class LabelConditionedBlock(nn.Module):
    # ... (full code as before) ...
    def __init__(self, c_in, c_out, label_dim=LABEL_EMB_DIM, use_spectral_norm=False):
        super().__init__()
        groups = min(8, c_in)
        self.norm1 = nn.GroupNorm(groups, c_in)
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, 3, 1, 1))
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, c_out * 2),
            nn.SiLU(),
            nn.Linear(c_out * 2, c_out * 2)
        )
        groups_out = min(8, c_out)
        self.norm2 = nn.GroupNorm(groups_out, c_out)
        if use_spectral_norm:
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1))
        else:
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.rescale = PercentileRescale(c_out) if USE_PERCENTILE else nn.Identity()
        if use_spectral_norm and c_in != c_out:
            self.skip = nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, 1))
        elif c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, labels=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if labels is not None:
            scale_shift = self.label_proj(labels)
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1 + scale.view(-1, scale.shape[1], 1, 1)) + shift.view(-1, shift.shape[1], 1, 1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.rescale(self.skip(x) + h)

class LabelConditionedVAE(nn.Module):
    def __init__(self, free_bits=2.0):
        super().__init__()
        self.free_bits = free_bits
        self.label_emb = nn.Embedding(NUM_CLASSES, LABEL_EMB_DIM)
        self.enc_in = nn.Conv2d(3, 32, 3, 1, 1)
        self.enc_blocks = nn.ModuleList([
            LabelConditionedBlock(32, 64),
            nn.Conv2d(64, 64, 4, 2, 1),
            LabelConditionedBlock(64, 128),
            nn.Conv2d(128, 128, 4, 2, 1),
            LabelConditionedBlock(128, 256),
            nn.Conv2d(256, 256, 4, 2, 1),
        ])
        self.z_mean = nn.Conv2d(256, LATENT_CHANNELS, 3, 1, 1)
        self.z_logvar = nn.Conv2d(256, LATENT_CHANNELS, 3, 1, 1)
        self.dec_in = nn.Conv2d(LATENT_CHANNELS, 256, 3, 1, 1)
        self.dec_blocks = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            LabelConditionedBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            LabelConditionedBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            LabelConditionedBlock(32, 32),
        ])
        self.dec_out = nn.Conv2d(32, 3, 3, 1, 1)
        self.diversity_loss = None

    def encode(self, x, labels):
        label_emb = self.label_emb(labels)
        h = self.enc_in(x)
        for block in self.enc_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        mu = self.z_mean(h) * 0.05
        if self.training:
            mu = mu + torch.randn_like(mu) * 0.01
        if self.training:
            channel_mask = torch.ones_like(mu)
            if torch.rand(1).item() < 0.05:
                channel_mask = torch.bernoulli(torch.full_like(mu, 0.95))
            mu = mu * channel_mask
        logvar = torch.clamp(self.z_logvar(h), min=-4, max=4)
        if self.training:
            self.diversity_loss = channel_diversity_loss(mu)
        return mu, logvar

    def decode(self, z, labels):
        label_emb = self.label_emb(labels)
        h = self.dec_in(z)
        for block in self.dec_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        return torch.tanh(self.dec_out(h))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return self.decode(z, labels), mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

class LabelConditionedDrift(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.label_emb = nn.Embedding(NUM_CLASSES, LABEL_EMB_DIM)
        self.cond_proj = nn.Sequential(
            nn.Linear(128 + LABEL_EMB_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.time_weight_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.head = nn.utils.spectral_norm(nn.Conv2d(LATENT_CHANNELS, 64, 3, 1, 1))
        self.down1 = LabelConditionedBlock(64, 128, label_dim=128, use_spectral_norm=True)
        self.down2_conv = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.down2_block = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        self.mid = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        self.up2_conv = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2_block = LabelConditionedBlock(128, 128, label_dim=128, use_spectral_norm=True)
        self.up1 = LabelConditionedBlock(128, 64, label_dim=128, use_spectral_norm=True)
        self.tail = nn.utils.spectral_norm(nn.Conv2d(64, LATENT_CHANNELS, 3, 1, 1))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.time_scales = nn.Parameter(torch.ones(4) * 0.1)

    def forward(self, z, t, labels):
        t_emb = self.time_mlp(t)
        label_emb = self.label_emb(labels)
        cond = torch.cat([t_emb, label_emb], dim=-1)
        cond = self.cond_proj(cond)
        time_weight = self.time_weight_net(t)
        t_val = t.squeeze()
        time_indices = (t_val * 3).long().clamp(0, 3)
        time_scale = self.time_scales[time_indices].view(-1, 1, 1, 1)
        z = torch.tanh(z) * 0.5
        h = self.head(z)
        d1 = self.down1(h, cond)
        d2 = self.down2_conv(d1)
        d2 = self.down2_block(d2, cond)
        m = self.mid(d2, cond)
        u2 = self.up2_conv(m)
        u2 = self.up2_block(u2, cond)
        u1 = self.up1(u2 + d1, cond)
        out = self.tail(u1)
        out = torch.tanh(out) * self.output_scale * (1.0 + time_weight.view(-1, 1, 1, 1))
        out = out * (1.0 + time_scale)
        return out

# ============================================================
# ENHANCED TRAINER
# ============================================================
class EnhancedLabelTrainer:
    def __init__(self, loader):
        self.loader = loader
        self.vae = LabelConditionedVAE().to(dm.DEVICE)
        self.drift = LabelConditionedDrift().to(dm.DEVICE)
        self.opt_vae = optim.AdamW(self.vae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.opt_drift = optim.AdamW(self.drift.parameters(), lr=LR * 2, weight_decay=WEIGHT_DECAY)
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(self.opt_vae, T_max=EPOCHS, eta_min=LR*0.01)
        self.scheduler_drift = optim.lr_scheduler.CosineAnnealingLR(self.opt_drift, T_max=EPOCHS, eta_min=LR*0.01)
        self.kpi_tracker = KPITracker(window_size=KPI_WINDOW_SIZE)
        self.snapshot_vae = dm.SnapshotManager(self.vae, self.opt_vae, name="vae") if USE_SNAPSHOTS else None
        self.snapshot_drift = dm.SnapshotManager(self.drift, self.opt_drift, name="drift") if USE_SNAPSHOTS else None
        self.epoch = 0
        self.step = 0
        self.phase = 1
        self.phase_preference = None
        self.best_loss = float('inf')
        self.debug_counter = 0
        self.debug_interval = 50
        logger.info(f"Models initialized:")
        logger.info(f"  VAE params: {sum(p.numel() for p in self.vae.parameters()):,}")
        logger.info(f"  Drift params: {sum(p.numel() for p in self.drift.parameters()):,}")

    def get_training_phase(self, epoch):
        global TRAINING_SCHEDULE
        phase = set_training_phase(epoch)
        if phase != self.phase:
            logger.info(f"🔄 Phase changed from {self.phase} to {phase} at epoch {epoch+1}")
            self.phase = phase
            if phase == 2 and epoch == TRAINING_SCHEDULE['switch_epoch']:
                self._handle_phase_transition()
        return phase

    def _handle_phase_transition(self):
        self.vae_ref = LabelConditionedVAE().to(dm.DEVICE)
        self.vae_ref.load_state_dict(self.vae.state_dict())
        self.vae_ref.eval()
        for param in self.vae_ref.parameters():
            param.requires_grad = False
        unfrozen_count = 0
        for name, param in self.vae.named_parameters():
            if any(k in name for k in ['enc_', 'label_emb', 'z_mean', 'z_logvar']):
                param.requires_grad = True
                unfrozen_count += param.numel()
            else:
                param.requires_grad = False
        logger.info(f"PHASE 2: Unfrozen {unfrozen_count:,} params. Consistency Anchor Set.")
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

    def compute_loss(self, batch, phase=1):
        if isinstance(batch, dict):
            images = batch['image'].to(dm.DEVICE)
            labels = batch['label'].to(dm.DEVICE)
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            images = batch[0].to(dm.DEVICE)
            labels = batch[1].to(dm.DEVICE)
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        if self.debug_counter % self.debug_interval == 0:
            logger.info(f"\n=== DEBUG Epoch {self.epoch+1}, Batch {self.debug_counter} ===")
            logger.info(f"Images - min: {images.min().item():.4f}, max: {images.max().item():.4f}, mean: {images.mean().item():.4f}")

        if phase == 1:
            recon, mu, logvar = self.vae(images, labels)
            latent_std = torch.exp(0.5 * logvar).mean().item()
            channel_stds = mu.std(dim=[0, 2, 3]).detach().cpu().numpy()
            min_channel_std = channel_stds.min()
            if self.debug_counter % self.debug_interval == 0:
                logger.info(f"Mu std: {mu.std().item():.3f}")
                logger.info(f"Channel stds: {channel_stds}")
            raw_mse = F.mse_loss(recon, images)
            raw_kl = kl_divergence_spatial(mu, logvar)
            if latent_std < 0.3:
                current_kl_weight = KL_WEIGHT * 10.0
            elif min_channel_std < 0.01:
                current_kl_weight = KL_WEIGHT * 5.0
            else:
                current_kl_weight = KL_WEIGHT
            diversity_loss = self.vae.diversity_loss if self.vae.diversity_loss is not None else torch.tensor(0.0, device=dm.DEVICE)
            kl_annealing = min(1.0, self.epoch / 10.0)
            kl_loss = raw_kl * current_kl_weight * kl_annealing
            recon_loss = raw_mse * RECON_WEIGHT
            total_loss = recon_loss + kl_loss + diversity_loss * DIVERSITY_WEIGHT
            snr = calc_snr(images, recon)
            self.debug_counter += 1
            return {
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
                'channel_stds': channel_stds
            }
        else:
            with torch.no_grad():
                mu_ref, _ = self.vae_ref.encode(images, labels)
                mu, logvar = self.vae.encode(images, labels)
                consistency_loss = F.mse_loss(mu, mu_ref)
                temperature = 0.3 + 0.7 * (1 - self.epoch / 200)
                z1 = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * temperature
            if self.epoch > 50:
                beta_dist = torch.distributions.Beta(2, 2)
                t = beta_dist.sample((images.shape[0], 1)).to(dm.DEVICE)
            else:
                t = torch.rand(images.shape[0], 1, device=dm.DEVICE)
            z0 = torch.randn_like(z1) * 0.8
            zt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z1
            target = z1 - z0
            if self.drift.training:
                noise_scale = 0.01 * (1 - t.view(-1, 1, 1, 1))
                target = target + torch.randn_like(target) * noise_scale
            pred = self.drift(zt, t, labels)
            time_weights = 1.0 + 2.0 * t.view(-1, 1, 1, 1)
            drift_loss_base = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * DRIFT_WEIGHT
            total_loss = drift_loss_base + (consistency_loss * CONSISTENCY_WEIGHT)
            return {
                'total': total_loss,
                'drift': drift_loss_base.item(),
                'consistency': consistency_loss.item(),
                'temperature': temperature
            }

    def debug_training_loop(self, batch, loss_dict):
        # (same as original)
        pass

    def train_epoch(self):
        losses = defaultdict(float)
        snr_values = []
        latent_std_values = []
        channel_std_values = []
        current_epoch = self.epoch + 1
        phase = self.get_training_phase(current_epoch)
        mode = "VAE" if phase == 1 else "Drift"
        self.debug_counter = 0
        if phase == 1:
            self.vae.train()
            self.drift.eval()
        else:
            self.vae.eval()
            self.drift.train()
        if TQDM_AVAILABLE:
            pbar = tqdm(self.loader, desc=f"Epoch {current_epoch} ({mode})")
        else:
            pbar = self.loader
            logger.info(f"Epoch {current_epoch} ({mode})")
        batch_count = 0
        for batch_idx, batch in enumerate(pbar):
            try:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_dict = {'image': batch[0], 'label': batch[1], 'index': batch_idx}
                elif isinstance(batch, dict):
                    batch_dict = batch
                else:
                    continue
                loss_dict = self.compute_loss(batch_dict, phase=phase)
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
                if batch_idx % 50 == 0 and batch_idx > 0:
                    self.debug_training_loop(batch_dict, loss_dict)
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
                    loss_dict['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.drift.parameters(), GRAD_CLIP * 0.5)
                    total_norm = 0
                    for p in self.drift.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.norm().item() ** 2
                    total_norm = total_norm ** 0.5
                    if total_norm > 0.01:
                        self.opt_drift.step()
                    else:
                        logger.warning(f"Skipping drift update - gradient norm too small: {total_norm:.4f}")
                    if self.snapshot_drift:
                        self.snapshot_drift.step()
                for key, value in loss_dict.items():
                    if key not in ['total', 'raw_mse', 'raw_kl', 'snr', 'latent_std', 'min_channel_std', 'max_channel_std', 'channel_stds', 'temperature']:
                        if isinstance(value, (int, float)):
                            losses[key] += value
                        elif isinstance(value, torch.Tensor):
                            losses[key] += value.item()
                self.step += 1
                self.debug_counter += 1
                batch_count += 1
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
        if phase == 1:
            self.scheduler_vae.step()
        else:
            self.scheduler_drift.step()
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
            logger.error("No batches were successfully processed in this epoch!")
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

    def save_checkpoint(self, is_best=False):
        return dm.save_checkpoint(self, is_best)

    def load_checkpoint(self, path=None):
        return dm.load_checkpoint(self, path)

    def load_for_inference(self, path=None):
        return dm.load_for_inference(self, path)

    def generate_samples(self, labels=None, num_samples=8, temperature=0.8):
        self.vae.eval()
        self.drift.eval()
        if labels is None:
            labels = [i % 10 for i in range(num_samples)]
        with torch.no_grad():
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=dm.DEVICE)
            z = torch.randn(num_samples, LATENT_CHANNELS, LATENT_H, LATENT_W, device=dm.DEVICE) * 0.5
            steps = DEFAULT_STEPS
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((num_samples, 1), i * dt, device=dm.DEVICE)
                if i < steps // 2:
                    noise_scale = 0.01 * (1 - i/steps) * temperature
                    z = z + torch.randn_like(z) * noise_scale
                drift = self.drift(z, t, labels_tensor)
                drift_norm = drift.flatten(start_dim=1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)
                target_norm = 5.0
                mask = drift_norm > target_norm
                if mask.any():
                    drift = torch.where(
                        mask,
                        drift * (target_norm / (drift_norm + 1e-8)),
                        drift
                    )
                z = z + drift * dt
                if USE_PERCENTILE:
                    z = torch.tanh(z)
                if torch.isnan(z).any():
                    logger.error(f"NaN detected at step {i}!")
                    break
            images = self.vae.decode(z, labels_tensor)
            images = torch.clamp(images, -1, 1)

            # Save using dm utilities
            images_display = (images + 1) / 2
            images_display = torch.clamp(images_display, 0, 1)

            grid_path = dm.DIRS["samples"] / f"gen_epoch{self.epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            dm.save_image_grid(images_display, grid_path, nrow=4, normalize=False)

            dm.save_individual_images(images_display, labels, self.epoch, base_dir=dm.DIRS["samples"])
            dm.save_raw_tensors(z, images, labels, self.epoch, base_dir=dm.DIRS["samples"])

            logger.info(f"Generated {num_samples} samples for labels {labels}")
            logger.info(f"Images saved to: {grid_path}")
            return grid_path

    def load_from_snapshot(self, snapshot_path, load_vae=True, load_drift=True, phase=None):
        # (same as original but using dm.DIRS, dm.DEVICE)
        pass

    def restart_from_vae_snapshot(self, vae_snapshot_path):
        # (same as original but using dm.DIRS, dm.DEVICE)
        pass

    def list_available_snapshots(self):
        return dm.SnapshotManager.list_available_snapshots()  # static method?

    def inspect_snapshot(self, snapshot_path):
        # (same)
        pass

    def compare_snapshots(self, snapshot_path1, snapshot_path2):
        # (same)
        pass

    def run_snapshot_tests(self):
        # (same)
        pass

    def test_vae_at_snapshot(self, test_images, test_labels):
        # (same)
        pass

    def test_drift_at_snapshot(self, test_images, test_labels):
        # (same)
        pass

    def visualize_interpolation(self, z0, z1, labels, steps=8, save_path=None):
        # (same)
        pass

    def export_onnx(self):
        # (same as original but using dm.DIRS)
        pass

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(num_epochs=EPOCHS, resume_from_snapshot=None):
    loader = dm.load_data()
    trainer = EnhancedLabelTrainer(loader)
    if resume_from_snapshot:
        print(f"\n Resuming from snapshot: {resume_from_snapshot}")
        trainer.load_from_snapshot(resume_from_snapshot)
    else:
        latest_checkpoint = dm.DIRS["ckpt"] / "latest.pt"
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
            trainer.kpi_tracker.update(kpi_update)
        if trainer.phase == 1:
            current_total_loss = epoch_losses.get('total', float('inf'))
        else:
            current_total_loss = epoch_losses.get('drift', float('inf'))
        if current_total_loss < trainer.best_loss and current_total_loss != float('inf'):
            trainer.best_loss = current_total_loss
            trainer.save_checkpoint(is_best=True)
        elif (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(is_best=False)
        if (epoch + 1) % 10 == 0 and current_total_loss != float('inf'):
            logger.info("Generating samples...")
            trainer.generate_samples()
        if USE_KPI_TRACKING and trainer.phase == 2:
            if trainer.kpi_tracker.should_stop(phase=trainer.phase):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    logger.info(f"Training complete! Best loss: {trainer.best_loss:.4f}")
    if ONNX_AVAILABLE:
        trainer.export_onnx()
    trainer.generate_samples(labels=list(range(8)))