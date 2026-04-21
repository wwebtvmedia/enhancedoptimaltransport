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
import torch.optim as optim
import torch.amp
import torchvision.utils as vutils
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
import scipy.linalg
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
        """Sample from the exact OU bridge between z0 and z1."""
        exp_neg_theta_t = torch.exp(-self.theta * t)
        exp_neg_theta_1_t = torch.exp(-self.theta * (1 - t))
        exp_neg_theta = torch.exp(-self.theta)
        
        numerator = (exp_neg_theta_t * (1 - exp_neg_theta_1_t**2) * z0 + 
                     (1 - exp_neg_theta_t**2) * exp_neg_theta_1_t * z1)
        denominator = 1 - exp_neg_theta**2
        mean = numerator / denominator
        
        var = (self.sigma**2 / (2 * self.theta)) * ((1 - exp_neg_theta_t**2) * (1 - exp_neg_theta_1_t**2)) / (1 - exp_neg_theta**2)
        var = var.clamp(min=0)
        return mean, var

    def bridge_velocity(self, z0, z1, t):
        """Compute the velocity (time derivative of the mean) of the OU bridge."""
        dt = 1e-4
        mean_plus, _ = self.bridge_sample(z0, z1, t + dt)
        mean_minus, _ = self.bridge_sample(z0, z1, (t - dt).clamp(min=0))
        return (mean_plus - mean_minus) / (2 * dt)

# ============================================================
# UTILITIES
# ============================================================
def calc_snr(real: torch.Tensor, recon: torch.Tensor) -> float:
    mse = F.mse_loss(recon, real)
    if mse == 0: return 100.0
    return 10 * torch.log10(1.0 / (mse + 1e-8)).item()
    
def kl_divergence_spatial(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.sum(kl, dim=[1, 2, 3])
    kl = torch.max(kl, torch.full_like(kl, config.FREE_BITS))
    return torch.mean(kl)

def total_variation_loss(img: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).mean()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).mean()
    return weight * (tv_h + tv_w)

def composite_score(loss_dict: Dict, phase: int) -> float:
    score = 0.0
    if phase == 1:
        if 'snr' in loss_dict: score += loss_dict['snr'] / config.TARGET_SNR
        if 'kl' in loss_dict: score -= loss_dict['kl'] * 10
        if 'diversity' in loss_dict: score -= loss_dict['diversity'] * 100
    else:
        if 'drift' in loss_dict: score -= loss_dict['drift'] * 10
        if 'consistency' in loss_dict: score -= loss_dict['consistency'] * 10
    return score

def set_training_phase(epoch: int) -> int:
    mode = config.TRAINING_SCHEDULE['mode']
    if mode == 'manual':
        phase = config.TRAINING_SCHEDULE['force_phase']
        return 1 if phase is None else phase
    elif mode == 'alternate':
        alt_freq = config.TRAINING_SCHEDULE.get('alternate_freq', 5)
        return 1 if (epoch // alt_freq) % 2 == 0 else 2
    elif mode == 'three_phase':
        e1 = config.TRAINING_SCHEDULE['switch_epoch_1']
        e2 = config.TRAINING_SCHEDULE['switch_epoch_2']
        if epoch < e1: return 1
        elif epoch < e2: return 2
        else: return 3
    else:
        return 1 if epoch < config.TRAINING_SCHEDULE['switch_epoch'] else 2

# ============================================================
# KPI TRACKER
# ============================================================
class KPITracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.composite_scores = []
        
    def update(self, metrics_dict: Dict) -> None:
        for key, value in metrics_dict.items():
            if value is not None:
                self.metrics[key].append(value)
                if len(self.metrics[key]) > self.window_size: self.metrics[key].pop(0)
        if 'composite_score' in metrics_dict:
            self.composite_scores.append(metrics_dict['composite_score'])
            if len(self.composite_scores) > self.window_size: self.composite_scores.pop(0)

    def compute_convergence(self) -> Dict:
        convergence = {}
        if 'loss' in self.metrics and len(self.metrics['loss']) >= 10:
            loss_values = self.metrics['loss'][-20:]
            convergence['loss_trend'] = np.polyfit(np.arange(len(loss_values)), loss_values, 1)[0]
            convergence['convergence_score'] = 1.0 / (1.0 + np.std(loss_values))
        return convergence

    def should_stop(self, patience: int = config.EARLY_STOP_PATIENCE, min_delta: float = 1e-4, phase: int = 1) -> bool:
        if phase == 1 or 'loss' not in self.metrics or len(self.metrics['loss']) < patience * 2: return False
        recent = self.metrics['loss'][-patience:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            if recent[-1] - min(recent) > min_delta: return True
        return False

# ============================================================
# ENHANCED TRAINER
# ============================================================
class EnhancedLabelTrainer:
    def __init__(self, loader):
        self.loader = loader
        self.contrastive_criterion = ContrastiveLoss()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        self.best_composite_score = float('-inf')
        self.phase2_start_epoch = None
        self.vae_ref = None
        self.ou_ref = OUReference(theta=config.OU_THETA, sigma=config.OU_SIGMA) if config.USE_OU_BRIDGE else None
        
        # Initialize AMP
        self.scaler = None
        if config.USE_AMP and config.AMP_AVAILABLE:
            if config.DEVICE.type == 'cuda': self.scaler = torch.cuda.amp.GradScaler()
        
        # 1. Initialize models on CPU to avoid startup OOM
        self.vae = models.LabelConditionedVAE()
        self.drift = models.LabelConditionedDrift()

        if config.USE_LORA:
            n_vae = lora.apply_lora(self.vae, r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES)
            n_drift = lora.apply_lora(self.drift, r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, target_modules=config.LORA_TARGET_MODULES)
            config.logger.info(f"🚀 LoRA: VAE ({n_vae} layers), Drift ({n_drift} layers)")

        # 2. Setup Optimizers while on CPU (references will persist after .to(device))
        self.opt_vae = optim.AdamW(filter(lambda p: p.requires_grad, self.vae.parameters()), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        self.opt_drift = optim.AdamW(filter(lambda p: p.requires_grad, self.drift.parameters()), lr=config.LR * config.DRIFT_LR_MULTIPLIER, weight_decay=config.WEIGHT_DECAY)
        self.scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(self.opt_vae, T_max=config.EPOCHS, eta_min=config.LR*0.01)
        self.scheduler_drift = optim.lr_scheduler.CosineAnnealingLR(self.opt_drift, T_max=config.EPOCHS, eta_min=config.LR*0.01)
        self.kpi_tracker = KPITracker(window_size=config.KPI_WINDOW_SIZE)
          
        if config.USE_SNAPSHOTS:
            self.snapshot_vae = dm.SnapshotManager(self.vae, self.opt_vae, name="vae")
            self.snapshot_drift = dm.SnapshotManager(self.drift, self.opt_drift, name="drift")
        else:
            self.snapshot_vae = self.snapshot_drift = None
            
        # 3. Register buffers
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        window_size = 11
        gauss = torch.exp(-(torch.arange(window_size, dtype=torch.float32) - window_size//2)**2 / (2 * 1.5**2))
        gauss = gauss / gauss.sum()
        window = (gauss[:, None] * gauss[None, :])[None, None, :, :].expand(3, -1, -1, -1).contiguous()
        self.register_buffer('ssim_window', window)
        
        # 4. Final step: trigger first phase switch to move active models to GPU
        current_epoch = self.epoch
        self.get_training_phase(current_epoch)
        
        config.logger.info(f"💓 Trainer initialized. Mode: Surgical Phase-based GPU Loading.")

    def register_buffer(self, name, tensor):
        if not hasattr(self, '_buffers'): self._buffers = {}
        self._buffers[name] = tensor.to(config.DEVICE)
        setattr(self, name, self._buffers[name])

    def perceptual_loss(self, recon, target):
        if not hasattr(self, 'vgg'):
            try:
                import torchvision.models as tv_models
                self.vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:16]
                if self.phase == 1: self.vgg.to(config.DEVICE)
                for p in self.vgg.parameters(): p.requires_grad = False
                self.vgg.eval()
            except: return torch.tensor(0.0, device=config.DEVICE)
        
        if self.phase != 1: return torch.tensor(0.0, device=config.DEVICE)
        
        vgg_device = next(self.vgg.parameters()).device
        if self.phase == 1 and vgg_device.type == 'cpu': self.vgg.to(config.DEVICE)
        elif self.phase != 1 and vgg_device.type != 'cpu': self.vgg.to('cpu')
            
        mean, std = self.vgg_mean.to(recon.dtype), self.vgg_std.to(recon.dtype)
        recon_norm = ((((recon + 1) / 2) - mean) / std).contiguous()
        target_norm = ((((target + 1) / 2) - mean) / std).contiguous()
        self.vgg.to(recon.dtype)
        recon_feat = self.vgg(recon_norm)
        with torch.no_grad(): target_feat = self.vgg(target_norm)
        return F.mse_loss(recon_feat, target_feat)

    def _switch_to_phase(self, new_phase: int):
        """Handle transition between training phases with adaptive VRAM management."""
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        # Check available memory to decide if we can afford both on GPU
        can_afford_both = False
        if config.DEVICE.type == 'cuda':
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                # Need ~2.5GB free for safe double-model training
                can_afford_both = free_mem > 2.5 * 1024**3
            except:
                can_afford_both = False

        if new_phase == 1:
            self.vae.to(config.DEVICE)
            if can_afford_both:
                self.drift.to(config.DEVICE)
                config.logger.info("🚀 VRAM: Both networks active on GPU (Memory sufficient).")
            else:
                self.drift.to('cpu')
                config.logger.info("⚡ VRAM: Surgical Mode - Drift offloaded to CPU (Memory low).")
                
            self.contrastive_criterion.to(config.DEVICE)
            if hasattr(self, 'vgg'): self.vgg.to(config.DEVICE)
        else:
            self.vae.to(config.DEVICE)
            self.drift.to(config.DEVICE)
            self.contrastive_criterion.to('cpu')
            if hasattr(self, 'vgg'): self.vgg.to('cpu')
            config.logger.info("⚡ VRAM: Core networks on GPU. Aux modules offloaded.")
            
        self.vae.train() if new_phase != 2 else self.vae.train() # VAE train mode for encoding
        self.drift.train() if new_phase >= 2 else self.drift.eval()

        if new_phase == 1:
            for n, p in self.vae.named_parameters(): p.requires_grad = ("lora_" in n if config.USE_LORA else True)
            self.vae_ref = None
        elif new_phase == 2:
            self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
            dm.flexible_load(self.vae_ref, self.vae.state_dict())
            self.vae_ref.eval()
            for p in self.vae_ref.parameters(): p.requires_grad = False
            for n, p in self.vae.named_parameters():
                is_enc = any(k in n for k in ['enc_', 'label_emb', 'z_mean', 'z_logvar', 'source_emb', 'cond_proj'])
                p.requires_grad = (is_enc and ("lora_" in n if config.USE_LORA else True))
            self.opt_vae.state.clear()
            self.phase2_start_epoch = self.epoch
        elif new_phase == 3:
            for n, p in self.vae.named_parameters(): p.requires_grad = ("lora_" in n if config.USE_LORA else True)
            if not self.vae_ref:
                self.vae_ref = models.LabelConditionedVAE().to(config.DEVICE)
                dm.flexible_load(self.vae_ref, self.vae.state_dict())
                self.vae_ref.eval()
            config.logger.info("Phase 3: Joint fine-tuning enabled.")

    def get_training_phase(self, epoch):
        phase = set_training_phase(epoch)
        if not hasattr(self, 'phase') or phase != self.phase:
            if hasattr(self, 'phase'): config.logger.info(f" Phase changed from {self.phase} to {phase} at epoch {epoch+1}")
            else: config.logger.info(f" Initializing training at Phase {phase}")
            self.phase = phase
            self._switch_to_phase(phase)
        return self.phase

    def ssim_loss(self, x, y):
        x_01, y_01 = (x + 1) / 2, (y + 1) / 2
        C1, C2 = 0.01**2, 0.03**2
        mu_x = F.conv2d(x_01, self.ssim_window, padding=5, groups=3)
        mu_y = F.conv2d(y_01, self.ssim_window, padding=5, groups=3)
        sigma_x_sq = F.conv2d(x_01*x_01, self.ssim_window, padding=5, groups=3) - mu_x.pow(2)
        sigma_y_sq = F.conv2d(y_01*y_01, self.ssim_window, padding=5, groups=3) - mu_y.pow(2)
        sigma_xy = F.conv2d(x_01*y_01, self.ssim_window, padding=5, groups=3) - mu_x*mu_y
        ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x.pow(2) + mu_y.pow(2) + C1)*(sigma_x_sq.clamp(min=0) + sigma_y_sq.clamp(min=0) + C2))
        return 1 - ssim_map.mean()

    def compute_loss(self, batch: Dict, phase: int = 1, batch_idx: int = 0) -> Dict:
        images = batch['image'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE)
        text_bytes = batch.get('text_bytes').to(config.DEVICE) if batch.get('text_bytes') is not None else None
        source_id = batch.get('source_id').to(config.DEVICE) if batch.get('source_id') is not None else None

        if phase == 1:
            recon, mu, logvar = self.vae(images, labels, text_bytes=text_bytes, source_id=source_id)
            c_loss = self.contrastive_criterion(self.vae.image_proj(mu.flatten(1)), self.vae.text_encoder(text_bytes)) if config.USE_PROJECTION_HEADS and text_bytes is not None else torch.tensor(0.0, device=config.DEVICE)
            kl_loss = kl_divergence_spatial(mu, logvar) * config.KL_WEIGHT * min(1.0, self.epoch / config.KL_ANNEALING_EPOCHS)
            recon_loss = F.l1_loss(recon, images) * config.RECON_WEIGHT + config.PERCEPTUAL_WEIGHT * self.perceptual_loss(recon, images)
            ssim_loss = self.ssim_loss(recon, images) * config.SSIM_WEIGHT if config.SSIM_WEIGHT > 0 else torch.tensor(0.0, device=config.DEVICE)
            edge_loss = (F.mse_loss(torch.abs(recon[:,:,:,1:]-recon[:,:,:,:-1]), torch.abs(images[:,:,:,1:]-images[:,:,:,:-1])) + F.mse_loss(torch.abs(recon[:,:,1:,:]-recon[:,:,:-1,:]), torch.abs(images[:,:,1:,:]-images[:,:,:-1,:]))) * config.EDGE_WEIGHT
            div_loss = self.vae.diversity_loss * config.DIVERSITY_WEIGHT if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)
            total = recon_loss + kl_loss + ssim_loss + edge_loss + total_variation_loss(recon, config.TV_WEIGHT) + div_loss + c_loss * config.CONTRASTIVE_WEIGHT
            return {'total': total, 'recon': recon_loss.item(), 'kl': kl_loss.item(), 'diversity': div_loss.item(), 'contrastive': c_loss.item(), 'ssim_loss': ssim_loss.item(), 'snr': calc_snr(images, recon)}
        else:
            with torch.no_grad(): mu_ref, _ = self.vae_ref.encode(images, labels, text_bytes, source_id)
            mu, logvar = self.vae.encode(images, labels, text_bytes, source_id)
            z1 = mu + torch.exp(0.5*logvar) * torch.randn_like(logvar) * (config.TEMPERATURE_START + (config.TEMPERATURE_END-config.TEMPERATURE_START)*(self.epoch/config.EPOCHS))
            z0 = torch.randn_like(z1) * config.CST_COEF_GAUSSIAN_PRIO
            t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            zt = (1 - t.reshape(-1,1,1,1)) * z0 + t.reshape(-1,1,1,1) * z1
            target = z1 - z0
            pred = self.drift(zt, t, labels, text_bytes, source_id)
            drift_loss = F.huber_loss(pred, target) * config.DRIFT_WEIGHT
            consistency_loss = F.mse_loss(mu, mu_ref) * config.CONSISTENCY_WEIGHT
            if phase == 3:
                recon_p3 = self.vae.decode(mu, labels, text_bytes, source_id)
                recon_loss_p3 = F.l1_loss(recon_p3, images) * config.RECON_WEIGHT * 0.1
                total = drift_loss + consistency_loss + recon_loss_p3 + (self.vae._channel_diversity_loss(mu) * config.DIVERSITY_WEIGHT)
                return {'total': total, 'drift': drift_loss.item(), 'consistency': consistency_loss.item(), 'recon_p3': recon_loss_p3.item(), 'mu_std': mu.std().item()}
            else:
                return {'total': drift_loss + consistency_loss, 'drift': drift_loss.item(), 'consistency': consistency_loss.item(), 'mu_std': mu.std().item()}

    def train_epoch(self) -> Dict:
        phase = self.get_training_phase(self.epoch)
        self.vae.train()
        self.drift.train() if phase >= 2 else self.drift.eval()
        pbar = tqdm(self.loader, desc=f"Epoch {self.epoch+1}") if TQDM_AVAILABLE else self.loader
        epoch_losses = defaultdict(float)
        count = 0
        for idx, batch in enumerate(pbar):
            try:
                # 1. Forward Pass
                if self.scaler:
                    with torch.amp.autocast('cuda', enabled=config.USE_AMP, dtype=config.DTYPE_AMP):
                        loss_dict = self.compute_loss(batch, phase, idx)
                else:
                    loss_dict = self.compute_loss(batch, phase, idx)
                
                # 2. Backward Pass
                self.opt_vae.zero_grad(set_to_none=True)
                self.opt_drift.zero_grad(set_to_none=True)
                
                if self.scaler:
                    self.scaler.scale(loss_dict['total']).backward()
                    
                    # 3. Surgical Unscale and Clip (Phase-aware)
                    if phase == 1:
                        self.scaler.unscale_(self.opt_vae)
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        self.scaler.step(self.opt_vae)
                    else:
                        # Phases 2 and 3: Both might have gradients
                        self.scaler.unscale_(self.opt_vae)
                        self.scaler.unscale_(self.opt_drift)
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP*2)
                        self.scaler.step(self.opt_vae)
                        self.scaler.step(self.opt_drift)
                    
                    # 4. Final Scale Update
                    self.scaler.update()
                else:
                    loss_dict['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                    torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP*2)
                    self.opt_vae.step()
                    self.opt_drift.step()

                # Metrics logging
                for k, v in loss_dict.items():
                    if isinstance(v, (int, float)): epoch_losses[k] += v
                    elif isinstance(v, torch.Tensor): epoch_losses[k] += v.item()
                count += 1
                
            except Exception as e:
                # CRITICAL: If an optimization step failed, the scaler might be in a bad state.
                # We must update it to reset internal flags even on failure.
                if self.scaler:
                    try:
                        self.scaler.update()
                    except:
                        # If update fails too, recreate the scaler to be safe
                        if config.DEVICE.type == 'cuda':
                            self.scaler = torch.cuda.amp.GradScaler()
                
                config.logger.error(f"Batch {idx} error: {e}")
                if "out of memory" in str(e): 
                    torch.cuda.empty_cache()
                continue
        
        self.epoch += 1
        return {k: v/count for k, v in epoch_losses.items()} if count > 0 else {}

    def save_checkpoint(self, is_best=False, is_best_overall=False): return dm.save_checkpoint(self, is_best, is_best_overall)
    def load_checkpoint(self, path=None): return dm.load_checkpoint(self, path)
    def load_for_inference(self, path=None): return dm.load_for_inference(self, path)
    
    def generate_reconstructions(self, batch=None):
        self.vae.eval()
        # Get the device the VAE is actually on (it might be CPU)
        vae_device = next(self.vae.parameters()).device
        
        batch = batch or next(iter(self.loader))
        imgs = batch['image'][:8].to(vae_device)
        lbls = batch['label'][:8].to(vae_device)
        
        text_bytes = None
        if batch.get('text_bytes') is not None:
            text_bytes = batch.get('text_bytes')[:8].to(vae_device)
            
        with torch.no_grad():
            recon, _, _ = self.vae(imgs, lbls, text_bytes=text_bytes)
        
        grid_path = config.DIRS["samples"] / f"recon_ep{self.epoch}.png"
        vutils.save_image(torch.cat([(imgs+1)/2, (recon+1)/2], dim=0), grid_path, nrow=8)

    def generate_samples(self, labels=None, num_samples=8):
        # Determine devices
        vae_device = next(self.vae.parameters()).device
        drift_device = next(self.drift.parameters()).device
        
        # If in Phase 1, we might skip samples because Drift is on CPU and untrained
        if self.phase == 1 and drift_device.type == 'cpu':
            config.logger.info("⏩ Skipping samples in Phase 1 (Drift network is offloaded to CPU).")
            return None

        self.vae.eval()
        self.drift.eval()
        
        labels = labels or [i % 10 for i in range(num_samples)]
        lbl_t = torch.tensor(labels, device=drift_device)
        
        with torch.no_grad():
            # Initial noise on drift device
            z = torch.randn(num_samples, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=drift_device) * config.CST_COEF_GAUSSIAN_PRIO
            
            # ODE Integration
            for i in range(config.DEFAULT_STEPS):
                t = torch.full((num_samples, 1), i / config.DEFAULT_STEPS, device=drift_device)
                z = z + self.drift(z, t, lbl_t) * (1.0 / config.DEFAULT_STEPS)
            
            # Move to VAE device for decoding
            z = z.to(vae_device)
            lbl_t = lbl_t.to(vae_device)
            imgs = torch.clamp(self.vae.decode(z, lbl_t), -1, 1)
            
        grid_path = config.DIRS["samples"] / f"gen_ep{self.epoch}.png"
        vutils.save_image((imgs + 1)/2, grid_path, nrow=4)
        return grid_path
