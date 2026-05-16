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

def calculate_sharpness(img: torch.Tensor) -> float:
    """Calculate average gradient magnitude as a proxy for sharpness."""
    if img.shape[1] > 1: # Convert to grayscale if RGB
        gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
    else:
        gray = img.squeeze(1)
    
    dx = torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean()
    dy = torch.abs(gray[:, 1:, :] - gray[:, :-1, :]).mean()
    return (dx + dy).item() / 2.0

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
            if config.DEVICE.type == 'cuda': self.scaler = torch.amp.GradScaler('cuda')
        
        # 1. Initialize shared components for multimodal alignment
        self.shared_text_encoder = None
        self.shared_image_proj = None
        
        if config.USE_NEURAL_TOKENIZER:
            self.shared_text_encoder = models.NeuralTokenizer()
        if config.USE_PROJECTION_HEADS:
            self.shared_image_proj = models.SharedEmbeddingHead(config.LATENT_DIM)
            
        # 2. Initialize models on CPU to avoid startup OOM
        self.vae = models.LabelConditionedVAE(
            text_encoder=self.shared_text_encoder, 
            image_proj=self.shared_image_proj
        )
        self.drift = models.LabelConditionedDrift(
            text_encoder=self.shared_text_encoder, 
            image_proj=self.shared_image_proj
        )

        # EMA shadow models (kept on CPU until needed)
        if config.USE_EMA:
            import copy
            self.ema_vae = copy.deepcopy(self.vae)
            self.ema_drift = copy.deepcopy(self.drift)
            
            # CRITICAL: Re-link EMA shared components to maintain consistency
            if config.USE_NEURAL_TOKENIZER:
                self.ema_vae.text_encoder = self.ema_drift.text_encoder = self.shared_text_encoder
            if config.USE_PROJECTION_HEADS:
                self.ema_vae.image_proj = self.ema_drift.image_proj = self.shared_image_proj
                
            for p in self.ema_vae.parameters(): p.requires_grad = False
            for p in self.ema_drift.parameters(): p.requires_grad = False
        else:
            self.ema_vae = None
            self.ema_drift = None

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
        self.register_buffer('vgg_mean', torch.tensor(config.VGG_NORM_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor(config.VGG_NORM_STD).reshape(1, 3, 1, 1))

        window_size = config.SSIM_WINDOW_SIZE
        gauss = torch.exp(-(torch.arange(window_size, dtype=torch.float32) - window_size//2)**2 / (2 * config.SSIM_SIGMA**2))
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

    def _update_ema(self):
        """Update EMA shadow weights after each optimizer step."""
        if not config.USE_EMA or self.ema_vae is None:
            return
        decay = config.EMA_DECAY
        for p_ema, p in zip(self.ema_vae.parameters(), self.vae.parameters()):
            p_ema.data.mul_(decay).add_(p.data.to(p_ema.device), alpha=1 - decay)
        for p_ema, p in zip(self.ema_drift.parameters(), self.drift.parameters()):
            p_ema.data.mul_(decay).add_(p.data.to(p_ema.device), alpha=1 - decay)

    def perceptual_loss(self, recon, target):
        if not hasattr(self, 'vgg'):
            try:
                import torchvision.models as tv_models
                self.vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features[:16]
                if self.phase == 1: self.vgg.to(config.DEVICE)
                for p in self.vgg.parameters(): p.requires_grad = False
                self.vgg.eval()
            except Exception: return torch.tensor(0.0, device=config.DEVICE)
        
        if self.phase != 1: return torch.tensor(0.0, device=config.DEVICE)
        
        vgg_device = next(self.vgg.parameters()).device
        if self.phase == 1 and vgg_device.type == 'cpu': 
            self.vgg.to(config.DEVICE)
        elif self.phase != 1 and vgg_device.type != 'cpu' and config.DEVICE.type == 'cpu':
            self.vgg.to('cpu')
            
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
        is_gpu = config.DEVICE.type != 'cpu'
        
        if config.DEVICE.type == 'cuda':
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                can_afford_both = free_mem > 2.5 * 1024**3
            except:
                can_afford_both = False
        elif is_gpu:
            # For MPS (Apple), XPU (Intel), and DirectML (AMD/Others), 
            # we generally assume unified memory or sufficient VRAM for these specific models.
            can_afford_both = True

        if new_phase == 1:
            self.vae.to(config.DEVICE)
            if not is_gpu:
                self.drift.to(config.DEVICE)
                config.logger.info(f"💻 Device: {config.DEVICE.type.upper()} Mode.")
            elif can_afford_both:
                self.drift.to(config.DEVICE)
                config.logger.info(f"🚀 VRAM: Both networks active on {config.DEVICE.type.upper()} (Memory sufficient).")
            else:
                self.drift.to('cpu')
                config.logger.info("⚡ VRAM: Surgical Mode - Drift offloaded to CPU (Memory low).")
                
            self.contrastive_criterion.to(config.DEVICE)
            if hasattr(self, 'vgg'): self.vgg.to(config.DEVICE)
        else:
            self.vae.to(config.DEVICE)
            self.drift.to(config.DEVICE)
            if is_gpu:
                # Keep on device for MPS/AMD/Intel, or if CUDA has enough memory
                target_aux_device = config.DEVICE if (config.DEVICE.type != 'cuda' or can_afford_both) else 'cpu'
                self.contrastive_criterion.to(target_aux_device)
                if hasattr(self, 'vgg'): self.vgg.to(target_aux_device)
                config.logger.info(f"🚀 VRAM: Core networks on {config.DEVICE.type.upper()}.")
            else:
                config.logger.info(f"💻 Device: {config.DEVICE.type.upper()} Mode.")
            
        self.vae.train()  # VAE always in train mode; encoder used with gradients in all phases
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

        # Multimodal alignment loss (Shared across all phases)
        c_loss = torch.tensor(0.0, device=config.DEVICE)
        if config.USE_NEURAL_TOKENIZER and config.USE_PROJECTION_HEADS and text_bytes is not None:
            # Note: We use the VAE's encoder output (mu) for image features
            mu_flat, _ = self.vae.encode(images, labels, text_bytes, source_id)
            c_loss = self.contrastive_criterion(
                self.vae.image_proj(mu_flat.flatten(1)), 
                self.vae.text_encoder(text_bytes)
            )

        if phase == 1:
            recon, mu, logvar = self.vae(images, labels, text_bytes=text_bytes, source_id=source_id)
            kl_loss = kl_divergence_spatial(mu, logvar) * config.KL_WEIGHT * min(1.0, self.epoch / config.KL_ANNEALING_EPOCHS)
            recon_loss = F.l1_loss(recon, images) * config.RECON_WEIGHT + config.PERCEPTUAL_WEIGHT * self.perceptual_loss(recon, images)
            ssim_loss = self.ssim_loss(recon, images) * config.SSIM_WEIGHT if config.SSIM_WEIGHT > 0 else torch.tensor(0.0, device=config.DEVICE)
            edge_loss = (F.mse_loss(torch.abs(recon[:,:,:,1:]-recon[:,:,:,:-1]), torch.abs(images[:,:,:,1:]-images[:,:,:,:-1])) + F.mse_loss(torch.abs(recon[:,:,1:,:]-recon[:,:,:-1,:]), torch.abs(images[:,:,1:,:]-images[:,:,:-1,:]))) * config.EDGE_WEIGHT
            div_loss = self.vae.diversity_loss * config.DIVERSITY_WEIGHT if self.vae.diversity_loss is not None else torch.tensor(0.0, device=config.DEVICE)
            
            total = recon_loss + kl_loss + ssim_loss + edge_loss + total_variation_loss(recon, config.TV_WEIGHT) + div_loss + c_loss * config.CONTRASTIVE_WEIGHT
            
            sharpness = calculate_sharpness(recon.detach())
            return {'total': total, 'recon': recon_loss.item(), 'kl': kl_loss.item(), 'diversity': div_loss.item(), 'contrastive': c_loss.item(), 'ssim_loss': ssim_loss.item(), 'snr': calc_snr(images, recon), 'sharpness': sharpness}
        else:
            with torch.no_grad(): 
                mu_ref, _ = self.vae_ref.encode(images, labels, text_bytes, source_id)
            
            mu, logvar = self.vae.encode(images, labels, text_bytes, source_id)
            
            # CRITICAL: Detach latents for Drift target to prevent VAE corruption
            mu_for_drift = mu.detach()
            logvar_for_drift = logvar.detach()
            
            temp_anneal = config.TEMPERATURE_START + (config.TEMPERATURE_END - config.TEMPERATURE_START) * (self.epoch / config.EPOCHS)
            z1 = mu_for_drift + torch.exp(0.5 * logvar_for_drift) * torch.randn_like(logvar_for_drift) * temp_anneal
            
            z0 = torch.randn_like(z1) * config.CST_COEF_GAUSSIAN_PRIO
            t = torch.rand(images.shape[0], 1, device=config.DEVICE)
            t_reshaped = t.reshape(-1, 1, 1, 1).contiguous()
            
            zt = (1 - t_reshaped) * z0 + t_reshaped * z1
            target = z1 - z0
            
            pred = self.drift(zt, t, labels, text_bytes, source_id)
            
            # Restore Time-weighted loss for stability
            time_weights = 1.0 + config.TIME_WEIGHT_FACTOR * t_reshaped
            drift_loss = F.huber_loss(pred * time_weights, target * time_weights, delta=1.0) * config.DRIFT_WEIGHT
            
            consistency_loss = F.mse_loss(mu, mu_ref) * config.CONSISTENCY_WEIGHT
            
            sharpness = 0.0
            if phase == 3:
                recon_p3 = self.vae.decode(mu, labels, text_bytes, source_id)
                # Increase Phase 3 recon scale to maintain integrity
                recon_loss_p3 = F.l1_loss(recon_p3, images) * config.RECON_WEIGHT * 0.5
                total = drift_loss + consistency_loss + recon_loss_p3 + (self.vae._channel_diversity_loss(mu) * config.DIVERSITY_WEIGHT) + c_loss * config.CONTRASTIVE_WEIGHT
                
                sharpness = calculate_sharpness(recon_p3.detach())
                return {
                    'total': total, 
                    'drift': drift_loss.item(), 
                    'consistency': consistency_loss.item(), 
                    'recon_p3': recon_loss_p3.item(), 
                    'contrastive': c_loss.item(),
                    'mu_std': mu.std().item(),
                    'sharpness': sharpness
                }
            else:
                total = drift_loss + consistency_loss + c_loss * config.CONTRASTIVE_WEIGHT
                return {
                    'total': total, 
                    'drift': drift_loss.item(), 
                    'consistency': consistency_loss.item(), 
                    'contrastive': c_loss.item(),
                    'mu_std': mu.std().item(),
                    'sharpness': sharpness
                }

    def train_epoch(self) -> Dict:
        phase = self.get_training_phase(self.epoch)
        self.vae.train()
        self.drift.train() if phase >= 2 else self.drift.eval()
        pbar = tqdm(self.loader, desc=f"Epoch {self.epoch+1}") if TQDM_AVAILABLE else self.loader
        epoch_losses = defaultdict(float)
        count = 0
        for idx, batch in enumerate(pbar):
            # Update VAE's internal epoch for adaptive diversity
            if hasattr(self.vae, 'current_epoch'):
                self.vae.current_epoch = self.epoch
                
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
                    self._update_ema()
                else:
                    loss_dict['total'].backward()
                    if phase == 1:
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        self.opt_vae.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), config.GRAD_CLIP)
                        torch.nn.utils.clip_grad_norm_(self.drift.parameters(), config.GRAD_CLIP * config.DRIFT_GRAD_CLIP_FACTOR * 2)
                        self.opt_vae.step()
                        self.opt_drift.step()
                    self._update_ema()

                # Metrics logging
                current_score = composite_score(loss_dict, phase)
                epoch_losses['composite_score'] += current_score
                
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
                            self.scaler = torch.amp.GradScaler('cuda')
                
                config.logger.error(f"Batch {idx} error: {e}")
                if "out of memory" in str(e): 
                    torch.cuda.empty_cache()
                continue
        
        self.epoch += 1
        return {k: v/count for k, v in epoch_losses.items()} if count > 0 else {}

    def save_checkpoint(self, is_best=False, is_best_overall=False): return dm.save_checkpoint(self, is_best, is_best_overall)
    def load_checkpoint(self, path=None): return dm.load_checkpoint(self, path)
    def load_for_inference(self, path=None): return dm.load_for_inference(self, path)
    
    def load_from_snapshot(self, snapshot_path: Path, load_vae: bool = True, load_drift: bool = True, phase: Optional[int] = None) -> bool:
        """Load model state from a snapshot file."""
        try:
            snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            
            # Load VAE if requested and available
            if load_vae:
                if 'model_state' in snapshot:
                    dm.flexible_load(self.vae, snapshot['model_state'])
                    if 'optimizer_state' in snapshot:
                        self.opt_vae.load_state_dict(snapshot['optimizer_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'vae' and 'model_state' in snapshot:
                    dm.flexible_load(self.vae, snapshot['model_state'])
                    config.logger.info(f"✅ Loaded VAE from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                else:
                    config.logger.warning("No VAE state found in snapshot")
            
            # Load Drift if requested and available
            if load_drift:
                if 'drift_state' in snapshot:
                    dm.flexible_load(self.drift, snapshot['drift_state'])
                    if 'opt_drift_state' in snapshot:
                        self.opt_drift.load_state_dict(snapshot['opt_drift_state'])
                    config.logger.info(f"✅ Loaded Drift from snapshot (epoch {snapshot.get('epoch', 'unknown')})")
                elif snapshot.get('model_type') == 'drift' and 'drift_state' in snapshot:
                    dm.flexible_load(self.drift, snapshot['drift_state'])
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
        
        # Helper to bake spectral norm into weights for quantization
        def bake_spectral_norm(model):
            import copy
            from torch.nn.utils import remove_spectral_norm
            
            # Use deepcopy to avoid modifying the live model
            model_copy = copy.deepcopy(model)
            for m in model_copy.modules():
                try:
                    remove_spectral_norm(m)
                except (ValueError, AttributeError):
                    pass
            return model_copy

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

        # Wrapper class to export ONLY the decoder (generator) part of the VAE.
        # When neural tokenizer is active the graph accepts a pre-computed float32
        # text_embedding so the NeuralTokenizer (byte_embedding Gather + CNN) never
        # appears in the exported graph.  source_id is folded to constant 0 so the
        # source_emb Gather is also eliminated by do_constant_folding + onnxsim.
        if config.USE_NEURAL_TOKENIZER:
            class VAEGenerator(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae
                def forward(self, z, text_embedding):
                    # Constant source_id=0: do_constant_folding folds source_emb(0) → no Gather
                    source_id = torch.zeros(1, dtype=torch.long, device=z.device)
                    dummy_labels = torch.zeros(1, dtype=torch.long, device=z.device)
                    return self.vae.decode(z, dummy_labels, text_emb=text_embedding, source_id=source_id)

            class DriftWrapper(torch.nn.Module):
                """Drift with CFG baked in; source_id folded to 0 constant.
                out = uncond + cfg_scale * (cond - uncond)
                cfg_scale is float32[1] so inference code can control it.
                """
                def __init__(self, drift):
                    super().__init__()
                    self.drift = drift

                def forward(self, z, t, text_embedding, cfg_scale):
                    if t.dim() == 1:
                        t = t.unsqueeze(-1)
                    t_emb = self.drift.time_mlp(t)
                    # Constant source_id=0: do_constant_folding folds source_emb(0) → no Gather
                    source_id = torch.zeros(1, dtype=torch.long, device=z.device)
                    dummy_labels = torch.zeros(1, dtype=torch.long, device=z.device)

                    cond = self.drift._forward_with_emb(z, t, dummy_labels, text_embedding, t_emb, source_id)
                    uncond_text_emb = torch.zeros(1, config.TEXT_EMBEDDING_DIM, device=z.device)
                    uncond = self.drift._forward_with_emb(z, t, dummy_labels, uncond_text_emb, t_emb, source_id)

                    # reshape avoids Gather(cfg_scale, 0) — broadcast multiply instead
                    return uncond + cfg_scale.reshape(-1, 1, 1, 1) * (cond - uncond)
        else:
            class VAEGenerator(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.vae = vae
                def forward(self, z, labels, text_bytes, source_id):
                    return self.vae.decode(z, labels, text_bytes=text_bytes, source_id=source_id)
            class DriftWrapper(torch.nn.Module):
                def __init__(self, drift):
                    super().__init__()
                    self.drift = drift
                def forward(self, z, t, labels, text_bytes, source_id):
                    return self.drift(z, t, labels, text_bytes=text_bytes, source_id=source_id)

        try:
            # --- Set export mode for VAE and Drift ---
            self.vae.eval()
            self.drift.eval()
            self.vae.apply(lambda m: set_export_mode(m, True))
            self.drift.apply(lambda m: set_export_mode(m, True))
            
            # Create baked copies for export to ensure static weights (essential for quantization)
            vae_for_export = bake_spectral_norm(self.vae).to(config.DEVICE)
            drift_for_export = bake_spectral_norm(self.drift).to(config.DEVICE)

            # Dummy inputs for export
            dummy_z = torch.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE)
            dummy_label = torch.tensor([0], device=config.DEVICE, dtype=torch.long)
            dummy_source = torch.tensor([0], device=config.DEVICE, dtype=torch.long)
            dummy_text = torch.zeros(1, config.MAX_TEXT_BYTES, device=config.DEVICE, dtype=torch.long)
            
            gen_path = config.DIRS["onnx"] / "generator.onnx"
            vae_gen = VAEGenerator(vae_for_export)

            if config.USE_NEURAL_TOKENIZER:
                dummy_text_emb = torch.zeros(1, config.TEXT_EMBEDDING_DIM, device=config.DEVICE, dtype=torch.float32)
                gen_args = (dummy_z, dummy_text_emb)
                gen_input_names = ['z', 'text_embedding']
                gen_dynamic_axes = {
                    'z': {0: 'batch_size'},
                    'text_embedding': {0: 'batch_size'},
                    'reconstruction': {0: 'batch_size'}
                }
            else:
                gen_args = (dummy_z, dummy_label, dummy_text, dummy_source)
                gen_input_names = ['z', 'label', 'text_bytes', 'source_id']
                gen_dynamic_axes = {
                    'z': {0: 'batch_size'},
                    'label': {0: 'batch_size'},
                    'text_bytes': {0: 'batch_size'},
                    'source_id': {0: 'batch_size'},
                    'reconstruction': {0: 'batch_size'}
                }

            with torch.no_grad():
                torch.onnx.export(
                    vae_gen, gen_args, str(gen_path),
                    export_params=True,
                    opset_version=config.ONNX_OPSET_VERSION,
                    do_constant_folding=True,
                    dynamo=False,
                    input_names=gen_input_names,
                    output_names=['reconstruction'],
                    dynamic_axes=gen_dynamic_axes,
                )
            merge_external_data(gen_path)
            config.logger.info(f"Generator exported to {gen_path} (inputs: {gen_input_names})")

            # Export Drift
            dummy_t = torch.tensor([[0.5]], device=config.DEVICE)
            drift_path = config.DIRS["onnx"] / "drift.onnx"
            drift_model = DriftWrapper(drift_for_export)

            if config.USE_NEURAL_TOKENIZER:
                dummy_text_emb = torch.zeros(1, config.TEXT_EMBEDDING_DIM, device=config.DEVICE, dtype=torch.float32)
                dummy_cfg = torch.tensor([config.CFG_SCALE], device=config.DEVICE)
                drift_args = (dummy_z, dummy_t, dummy_text_emb, dummy_cfg)
                drift_input_names = ['z', 't', 'text_embedding', 'cfg_scale']
                drift_dynamic_axes = {
                    'z': {0: 'batch_size'},
                    't': {0: 'batch_size'},
                    'text_embedding': {0: 'batch_size'},
                    'drift': {0: 'batch_size'}
                }
            else:
                drift_args = (dummy_z, dummy_t, dummy_label, dummy_text, dummy_source)
                drift_input_names = ['z', 't', 'label', 'text_bytes', 'source_id']
                drift_dynamic_axes = {
                    'z': {0: 'batch_size'},
                    't': {0: 'batch_size'},
                    'label': {0: 'batch_size'},
                    'text_bytes': {0: 'batch_size'},
                    'source_id': {0: 'batch_size'},
                    'drift': {0: 'batch_size'}
                }

            with torch.no_grad():
                torch.onnx.export(
                    drift_model, drift_args, str(drift_path),
                    export_params=True,
                    opset_version=config.ONNX_OPSET_VERSION,
                    do_constant_folding=True,
                    dynamo=False,
                    input_names=drift_input_names,
                    output_names=['drift'],
                    dynamic_axes=drift_dynamic_axes,
                )
            merge_external_data(drift_path)
            config.logger.info(f"Drift exported to {drift_path} (inputs: {drift_input_names})")

            # --- Export label embeddings for browser-side inference ---
            try:
                import json
                embeddings = {}
                config.logger.info("Exporting label embeddings to JSON for browser support...")
                with torch.no_grad():
                    for i in range(10):
                        desc = dm.CLASS_DESCRIPTIONS[i]
                        text_bytes = torch.tensor([dm.text_to_bytes(desc)], device=config.DEVICE)
                        if config.USE_NEURAL_TOKENIZER:
                            emb = self.vae.text_encoder(text_bytes)
                        else:
                            emb = self.vae.label_emb(torch.tensor([i], device=config.DEVICE))
                        embeddings[str(i)] = emb.squeeze().cpu().numpy().tolist()
                    
                    # NULL/Unconditional class
                    uncond_emb = torch.zeros(config.TEXT_EMBEDDING_DIM)
                    embeddings["uncond"] = uncond_emb.numpy().tolist()

                embed_path = config.DIRS["onnx"] / "label_embeddings.json"
                with open(embed_path, 'w') as f:
                    json.dump(embeddings, f)
                config.logger.info(f"Label embeddings saved to {embed_path.name}")
            except Exception as e:
                config.logger.warning(f"Could not export label embeddings: {e}")

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
            
        source_id = None
        if batch.get('source_id') is not None:
            source_id = batch.get('source_id')[:8].to(vae_device)
            
        with torch.no_grad():
            recon, _, _ = self.vae(imgs, lbls, text_bytes=text_bytes, source_id=source_id)
        
        grid_path = config.DIRS["samples"] / f"recon_ep{self.epoch}.png"
        vutils.save_image(torch.cat([(imgs+1)/2, (recon+1)/2], dim=0), grid_path, nrow=8)

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
        self.vae.to(config.DEVICE)
        self.drift.to(config.DEVICE)

        # Use EMA models for generation if available
        vae_to_use = self.ema_vae if (config.USE_EMA and self.ema_vae is not None) else self.vae
        drift_to_use = self.ema_drift if (config.USE_EMA and self.ema_drift is not None) else self.drift
        vae_to_use.eval()
        drift_to_use.eval()
        vae_to_use.to(config.DEVICE)
        drift_to_use.to(config.DEVICE)

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
                drift = drift_to_use(z, t_cur, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)

                # Monitor drift magnitude (adaptive clipping is inside the drift network)
                drift_norm = drift.flatten(start_dim=1).norm(p=2, dim=1).mean().item()

                if method == 'euler':
                    z = z + drift * dt
                elif method == 'heun':
                    k1 = drift
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_pred = z + dt * k1
                    k2 = drift_to_use(z_pred, t_next, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z = z + (dt / 2.0) * (k1 + k2)
                elif method == 'rk4':
                    k1 = drift
                    t_half = torch.full((num_samples, 1), (i + 0.5) * dt, device=config.DEVICE)
                    z_half = z + 0.5 * dt * k1
                    k2 = drift_to_use(z_half, t_half, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z_half2 = z + 0.5 * dt * k2
                    k3 = drift_to_use(z_half2, t_half, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    t_next = torch.full((num_samples, 1), (i + 1) * dt, device=config.DEVICE)
                    z_next = z + dt * k3
                    k4 = drift_to_use(z_next, t_next, labels_tensor, text_bytes=text_bytes_tensor, cfg_scale=cfg_scale, source_id=s_id)
                    z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
                z = torch.clamp(z, -config.ODE_CLAMP_MAX, config.ODE_CLAMP_MAX)   # gentle clamping
                
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
                    drift_at_end = drift_to_use(z, t_one, labels_tensor, text_bytes=text_bytes_tensor, source_id=s_id)
                    
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
                    z = torch.clamp(z, -config.LANGEVIN_MANIFOLD_CLAMP, config.LANGEVIN_MANIFOLD_CLAMP)
                    
                    if (step + 1) % 5 == 0:
                        config.logger.info(f" Langevin Step {step+1}: z_std={z.std():.4f}")

                z = z.detach() # Final cleanup
            config.logger.info(f"Refinement complete: Final z_std={z.std():.4f}")
            
            # Decode
            vae_to_use.set_force_active(True)
            images = vae_to_use.decode(z, labels_tensor, text_bytes=text_bytes_tensor, source_id=s_id)
            vae_to_use.set_force_active(False)
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


# ============================================================
# TRAINING FUNCTION
# ============================================================
