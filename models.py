# ============================================================================
# MODEL ARCHITECTURES FOR SCHRÖDINGER BRIDGE
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.silu(out)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
    def forward(self, x):
        b, c, h, w = x.shape
        res = x.view(b, c, -1).permute(0, 2, 1)
        attn_out, _ = self.mha(res, res, res)
        out = self.ln(res + attn_out)
        return out.permute(0, 2, 1).view(b, c, h, w)

# ============================================================
# PERCENTILE RESCALING
# ============================================================
class PercentileRescale(nn.Module):
    """Adaptive percentile-based rescaling to [-1, 1]."""
    
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
            return torch.tanh((x - shift) / scale * 0.9)  # Slightly reduced scale for stability
        
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
        return torch.tanh((x - shift) / scale * 0.9)

# ============================================================
# LABEL-CONDITIONED BLOCK
# ============================================================
class LabelConditionedBlock(nn.Module):
    """Residual block with label conditioning via scale-shift modulation."""
    
    def __init__(self, c_in, c_out, label_dim=config.LABEL_EMB_DIM, use_spectral_norm=False):
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
        
        self.rescale = PercentileRescale(c_out) if config.USE_PERCENTILE else nn.Identity()
        
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

# ============================================================
# LABEL-CONDITIONED VAE
# ============================================================
class LabelConditionedVAE(nn.Module):
    """VAE with label conditioning for latent space encoding."""    
    def __init__(self, free_bits=config.FREE_BITS):
        super().__init__()
        self.free_bits = free_bits
        self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
        self.current_epoch = 0  # For adaptive diversity loss scheduling
        self.mu_noise_scale = config.MU_NOISE_SCALE # For adaptive collapse recovery    

        # Fourier feature channels
        if config.USE_FOURIER_FEATURES:
            self.fourier_channels = len(config.FOURIER_FREQS) * 4
        else:
            self.fourier_channels = 0

        # Doubled from 32 to 64
        self.enc_in = nn.Conv2d(3 + self.fourier_channels, 64, 3, 1, 1)

        # Patched with Residual and Attention upgrades
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2),           # 16x16
            LabelConditionedBlock(128, 128),
            ResidualBlock(128, 256, stride=2),          # 8x8
            SelfAttention(256),
            LabelConditionedBlock(256, 512),
            nn.Conv2d(512, 512, 3, 1, 1),               # Maintain spatial resolution
        ])
        
        # Input channels doubled from 256 to 512
        self.z_mean = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)
        self.z_logvar = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)
        
        # Decoder
        # Input channels doubled from 256 to 512
        self.dec_in = nn.Conv2d(config.LATENT_CHANNELS, 512, 3, 1, 1)
        
        self.dec_blocks = nn.ModuleList([
            # Stage 1: 8x8 -> 16x16
            # In: 512 | Conv out: 1024 | PixelShuffle out: 256
            nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU()), 
            LabelConditionedBlock(256, 256),
            
            # Stage 2: 16x16 -> 32x32
            # In: 256 | Conv out: 512 | PixelShuffle out: 128
            nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU()),  
            LabelConditionedBlock(128, 128),
            
            # Stage 3: 32x32 -> 64x64 (Added to fix resolution mismatch)
            # In: 128 | Conv out: 256 | PixelShuffle out: 64
            nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.PixelShuffle(2), nn.SiLU()),
            LabelConditionedBlock(64, 64),
        ])
        
        # Input channels doubled from 32 to 64
        self.dec_out = nn.Conv2d(64, 3, 3, 1, 1)
        self.diversity_loss = None

    def _get_fourier_features(self, x):
        """
        x: image tensor of shape (B, 3, H, W), values in [-1,1]
        Returns: tensor of shape (B, C_fourier, H, W) or None if disabled.
        """
        if not config.USE_FOURIER_FEATURES:
            return None
        B, C, H, W = x.shape
        # Normalised coordinates in [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
        y_grid = y_coords.expand(B, 1, H, W)   # (B,1,H,W)
        x_grid = x_coords.expand(B, 1, H, W)
        feats = []
        for f in config.FOURIER_FREQS:
            feats.append(torch.sin(np.pi * f * x_grid))
            feats.append(torch.cos(np.pi * f * x_grid))
            feats.append(torch.sin(np.pi * f * y_grid))
            feats.append(torch.cos(np.pi * f * y_grid))
        return torch.cat(feats, dim=1)   # (B, 4*len(FOURIER_FREQS), H, W)

    def _channel_diversity_loss(self, mu):
        """Compute diversity loss to encourage all channels to be used.
        
        Uses configurable parameters from config that can be
        updated dynamically during training.
        """
        channel_stds = mu.std(dim=[0, 2, 3])
        
        # Calculate adaptive target if enabled
        if config.DIVERSITY_ADAPTIVE and hasattr(self, 'current_epoch'):
            progress = min(1.0, self.current_epoch / config.DIVERSITY_ADAPT_EPOCHS)
            target_std = config.DIVERSITY_TARGET_START + progress * (config.DIVERSITY_TARGET_END - config.DIVERSITY_TARGET_START)
        else:
            target_std = config.DIVERSITY_TARGET_STD
        
        # Penalize channels that are too low
        low_penalty = torch.mean(F.relu(target_std - channel_stds)) * config.DIVERSITY_LOW_PENALTY
        
        # Penalize channels that are too high
        high_penalty = torch.mean(F.relu(channel_stds - config.DIVERSITY_MAX_STD)) * config.DIVERSITY_HIGH_PENALTY
        
        # Balance loss - encourage all channels to have similar std
        balance_loss = channel_stds.std() * config.DIVERSITY_BALANCE_WEIGHT
        
        # Store stats for monitoring
        if self.training:
            self.diversity_stats = {
                'channel_stds': channel_stds.detach().cpu(),
                'target_std': target_std,
                'low_penalty': low_penalty.item(),
                'high_penalty': high_penalty.item() if isinstance(high_penalty, torch.Tensor) else 0,
                'balance_loss': balance_loss.item()
            }
        
        return low_penalty + high_penalty + balance_loss

    def encode(self, x, labels):
        """Encode images to latent distribution parameters."""
        label_emb = self.label_emb(labels)
      
        if config.USE_FOURIER_FEATURES:
            fourier = self._get_fourier_features(x)
            x = torch.cat([x, fourier], dim=1)   # (B, 3+4*len(FREQS), H, W)
        
        h = self.enc_in(x)
        
        for block in self.enc_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        
        # Stabilized mean prediction with scaling
        mu = self.z_mean(h) * config.LATENT_SCALE
        
        # Add small noise during training for stability
        if self.training:
            mu = mu + torch.randn_like(mu) * config.mu_noise_scale
        
        # Channel dropout for regularization - using config values
        if self.training and torch.rand(1).item() < config.CHANNEL_DROPOUT_PROB:
            channel_mask = torch.bernoulli(torch.full((mu.shape[0], mu.shape[1], 1, 1), config.CHANNEL_DROPOUT_SURVIVAL, device=mu.device))
            mu = (mu * channel_mask) / config.CHANNEL_DROPOUT_SURVIVAL
            
        logvar = torch.clamp(self.z_logvar(h), min=config.LOGVAR_CLAMP_MIN, max=config.LOGVAR_CLAMP_MAX)
        
        # Track channel diversity loss during training
        if self.training:
            self.diversity_loss = self._channel_diversity_loss(mu)
        
        return mu, logvar

    def decode(self, z, labels):
        """Decode latents to images."""
        label_emb = self.label_emb(labels)
        
        h = self.dec_in(z)
        for block in self.dec_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        
        return torch.tanh(self.dec_out(h))

    def forward(self, x, labels):
        """Forward pass with reparameterization."""
        mu, logvar = self.encode(x, labels)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
            
        return self.decode(z, labels), mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

# ============================================================
# LABEL-CONDITIONED DRIFT
# ============================================================
class LabelConditionedDrift(nn.Module):
    """Drift network for probability flow ODE."""
    
    def __init__(self):
        super().__init__()
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Label conditioning
        self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
        self.cond_proj = nn.Sequential(
            nn.Linear(128 + config.LABEL_EMB_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Time-adaptive scaling
        self.time_weight_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # U-Net like architecture
        self.head = nn.utils.spectral_norm(nn.Conv2d(config.LATENT_CHANNELS, 64, 3, 1, 1))
        
        self.down1 = LabelConditionedBlock(64, 128, label_dim=128, use_spectral_norm=True)
        self.down2_conv = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.down2_block = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        
        self.mid = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        
        self.up2_conv = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2_block = LabelConditionedBlock(128, 128, label_dim=128, use_spectral_norm=True)
        self.up1 = LabelConditionedBlock(128, 64, label_dim=128, use_spectral_norm=True)
        
        self.tail = nn.utils.spectral_norm(nn.Conv2d(64, config.LATENT_CHANNELS, 3, 1, 1))
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.time_scales = nn.Parameter(torch.ones(4) * 0.1)
        
        # Running statistics for adaptive clipping
        self.register_buffer('drift_mean', torch.zeros(1))
        self.register_buffer('drift_std', torch.ones(1))
        self.register_buffer('n_samples', torch.zeros(1))
        self.momentum = 0.99

    def forward(self, z, t, labels):
        """Forward pass - predict drift at time t."""
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Label embedding
        label_emb = self.label_emb(labels)
        
        # Combined conditioning
        cond = torch.cat([t_emb, label_emb], dim=-1)
        cond = self.cond_proj(cond)
        
        # Time-adaptive scaling
        time_weight = self.time_weight_net(t)
       
        # t shape: (batch, 1)
        t_scaled = t * 3.0                     # (batch, 1) in [0, 3]
        idx_floor = torch.floor(t_scaled).long()          # 0,1,2
        idx_ceil = (idx_floor + 1).clamp(max=3)           # 1,2,3
        frac = (t_scaled - idx_floor.float())             # fractional part in [0,1)

        # Gather scales (time_scales is a 1‑D tensor of length 4)
        scale_floor = self.time_scales[idx_floor.view(-1)]   # (batch,)
        scale_ceil  = self.time_scales[idx_ceil.view(-1)]    # (batch,)

        # Linear blend
        blended = scale_floor * (1 - frac.view(-1)) + scale_ceil * frac.view(-1)
        time_scale = blended.view(-1, 1, 1, 1)                # (batch, 1, 1, 1)
                
        # Gentle clamping instead of tanh
        z = torch.clamp(z, -10, 10)
        
        # U-Net forward
        h = self.head(z)
        d1 = self.down1(h, cond)
        d2 = self.down2_conv(d1)
        d2 = self.down2_block(d2, cond)
        m = self.mid(d2, cond)
        u2 = self.up2_conv(m)
        u2 = self.up2_block(u2, cond)
        u1 = self.up1(u2 + d1, cond)
        out = self.tail(u1)
        
        # Scale output
        out = torch.tanh(out) * self.output_scale * (1.0 + time_weight.view(-1, 1, 1, 1))
        out = out * (1.0 + time_scale)
        
        # Update running statistics for adaptive clipping (only in training)
        if self.training:
            with torch.no_grad():
                batch_mean = out.mean().item()
                batch_std = out.std().item()
                batch_size = out.shape[0]
                
                # Update running statistics
                self.drift_mean = self.momentum * self.drift_mean + (1 - self.momentum) * batch_mean
                self.drift_std = self.momentum * self.drift_std + (1 - self.momentum) * batch_std
                self.n_samples += batch_size
        
        # Apply adaptive clipping during inference only
        if not self.training:
            threshold = self.get_adaptive_threshold(num_std=3.0)
            out = torch.clamp(out, -threshold, threshold)

        return out

    def get_adaptive_threshold(self, num_std=3.0):
        """Get adaptive clipping threshold based on running statistics."""
        if self.n_samples < 100:  # Not enough samples yet
            return 5.0
        return abs(self.drift_mean.item()) + num_std * self.drift_std.item()