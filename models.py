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
        
        shortcut = self.shortcut(x)
        if out.shape[2:] != shortcut.shape[2:]:
            shortcut = F.interpolate(shortcut, size=out.shape[2:], mode='nearest')
            
        out += shortcut
        return F.silu(out)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
    def forward(self, x):
        b, c, h, w = x.shape
        # Ensure contiguous before reshape
        res = x.reshape(b, c, -1).permute(0, 2, 1).contiguous()
        attn_out, _ = self.mha(res, res, res)
        out = self.ln(res + attn_out)
        # Ensure contiguous before reshape
        return out.permute(0, 2, 1).contiguous().reshape(b, c, h, w)

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
            scale = (self.high - self.low).clamp(min=1e-6).reshape(1, -1, 1, 1)
            shift = self.low.reshape(1, -1, 1, 1)
            return torch.tanh((x - shift) / scale)
        
        if self.training:
            with torch.no_grad():
                # Added .contiguous() to handle MPS layout issues
                flat = x.float().transpose(0, 1).contiguous().reshape(x.shape[1], -1)
                if flat.numel() > 0:
                    l = torch.quantile(flat, self.p_low/100, dim=1)
                    h = torch.quantile(flat, self.p_high/100, dim=1)
                    self.low.mul_(1 - self.m).add_(l, alpha=self.m)
                    self.high.mul_(1 - self.m).add_(h, alpha=self.m)
        
        scale = (self.high - self.low).clamp(min=1e-6).reshape(1, -1, 1, 1)
        shift = self.low.reshape(1, -1, 1, 1)
        return torch.tanh((x - shift) / scale)

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
            # Stronger FiLM-like modulation
            scale_shift = self.label_proj(labels)
            scale, shift = scale_shift.chunk(2, dim=1)
            # Reshape for broadcasting - added .contiguous() for MPS stability
            scale = scale.reshape(-1, scale.shape[1], 1, 1).contiguous()
            shift = shift.reshape(-1, shift.shape[1], 1, 1).contiguous()
            h = h * (1 + scale) + shift
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        skip = self.skip(x)
        if h.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
            
        return self.rescale(skip + h)

# ============================================================
# SUBPIXEL UPSAMPLING (NEW)
# ============================================================
class SubpixelUpsample(nn.Module):
    """Artifact-free upsampling with Bilinear fallback for stability."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1. Main Path: Subpixel Conv
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        
        # 2. Fallback Path: Bilinear (ensures we don't lose progress)
        self.fallback = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # 3. Learnable mix (initialized from config)
        self.mix = nn.Parameter(torch.tensor(config.SUBPIXEL_INITIAL_MIX)) 
        self.act = nn.SiLU()
        
    def forward(self, x):
        subpixel = self.shuffle(self.conv(x))
        bilinear = self.fallback(x)
        
        # Blend the two (starts mostly bilinear to keep training, fades into subpixel)
        alpha = torch.sigmoid(self.mix)
        out = alpha * subpixel + (1 - alpha) * bilinear
        
        return self.act(out)

# ============================================================
# LABEL-CONDITIONED VAE
# ============================================================
class LabelConditionedVAE(nn.Module):
    """VAE with label conditioning for latent space encoding."""
    def __init__(self, free_bits=config.FREE_BITS):
        super().__init__()
        self.free_bits = free_bits
        self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
        self.current_epoch = 0
        self.mu_noise_scale = config.MU_NOISE_SCALE
        
        # Fourier feature channels
        if config.USE_FOURIER_FEATURES:
            self.fourier_channels = len(config.FOURIER_FREQS) * 4
        else:
            self.fourier_channels = 0
        
        # Encoder with 4 stages (96 -> 48 -> 24 -> 12 -> 6)
        self.enc_in = nn.Conv2d(3 + self.fourier_channels, 64, 3, 1, 1)
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2),           # 96 -> 48
            LabelConditionedBlock(128, 128),
            ResidualBlock(128, 256, stride=2),          # 48 -> 24
            SelfAttention(256),
            LabelConditionedBlock(256, 512),
            ResidualBlock(512, 512, stride=2),          # 24 -> 12
            nn.Conv2d(512, 512, 4, 2, 1),               # 12 -> 6
        ])
        self.z_mean = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)
        self.z_logvar = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)

        # Contextual embedding (Source Dataset ID)
        if config.USE_CONTEXT:
            self.source_emb = nn.Embedding(config.NUM_SOURCES, config.CONTEXT_DIM)
            self.cond_proj = nn.Sequential(
                nn.Linear(config.LABEL_EMB_DIM + config.CONTEXT_DIM, config.LABEL_EMB_DIM),
                nn.SiLU(),
                nn.Linear(config.LABEL_EMB_DIM, config.LABEL_EMB_DIM)
            )

        # ========== ENHANCED DECODER WITH SUBPIXEL CONV (4 stages) ==========
        self.dec_in = nn.Conv2d(config.LATENT_CHANNELS, 512, 3, 1, 1)
        self.dec_blocks = nn.ModuleList([
            # Stage 1: 6x6 -> 12x12
            SubpixelUpsample(512, 512),
            LabelConditionedBlock(512, 512),
            SelfAttention(512),

            # Stage 2: 12x12 -> 24x24
            SubpixelUpsample(512, 256),
            LabelConditionedBlock(256, 256),
            SelfAttention(256),

            # Stage 3: 24x24 -> 48x48
            SubpixelUpsample(256, 128),
            LabelConditionedBlock(128, 128),

            # Stage 4: 48x48 -> 96x96
            SubpixelUpsample(128, 64),
            LabelConditionedBlock(64, 64),
        ])

        self.dec_out = nn.Conv2d(64, 3, 3, 1, 1)
        self.diversity_loss = None
    
    def _get_fourier_features(self, x):
        """Generate Fourier features for spatial coordinates."""
        if not config.USE_FOURIER_FEATURES:
            return None
        B, C, H, W = x.shape
        y_coords = torch.linspace(-1, 1, H, device=x.device).reshape(1, 1, H, 1)
        x_coords = torch.linspace(-1, 1, W, device=x.device).reshape(1, 1, 1, W)
        y_grid = y_coords.expand(B, 1, H, W)
        x_grid = x_coords.expand(B, 1, H, W)
        feats = []
        for f in config.FOURIER_FREQS:
            feats.append(torch.sin(np.pi * f * x_grid))
            feats.append(torch.cos(np.pi * f * x_grid))
            feats.append(torch.sin(np.pi * f * y_grid))
            feats.append(torch.cos(np.pi * f * y_grid))
        return torch.cat(feats, dim=1)
    
    def _channel_diversity_loss(self, mu):
        """Compute diversity loss to encourage all channels to be used."""
        channel_stds = mu.std(dim=[0, 2, 3])
        if config.DIVERSITY_ADAPTIVE and hasattr(self, 'current_epoch'):
            progress = min(1.0, self.current_epoch / config.DIVERSITY_ADAPT_EPOCHS)
            target_std = config.DIVERSITY_TARGET_START + progress * (
                config.DIVERSITY_TARGET_END - config.DIVERSITY_TARGET_START
            )
        else:
            target_std = config.DIVERSITY_TARGET_STD
        
        low_penalty = torch.mean(F.relu(target_std - channel_stds)) * config.DIVERSITY_LOW_PENALTY
        high_penalty = torch.mean(F.relu(channel_stds - config.DIVERSITY_MAX_STD)) * config.DIVERSITY_HIGH_PENALTY
        balance_loss = channel_stds.std() * config.DIVERSITY_BALANCE_WEIGHT
        
        if self.training:
            self.diversity_stats = {
                'channel_stds': channel_stds.detach().cpu(),
                'target_std': target_std,
                'low_penalty': low_penalty.item(),
                'high_penalty': high_penalty.item() if isinstance(high_penalty, torch.Tensor) else 0,
                'balance_loss': balance_loss.item()
            }
        return low_penalty + high_penalty + balance_loss
    
    def encode(self, x, labels, source_id=None):
        """Encode images to latent distribution parameters."""
        label_emb = self.label_emb(labels)
        if config.USE_CONTEXT and source_id is not None:
            s_emb = self.source_emb(source_id)
            label_emb = self.cond_proj(torch.cat([label_emb, s_emb], dim=-1))
            
        if config.USE_FOURIER_FEATURES:
            fourier = self._get_fourier_features(x)
            x = torch.cat([x, fourier], dim=1)
        h = self.enc_in(x)
        for block in self.enc_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        mu = self.z_mean(h) * config.LATENT_SCALE
        if self.training:
            mu = mu + torch.randn_like(mu) * self.mu_noise_scale
        if self.training and torch.rand(1).item() < config.CHANNEL_DROPOUT_PROB:
            channel_mask = torch.bernoulli(
                torch.full((mu.shape[0], mu.shape[1], 1, 1), 
                          config.CHANNEL_DROPOUT_SURVIVAL, device=mu.device)
            )
            mu = (mu * channel_mask) / config.CHANNEL_DROPOUT_SURVIVAL
        logvar = torch.clamp(self.z_logvar(h), min=config.LOGVAR_CLAMP_MIN, max=config.LOGVAR_CLAMP_MAX)
        if self.training:
            self.diversity_loss = self._channel_diversity_loss(mu)
        return mu, logvar
    
    def decode(self, z, labels, source_id=None):
        """Decode latents to images with enhanced architecture."""
        label_emb = self.label_emb(labels)
        if config.USE_CONTEXT and source_id is not None:
            s_emb = self.source_emb(source_id)
            label_emb = self.cond_proj(torch.cat([label_emb, s_emb], dim=-1))
            
        h = self.dec_in(z)
        for block in self.dec_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, label_emb)
            else:
                h = block(h)
        h = self.dec_out(h)
        
        # Robust final upsampling to match target image size
        if h.shape[2:] != (config.IMG_SIZE, config.IMG_SIZE):
            h = F.interpolate(h, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear', align_corners=False)
            
        return torch.tanh(h)
    
    def forward(self, x, labels, source_id=None):
        """Forward pass with reparameterization."""
        mu, logvar = self.encode(x, labels, source_id)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return self.decode(z, labels, source_id), mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

# ============================================================================
# FOURIER TIME EMBEDDING (NEW)
# ============================================================================
class FourierTimeEmbed(nn.Module):
    """Fourier features for time conditioning - captures complex temporal dynamics."""
    def __init__(self, dim=128, max_freq=64):
        super().__init__()
        self.dim = dim
        self.register_buffer('freqs', torch.linspace(1, max_freq, dim//2))
    
    def forward(self, t):
        """
        Args:
            t: Time tensor of shape (B, 1) in range [0, 1]
        Returns:
            Time embedding of shape (B, dim)
        """
        t = t * 2 * math.pi  # Scale to [0, 2π]
        sin_emb = torch.sin(t * self.freqs)
        cos_emb = torch.cos(t * self.freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)


# ============================================================
# LABEL-CONDITIONED DRIFT
# ============================================================
class LabelConditionedDrift(nn.Module):
    """Drift network for probability flow ODE."""
    def __init__(self):
        super().__init__()
        # ========== ENHANCED TIME EMBEDDING WITH FOURIER FEATURES ==========
        self.time_mlp = nn.Sequential(
            FourierTimeEmbed(dim=128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        # Rest of initialization remains the same...
        # Label conditioning
        self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
        
        # Contextual embedding (Source Dataset ID)
        if config.USE_CONTEXT:
            self.source_emb = nn.Embedding(config.NUM_SOURCES, config.CONTEXT_DIM)
            self.cond_proj = nn.Sequential(
                nn.Linear(256 + config.LABEL_EMB_DIM + config.CONTEXT_DIM, 128),
                nn.SiLU(),
                nn.Linear(128, 128)
            )
        else:
            self.cond_proj = nn.Sequential(
                nn.Linear(256 + config.LABEL_EMB_DIM, 128),
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
        
        self.mid1 = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        self.mid_attn = SelfAttention(256)
        self.mid2 = LabelConditionedBlock(256, 256, label_dim=128, use_spectral_norm=True)
        
        self.up2_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1)
        )
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
        self._is_exporting = False

    def _set_export_mode(self, is_exporting=True):
        self._is_exporting = is_exporting

    def forward(self, z, t, labels, cfg_scale=1.0, source_id=None):
        """Forward pass - predict drift at time t with CFG and context support."""
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t)

        # Handle Classifier-Free Guidance during inference
        if cfg_scale != 1.0 and not self.training:
            # Predict with conditional labels
            cond_drift = self._forward_internal(z, t, labels, t_emb, source_id)
            
            # Predict with unconditional labels (using class index 0 as placeholder for 'null')
            uncond_labels = torch.zeros_like(labels) 
            uncond_drift = self._forward_internal(z, t, uncond_labels, t_emb, source_id)
            
            return uncond_drift + cfg_scale * (cond_drift - uncond_drift)
            
        return self._forward_internal(z, t, labels, t_emb, source_id)

    def _forward_internal(self, z, t, labels, t_emb, source_id=None):
        """Internal forward pass for a single label and context set."""
        # Label embedding
        label_emb = self.label_emb(labels)

        # Combine embeddings with optional context
        if config.USE_CONTEXT and source_id is not None:
            s_emb = self.source_emb(source_id)
            cond = torch.cat([t_emb, label_emb, s_emb], dim=-1)
        else:
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
        # Added .contiguous() to index and result for MPS stability
        idx_f_flat = idx_floor.reshape(-1).contiguous()
        idx_c_flat = idx_ceil.reshape(-1).contiguous()
        scale_floor = self.time_scales[idx_f_flat]   # (batch,)
        scale_ceil  = self.time_scales[idx_c_flat]    # (batch,)

        # Linear blend
        blended = scale_floor * (1 - frac.reshape(-1)) + scale_ceil * frac.reshape(-1)
        time_scale = blended.reshape(-1, 1, 1, 1).contiguous()                # (batch, 1, 1, 1)
                
        # Gentle clamping instead of tanh
        z = torch.clamp(z, -10, 10)
        
        # U-Net forward
        h = self.head(z)
        d1 = self.down1(h, cond)
        d2 = self.down2_conv(d1)
        d2 = self.down2_block(d2, cond)
        m = self.mid1(d2, cond)
        m = self.mid_attn(m)
        m = self.mid2(m, cond)
        u2 = self.up2_conv(m)
        u2 = self.up2_block(u2, cond)
        
        # Robust skip connection: ensure spatial dimensions match exactly
        if u2.shape[2:] != d1.shape[2:]:
            u2 = F.interpolate(u2, size=d1.shape[2:], mode='nearest')
            
        u1 = self.up1(u2 + d1, cond)
        out = self.tail(u1)
        
        # Scale output (removed tanh to prevent capping drift values)
        out = out * self.output_scale * (1.0 + time_weight.reshape(-1, 1, 1, 1).contiguous())
        out = out * (1.0 + time_scale)
        
        # Update running statistics for adaptive clipping (only in training)
        if self.training and not self._is_exporting:
            with torch.no_grad():
                # Avoid .item() to keep it symbolic/onnx-friendly
                batch_mean = out.mean()
                batch_std = out.std()
                
                # Update running statistics
                self.drift_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.drift_std.mul_(self.momentum).add_(batch_std, alpha=1 - self.momentum)
                self.n_samples.add_(out.shape[0])
        
        # Apply adaptive clipping during inference only
        if not self.training or self._is_exporting:
            threshold = self.get_adaptive_threshold(num_std=3.0)
            # Use out.clamp with tensor threshold
            out = torch.clamp(out, -threshold, threshold)

        return out

    def get_adaptive_threshold(self, num_std=3.0):
        """Get adaptive clipping threshold based on running statistics."""
        # Use torch.where to avoid data-dependent control flow during export
        threshold = torch.abs(self.drift_mean) + num_std * self.drift_std
        default_threshold = torch.tensor(5.0, device=self.n_samples.device)
        return torch.where(self.n_samples < 100, default_threshold, threshold)