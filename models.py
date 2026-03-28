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
    """Enhanced Multi-Head Self-Attention for 2D feature maps."""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(in_channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten spatial dims and transpose for MHA: [B, Pixels, C]
        res = x.view(b, c, -1).permute(0, 2, 1).contiguous()
        
        # Self-attention
        attn_out, _ = self.mha(res, res, res)
        out = self.ln(res + attn_out)
        
        # Reshape back to 2D: [B, C, H, W]
        return out.permute(0, 2, 1).contiguous().view(b, c, h, w)

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
# MULTIMODAL CONDITIONING BLOCK
# ============================================================
class MultimodalConditionedBlock(nn.Module):
    """
    Enhanced block that accepts either a discrete label or a high-dimensional context vector.
    Integrates both Multi-Head Self-Attention and Cross-Attention.
    """
    def __init__(self, c_in, c_out, context_dim=config.TEXT_EMBEDDING_DIM, use_spectral_norm=False, use_self_attn=True):
        super().__init__()
        groups = min(8, c_in)
        self.norm1 = nn.GroupNorm(groups, c_in)
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, 3, 1, 1))
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        
        # Context projection for scale/shift (FiLM-style)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, c_out * 2),
            nn.SiLU(),
            nn.Linear(c_out * 2, c_out * 2)
        )
        
        # Self-Attention
        self.self_attn = SelfAttention(c_out) if use_self_attn else nn.Identity()

        # Optional Cross-Attention for deeper fusion if configured
        if config.MULTIMODAL_FUSION == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(c_out, num_heads=4, batch_first=True)
            self.ln_cross = nn.LayerNorm(c_out)
        else:
            self.cross_attn = None

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

    def forward(self, x, context=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        if context is not None:
            # 1. Apply FiLM-style modulation
            if context.dim() == 3:
                global_ctx = context.mean(dim=1)
            else:
                global_ctx = context
                
            scale_shift = self.context_proj(global_ctx)
            scale, shift = scale_shift.chunk(2, dim=1)
            scale = scale.view(-1, scale.shape[1], 1, 1).contiguous()
            shift = shift.view(-1, shift.shape[1], 1, 1).contiguous()
            h = h * (1 + scale) + shift
            
            # 2. Apply Cross-Attention if context is sequential
            if self.cross_attn is not None and context.dim() == 3:
                b, c, hh, ww = h.shape
                h_flat = h.view(b, c, -1).permute(0, 2, 1).contiguous()
                attn_out, _ = self.cross_attn(h_flat, context, context)
                h_flat = self.ln_cross(h_flat + attn_out)
                h = h_flat.permute(0, 2, 1).contiguous().view(b, c, hh, ww)
        
        # 3. Apply Multi-Head Self-Attention
        h = self.self_attn(h)
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        return self.rescale(self.skip(x) + h)

# ============================================================
# TEXT ENCODER
# ============================================================
class TextEncoder(nn.Module):
    """
    Maps text strings (via pre-tokenized IDs or class names) to embeddings.
    For this prototype, it uses a small MLP over a vocabulary.
    """
    def __init__(self, vocab_size=1000, embed_dim=config.TEXT_EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Token IDs [B] or [B, S]
        """
        h = self.embedding(x)
        if h.dim() == 3: # [B, S, D]
            return self.encoder(h)
        return self.encoder(h) # [B, D]

# ============================================================
# CONTEXT ENCODER & DECODER
# ============================================================
class ContextEncoder(nn.Module):
    def __init__(self, label_dim=config.LABEL_EMB_DIM, text_dim=config.TEXT_EMBEDDING_DIM):
        super().__init__()
        self.label_emb = nn.Embedding(config.NUM_CLASSES, label_dim)
        self.label_to_text = nn.Linear(label_dim, text_dim)
        
    def forward(self, labels=None, text_emb=None):
        if text_emb is not None:
            return text_emb
        if labels is not None:
            return self.label_to_text(self.label_emb(labels))
        return None

class ContextDecoder(nn.Module):
    """Maps latent space back to text/label space."""
    def __init__(self, latent_dim=512, text_dim=config.TEXT_EMBEDDING_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim)
        )
        self.classifier = nn.Linear(text_dim, config.NUM_CLASSES)
        
    def forward(self, z):
        # z: [B, C, H, W] -> flatten to [B, D]
        b = z.shape[0]
        h = z.view(b, -1)
        text_emb = self.proj(h)
        logits = self.classifier(text_emb)
        return text_emb, logits

# ============================================================
# MULTIMODAL VAE
# ============================================================
class MultimodalVAE(nn.Module):
    def __init__(self, free_bits=config.FREE_BITS):
        super().__init__()
        self.free_bits = free_bits
        self.context_encoder = ContextEncoder()
        self.context_decoder = ContextDecoder(latent_dim=config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W)
        self.current_epoch = 0
        self.mu_noise_scale = config.MU_NOISE_SCALE
        
        self.fourier_channels = len(config.FOURIER_FREQS) * 4 if config.USE_FOURIER_FEATURES else 0
        
        # Encoder: 96 -> 48 -> 24 -> 12 -> 6
        self.enc_in = nn.Conv2d(3 + self.fourier_channels, 64, 3, 1, 1)
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2),
            MultimodalConditionedBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            SelfAttention(256),
            MultimodalConditionedBlock(256, 512),
            ResidualBlock(512, 512, stride=2),
            SelfAttention(512),
            nn.Conv2d(512, 512, 4, 2, 1),
        ])
        self.z_mean = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)
        self.z_logvar = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)

        # Decoder: 6 -> 12 -> 24 -> 48 -> 96
        self.dec_in = nn.Conv2d(config.LATENT_CHANNELS, 512, 3, 1, 1)
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 512, 3, 1, 1), nn.SiLU()),
            MultimodalConditionedBlock(512, 512),
            SelfAttention(512),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(512, 256, 3, 1, 1), nn.SiLU()),
            MultimodalConditionedBlock(256, 256),
            SelfAttention(256),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256, 128, 3, 1, 1), nn.SiLU()),
            MultimodalConditionedBlock(128, 128),
            nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU()),
            MultimodalConditionedBlock(64, 64),
        ])
        self.dec_out = nn.Conv2d(64, 3, 3, 1, 1)
        self.diversity_loss = None
        
        # New: Text to Latent Projection
        self.text_to_latent = nn.Sequential(
            nn.Linear(config.TEXT_EMBEDDING_DIM, 512),
            nn.SiLU(),
            nn.Linear(512, config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W)
        )

    def encode(self, x, labels=None, text_emb=None):
        context = self.context_encoder(labels, text_emb)
        h = self.enc_in(x)
        for block in self.enc_blocks:
            h = block(h, context) if isinstance(block, MultimodalConditionedBlock) else block(h)
        return self.z_mean(h) * config.LATENT_SCALE, torch.clamp(self.z_logvar(h), -4, 4)

    def encode_text(self, text_emb):
        """Directly maps a text embedding into the latent space z."""
        h = self.text_to_latent(text_emb)
        z = h.view(-1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W)
        return z * config.LATENT_SCALE

    def decode(self, z, labels=None, text_emb=None):
        context = self.context_encoder(labels, text_emb)
        h = self.dec_in(z)
        for block in self.dec_blocks:
            h = block(h, context) if isinstance(block, MultimodalConditionedBlock) else block(h)
        return torch.tanh(self.dec_out(h))

    def forward(self, x, labels=None, text_emb=None):
        mu, logvar = self.encode(x, labels, text_emb)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) if self.training else mu
        return self.decode(z, labels, text_emb), mu, logvar

# ============================================================================
# FOURIER TIME EMBEDDING
# ============================================================================
class FourierTimeEmbed(nn.Module):
    def __init__(self, dim=128, max_freq=64):
        super().__init__()
        self.register_buffer('freqs', torch.linspace(1, max_freq, dim//2))
    def forward(self, t):
        t = t * 2 * math.pi
        return torch.cat([torch.sin(t * self.freqs), torch.cos(t * self.freqs)], dim=-1)

# ============================================================
# MULTIMODAL DRIFT
# ============================================================
class MultimodalDrift(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(FourierTimeEmbed(dim=128), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.context_encoder = ContextEncoder()
        self.cond_proj = nn.Sequential(nn.Linear(256 + config.TEXT_EMBEDDING_DIM, 256), nn.SiLU(), nn.Linear(256, 256))
        self.time_weight_net = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1), nn.Sigmoid())
        
        self.head = nn.utils.spectral_norm(nn.Conv2d(config.LATENT_CHANNELS, 64, 3, 1, 1))
        
        # U-Net structure with widespread Self-Attention
        self.down1 = MultimodalConditionedBlock(64, 128, context_dim=256, use_spectral_norm=True)
        self.down2_conv = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.down2_block = MultimodalConditionedBlock(256, 256, context_dim=256, use_spectral_norm=True)
        self.down2_attn = SelfAttention(256)
        
        self.mid1 = MultimodalConditionedBlock(256, 256, context_dim=256, use_spectral_norm=True)
        self.mid_attn = SelfAttention(256)
        self.mid2 = MultimodalConditionedBlock(256, 256, context_dim=256, use_spectral_norm=True)
        
        self.up2_conv = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.up2_block = MultimodalConditionedBlock(128, 128, context_dim=256, use_spectral_norm=True)
        self.up2_attn = SelfAttention(128)
        self.up1 = MultimodalConditionedBlock(128, 64, context_dim=256, use_spectral_norm=True)
        
        self.tail = nn.utils.spectral_norm(nn.Conv2d(64, config.LATENT_CHANNELS, 3, 1, 1))
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.time_scales = nn.Parameter(torch.ones(4) * 0.1)
        self.register_buffer('drift_mean', torch.zeros(1))
        self.register_buffer('drift_std', torch.ones(1))
        self.register_buffer('n_samples', torch.zeros(1))
        self.momentum = 0.99
        self._is_exporting = False

    def forward(self, z, t, labels=None, text_emb=None, cfg_scale=1.0):
        t_emb = self.time_mlp(t.unsqueeze(-1) if t.dim() == 1 else t)
        if cfg_scale != 1.0 and not self.training:
            cond_drift = self._forward_internal(z, t, labels, text_emb, t_emb)
            uncond_drift = self._forward_internal(z, t, None, None, t_emb)
            return uncond_drift + cfg_scale * (cond_drift - uncond_drift)
        return self._forward_internal(z, t, labels, text_emb, t_emb)

    def _forward_internal(self, z, t, labels, text_emb, t_emb):
        context = self.context_encoder(labels, text_emb)
        global_context = context.mean(dim=1) if (context is not None and context.dim() == 3) else (context if context is not None else torch.zeros(z.shape[0], config.TEXT_EMBEDDING_DIM, device=z.device))
        cond = self.cond_proj(torch.cat([t_emb, global_context], dim=-1))
        
        h = self.head(torch.clamp(z, -10, 10))
        
        d1 = self.down1(h, cond if context is None or context.dim() == 2 else context)
        d2 = self.down2_attn(self.down2_block(self.down2_conv(d1), cond if context is None or context.dim() == 2 else context))
        
        m = self.mid2(self.mid_attn(self.mid1(d2, cond if context is None or context.dim() == 2 else context)), cond if context is None or context.dim() == 2 else context)
        
        u2 = self.up2_attn(self.up2_block(self.up2_conv(m), cond if context is None or context.dim() == 2 else context))
        u1 = self.up1(u2 + d1, cond if context is None or context.dim() == 2 else context)
        
        out = self.tail(u1) * self.output_scale * (1.0 + self.time_weight_net(t).view(-1, 1, 1, 1))
        
        if self.training and not self._is_exporting:
            with torch.no_grad():
                self.drift_mean.mul_(self.momentum).add_(out.mean(), alpha=1 - self.momentum)
                self.drift_std.mul_(self.momentum).add_(out.std(), alpha=1 - self.momentum)
                self.n_samples.add_(out.shape[0])
        
        if not self.training or self._is_exporting:
            thresh = torch.abs(self.drift_mean) + 3.0 * self.drift_std
            out = torch.clamp(out, -thresh, thresh)
        return out

# ============================================================
# TEXT DRIFT (LLM COMPONENT)
# ============================================================
class TextDrift(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=512, nhead=8, num_layers=6):
        super().__init__()
        self.time_mlp = nn.Sequential(FourierTimeEmbed(dim=128), nn.Linear(128, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True), num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, z, t, context_emb=None):
        h = z + self.time_mlp(t).unsqueeze(1)
        if context_emb is not None:
            h = h + (context_emb.unsqueeze(1) if context_emb.dim() == 2 else context_emb)
        return self.out_proj(self.transformer(h))
