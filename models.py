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
        groups1 = min(8, out_channels)
        self.bn1 = nn.GroupNorm(groups1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        groups2 = min(8, out_channels)
        self.bn2 = nn.GroupNorm(groups2, out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels)
            )
    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        shortcut = self.shortcut(x)
        if out.shape[2:] != shortcut.shape[2:]:
            shortcut = F.interpolate(shortcut, size=out.shape[2:], mode='nearest')
            
        out += shortcut
        return F.silu(out)

def _mha_sdpa(mha_module, x, num_heads):
    """Apply nn.MultiheadAttention weights via F.scaled_dot_product_attention.
    This is ONNX-exportable (avoids the non-exportable _native_multi_head_attention kernel)
    and numerically equivalent to mha_module(x, x, x)."""
    b, seq, c = x.shape
    head_dim = c // num_heads
    qkv = F.linear(x, mha_module.in_proj_weight, mha_module.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(b, seq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(b, seq, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(b, seq, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(b, seq, c)
    return F.linear(out, mha_module.out_proj.weight, mha_module.out_proj.bias)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.num_heads = 4
        self.mha = nn.MultiheadAttention(in_channels, num_heads=self.num_heads, batch_first=True)
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        res = x.reshape(b, c, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]
        attn_out = _mha_sdpa(self.mha, res, self.num_heads)
        out = self.ln(res + attn_out)
        return out.permute(0, 2, 1).contiguous().reshape(b, c, h, w)


class SpatialSplitAttention(nn.Module):
    """Axial attention: decomposes global attention into H and W passes."""
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.ln_h = nn.LayerNorm(in_channels)
        self.ln_w = nn.LayerNorm(in_channels)
        self.h_mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.w_mha = nn.MultiheadAttention(in_channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Vertical pass: attend along H for each W column
        v = x.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()
        v_attn = _mha_sdpa(self.h_mha, self.ln_h(v), self.num_heads)
        x = x + v_attn.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()

        # 2. Horizontal pass: attend along W for each H row
        h_in = x.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous()
        h_attn = _mha_sdpa(self.w_mha, self.ln_w(h_in), self.num_heads)
        x = x + h_attn.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        return x

# ============================================================================
# NEURAL TOKENIZER AND SHARED EMBEDDING (NEW)
# ============================================================================
class NeuralTokenizer(nn.Module):
    """
    Character/byte-level CNN encoder for raw text.
    Processes UTF-8 bytes directly into dense embeddings.
    """
    def __init__(self, max_length=config.MAX_TEXT_BYTES, 
                 embed_dim=256, 
                 hidden_dim=config.NEURAL_TOKEN_HIDDEN_DIM,
                 vocab_size=260): # 256 bytes + 4 special tokens
        super().__init__()
        self.max_length = max_length
        self.byte_embedding = nn.Embedding(vocab_size, embed_dim)

        # Lightweight CNN for text processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.projection = nn.Linear(hidden_dim, config.TEXT_EMBEDDING_DIM)
        self.norm = nn.LayerNorm(config.TEXT_EMBEDDING_DIM)

    def forward(self, text_bytes):
        # text_bytes: [B, max_length] with values in [0, 259]
        # 1. Embedding
        x = self.byte_embedding(text_bytes) # [B, L, D]
        # 2. Transpose for Conv1d: [B, D, L]
        x = x.transpose(1, 2).contiguous()
        # 3. Conv features + Max Pool: [B, hidden_dim, 1] -> [B, hidden_dim]
        features = self.conv_layers(x).squeeze(-1)
        # 4. Final Projection
        out = self.projection(features)
        return self.norm(out)

class SharedEmbeddingHead(nn.Module):
    """
    Projection head to map different modalities into a shared space.
    """
    def __init__(self, in_dim, out_dim=config.TEXT_EMBEDDING_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.proj(x))

# ============================================================
# PERCENTILE RESCALING
# ============================================================
class PercentileRescale(nn.Module):
    """Adaptive percentile-based rescaling to [-1, 1]."""
    
    def __init__(self, features, p_low=1.0, p_high=99.0, momentum=0.005, use_tanh=True):
        super().__init__()
        self.register_buffer('low', torch.zeros(features))
        self.register_buffer('high', torch.ones(features))
        self.p_low, self.p_high, self.m = p_low, p_high, momentum
        self.use_tanh = use_tanh
        self._is_exporting = False
        self.force_active = False

    def _set_export_mode(self, is_exporting=True):
        self._is_exporting = is_exporting

    def forward(self, x):
        # Calculate scale and shift to map [low, high] to [-1, 1]
        # range = high - low, mid = (high + low) / 2
        # out = (x - mid) / (range / 2)
        
        low = self.low.reshape(1, -1, 1, 1)
        high = self.high.reshape(1, -1, 1, 1)
        
        mid = (high + low) / 2
        half_range = (high - low).clamp(min=1e-6) / 2

        if self._is_exporting:
            out = (x - mid) / half_range
            return torch.tanh(out) if self.use_tanh else out
        
        if self.training:
            with torch.no_grad():
                # Added .contiguous() to handle MPS layout issues
                flat = x.float().transpose(0, 1).contiguous().reshape(x.shape[1], -1)
                if flat.numel() > 0:
                    l = torch.quantile(flat, self.p_low/100, dim=1)
                    h = torch.quantile(flat, self.p_high/100, dim=1)
                    self.low.mul_(1 - self.m).add_(l, alpha=self.m)
                    self.high.mul_(1 - self.m).add_(h, alpha=self.m)
        
        # Rescale to [-1, 1] range
        out = (x - mid) / half_range
        
        if self.use_tanh:
             return torch.tanh(out)
        else:
             return out

# ============================================================================
# UTILITIES FOR QUALITY (NEW)
# ============================================================================
def icnr_init(conv_weight, upscale_factor=2, init=nn.init.kaiming_normal_):
    """
    ICNR initialization for subpixel convolution to prevent checkerboard artifacts.
    Ensures that the subpixel filters are initialized to be as close to bilinear 
    upsampling as possible, preventing early-stage grid motifs.
    """
    out_channels, in_channels, h, w = conv_weight.shape
    new_shape = [out_channels // (upscale_factor**2), in_channels, h, w]
    subkernel = torch.zeros(new_shape)
    init(subkernel)
    subkernel = subkernel.repeat_interleave(upscale_factor**2, dim=0)
    return subkernel

class NoiseInjection(nn.Module):
    """
    Injects learnable per-channel noise to break up repetitive textures (motifs).
    Commonly used in StyleGAN to improve natural texture variation.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._is_exporting = False
        self.force_active = False

    def _set_export_mode(self, is_exporting=True):
        self._is_exporting = is_exporting

    def forward(self, x):
        if self._is_exporting:
            return x
        if not self.training and not self.force_active:
            return x
            
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise

# ============================================================
# LABEL-CONDITIONED BLOCK
# ============================================================
class LabelConditionedBlock(nn.Module):
    """Residual block with label conditioning and noise injection."""
    
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
        
        # Noise injection to break motifs
        self.noise1 = NoiseInjection(c_out)
        self.noise2 = NoiseInjection(c_out)
        
        groups_out = min(8, c_out)
        self.norm2 = nn.GroupNorm(groups_out, c_out)
        
        if use_spectral_norm:
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(c_out, c_out, 3, 1, 1))
        else:
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        
        self.rescale = PercentileRescale(c_out, use_tanh=False) if config.USE_PERCENTILE else nn.Identity()
        
        if use_spectral_norm and c_in != c_out:
            self.skip = nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, 1))
        elif c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, labels=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.noise1(h) # Inject noise after first convolution
        
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
        h = self.noise2(h) # Inject noise after second convolution
        
        skip = self.skip(x)
        if h.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=h.shape[2:], mode='nearest')
            
        return self.rescale(skip + h)

class SubpixelUpsample(nn.Module):
    """
    Subpixel Convolution (PixelShuffle) for sharper upsampling.
    Uses ICNR initialization to prevent checkerboard artifacts.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Initialize with ICNR
        self.conv.weight.data.copy_(icnr_init(self.conv.weight.data, upscale_factor))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.silu(self.norm(self.shuffle(self.conv(x))))

# ============================================================
# LABEL-CONDITIONED VAE
# ============================================================
class LabelConditionedVAE(nn.Module):
    """VAE with label conditioning and optimized spatial resolution (12x12)."""
    def __init__(self, free_bits=config.FREE_BITS):
        super().__init__()
        self.free_bits = free_bits
        self._is_exporting = False
        
        # Multimodal Text Encoder
        if config.USE_NEURAL_TOKENIZER:
            self.text_encoder = NeuralTokenizer()
        else:
            self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
            
        self.current_epoch = 0
        self.mu_noise_scale = config.MU_NOISE_SCALE
        
        # Shared projection for image-text alignment
        if config.USE_PROJECTION_HEADS:
            # Latent dim: C * H * W
            self.image_proj = SharedEmbeddingHead(config.LATENT_DIM)
        
        # Fourier feature channels
        if config.USE_FOURIER_FEATURES:
            self.fourier_channels = len(config.FOURIER_FREQS) * 4
        else:
            self.fourier_channels = 0
        
        # Encoder with 3 stages (96 -> 48 -> 24 -> 12)
        self.enc_in = nn.Conv2d(3 + self.fourier_channels, 64, 3, 1, 1)
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(64, 128, stride=2),           # 96 -> 48
            LabelConditionedBlock(128, 128, label_dim=config.TEXT_EMBEDDING_DIM if config.USE_NEURAL_TOKENIZER else config.LABEL_EMB_DIM),
            ResidualBlock(128, 256, stride=2),          # 48 -> 24
            SpatialSplitAttention(256),
            LabelConditionedBlock(256, 512, label_dim=config.TEXT_EMBEDDING_DIM if config.USE_NEURAL_TOKENIZER else config.LABEL_EMB_DIM),
            ResidualBlock(512, 512, stride=2),          # 24 -> 12
        ])
        self.z_mean = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)
        self.z_logvar = nn.Conv2d(512, config.LATENT_CHANNELS, 3, 1, 1)

        # Contextual embedding (Source Dataset ID)
        if config.USE_CONTEXT:
            self.source_emb = nn.Embedding(config.NUM_SOURCES, config.CONTEXT_DIM)
            cond_in_dim = (config.TEXT_EMBEDDING_DIM if config.USE_NEURAL_TOKENIZER else config.LABEL_EMB_DIM) + config.CONTEXT_DIM
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_in_dim, config.LABEL_EMB_DIM),
                nn.SiLU(),
                nn.Linear(config.LABEL_EMB_DIM, config.LABEL_EMB_DIM if not config.USE_NEURAL_TOKENIZER else config.TEXT_EMBEDDING_DIM)
            )

        # ========== ENHANCED DECODER WITH SUBPIXEL CONV (3 stages) ==========
        # NEW: Latent Normalization for stability
        self.latent_norm = nn.GroupNorm(min(8, config.LATENT_CHANNELS), config.LATENT_CHANNELS)
        self.dec_in = nn.Conv2d(config.LATENT_CHANNELS, 512, 3, 1, 1)
        l_dim = config.TEXT_EMBEDDING_DIM if config.USE_NEURAL_TOKENIZER else config.LABEL_EMB_DIM
        
        if config.USE_SUBPIXEL_CONV:
            self.dec_blocks = nn.ModuleList([
                # Stage 1: 12x12 -> 24x24
                SubpixelUpsample(512, 256),
                LabelConditionedBlock(256, 256, label_dim=l_dim),
                SpatialSplitAttention(256),

                # Stage 2: 24x24 -> 48x48
                SubpixelUpsample(256, 128),
                LabelConditionedBlock(128, 128, label_dim=l_dim),
                SpatialSplitAttention(128),

                # Stage 3: 48x48 -> 96x96
                SubpixelUpsample(128, 64),
                LabelConditionedBlock(64, 64, label_dim=l_dim),
            ])
        else:
            self.dec_blocks = nn.ModuleList([
                # Stage 1: 12x12 -> 24x24
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(512, 256, 3, 1, 1),
                    nn.SiLU()
                ),
                LabelConditionedBlock(256, 256, label_dim=l_dim),
                SpatialSplitAttention(256),

                # Stage 2: 24x24 -> 48x48
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(256, 128, 3, 1, 1),
                    nn.SiLU()
                ),
                LabelConditionedBlock(128, 128, label_dim=l_dim),
                SpatialSplitAttention(128),

                # Stage 3: 48x48 -> 96x96
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.SiLU()
                ),
                LabelConditionedBlock(64, 64, label_dim=l_dim),
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

    def _get_conditioning(self, labels, text_bytes=None, source_id=None):
        """Standardized conditioning extractor for both modalities."""
        if config.USE_NEURAL_TOKENIZER and text_bytes is not None:
            cond_emb = self.text_encoder(text_bytes)
        else:
            if hasattr(self, 'label_emb'):
                cond_emb = self.label_emb(labels)
            else:
                cond_emb = torch.zeros(labels.shape[0], config.TEXT_EMBEDDING_DIM, device=labels.device)

        if config.USE_CONTEXT:
            if source_id is None:
                # Use default source (0) if context enabled but missing
                source_id = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
            s_emb = self.source_emb(source_id)
            cond_emb = self.cond_proj(torch.cat([cond_emb, s_emb], dim=-1))
            
        return cond_emb

    def encode(self, x, labels, text_bytes=None, source_id=None):
        """Encode images to latent distribution parameters."""
        cond_emb = self._get_conditioning(labels, text_bytes, source_id)
            
        if config.USE_FOURIER_FEATURES:
            fourier = self._get_fourier_features(x)
            x = torch.cat([x, fourier], dim=1)
        h = self.enc_in(x)
        for block in self.enc_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, cond_emb)
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
    
    def decode(self, z, labels, text_bytes=None, source_id=None):
        """Decode latents to images with enhanced architecture."""
        cond_emb = self._get_conditioning(labels, text_bytes, source_id)

        # Apply latent normalization for stability
        h = self.latent_norm(z)
        h = self.dec_in(h)
        for block in self.dec_blocks:
            if isinstance(block, LabelConditionedBlock):
                h = block(h, cond_emb)
            else:
                h = block(h)
        out = self.dec_out(h)
        return torch.tanh(out)

    def set_force_active(self, active=True):
        """Toggle force_active on submodules for better inference quality."""
        for m in self.modules():
            if isinstance(m, (NoiseInjection, PercentileRescale)):
                m.force_active = active

    def _set_export_mode(self, is_exporting=True):
        """Enable export-optimized forward paths."""
        self._is_exporting = is_exporting

    def forward(self, x, labels, text_bytes=None, source_id=None):
        """Forward pass with reparameterization."""
        mu, logvar = self.encode(x, labels, text_bytes, source_id)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels, text_bytes, source_id), mu, logvar
    
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
        self._is_exporting = False
        # ========== ENHANCED TIME EMBEDDING WITH FOURIER FEATURES ==========
        self.time_mlp = nn.Sequential(
            FourierTimeEmbed(dim=128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # Multimodal Text Encoder (same as VAE)
        if config.USE_NEURAL_TOKENIZER:
            self.text_encoder = NeuralTokenizer()
        else:
            self.label_emb = nn.Embedding(config.NUM_CLASSES, config.LABEL_EMB_DIM)
        
        # Shared projection for image-text alignment
        if config.USE_PROJECTION_HEADS:
            self.image_proj = SharedEmbeddingHead(config.LATENT_DIM)
        
        # Contextual embedding (Source Dataset ID)
        l_dim = config.TEXT_EMBEDDING_DIM if config.USE_NEURAL_TOKENIZER else config.LABEL_EMB_DIM
        if config.USE_CONTEXT:
            self.source_emb = nn.Embedding(config.NUM_SOURCES, config.CONTEXT_DIM)
            # 256 (time) + 512 (text) + 64 (source) = 832
            self.cond_proj = nn.Sequential(
                nn.Linear(256 + l_dim + config.CONTEXT_DIM, 256),
                nn.SiLU(),
                nn.Linear(256, 256) 
            )
        else:
            # 256 (time) + 512 (text) = 768
            self.cond_proj = nn.Sequential(
                nn.Linear(256 + l_dim, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
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
        
        self.down1 = LabelConditionedBlock(64, 128, label_dim=256, use_spectral_norm=True)
        self.down2_conv = nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.down2_block = LabelConditionedBlock(256, 256, label_dim=256, use_spectral_norm=True)
        
        self.mid1 = LabelConditionedBlock(256, 256, label_dim=256, use_spectral_norm=True)
        self.mid_attn = SpatialSplitAttention(256)
        self.mid2 = LabelConditionedBlock(256, 256, label_dim=256, use_spectral_norm=True)
        
        self.up2_conv = SubpixelUpsample(256, 128)
        self.up2_block = LabelConditionedBlock(128, 128, label_dim=256, use_spectral_norm=True)
        self.up1 = LabelConditionedBlock(128, 64, label_dim=256, use_spectral_norm=True)
        
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

    def forward(self, z, t, labels, text_bytes=None, source_id=None, cfg_scale=1.0):
        """Forward pass - predict drift at time t with CFG and context support."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t)

        if cfg_scale != 1.0 and not self.training and not self._is_exporting:
            B = z.shape[0]
            dummy_labels = torch.zeros(B, dtype=torch.long, device=z.device)

            # Conditional pass
            if config.USE_NEURAL_TOKENIZER and text_bytes is not None:
                cond_text_emb = self.text_encoder(text_bytes)
            elif hasattr(self, 'label_emb'):
                cond_text_emb = self.label_emb(labels)
            else:
                cond_text_emb = torch.zeros(B, config.TEXT_EMBEDDING_DIM, device=z.device)
            cond_drift = self._forward_with_emb(z, t, dummy_labels, cond_text_emb, t_emb, source_id)

            # Unconditional pass: zero text embedding, matching ONNX DriftCFGWrapper
            uncond_text_emb = torch.zeros(B, config.TEXT_EMBEDDING_DIM, device=z.device)
            uncond_drift = self._forward_with_emb(z, t, dummy_labels, uncond_text_emb, t_emb, source_id)

            return uncond_drift + cfg_scale * (cond_drift - uncond_drift)

        return self._forward_internal(z, t, labels, text_bytes, t_emb, source_id)

    def _forward_internal(self, z, t, labels, text_bytes, t_emb, source_id=None):
        """Internal forward pass for a single label and context set."""
        # Text/Label embedding
        if config.USE_NEURAL_TOKENIZER and text_bytes is not None:
            text_emb = self.text_encoder(text_bytes)
        else:
            if hasattr(self, 'label_emb'):
                text_emb = self.label_emb(labels)
            else:
                text_emb = torch.zeros(labels.shape[0], config.TEXT_EMBEDDING_DIM, device=labels.device)

        return self._forward_with_emb(z, t, labels, text_emb, t_emb, source_id)

    def _forward_with_emb(self, z, t, labels, text_emb, t_emb, source_id=None):
        """Internal forward pass with pre-computed text embedding (used for ONNX CFG export)."""
        # Combine embeddings with optional context
        if config.USE_CONTEXT:
            if source_id is None:
                # Use default source (0) if context enabled but missing
                source_id = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
            s_emb = self.source_emb(source_id)
            cond = torch.cat([t_emb, text_emb, s_emb], dim=-1)
        else:
            cond = torch.cat([t_emb, text_emb], dim=-1)
            
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
        z = torch.clamp(z, -config.ODE_CLAMP_MAX, config.ODE_CLAMP_MAX)
        
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