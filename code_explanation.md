# Code Architecture & Implementation Deep-Dive

This document provides a comprehensive technical walkthrough of the Enhanced Label-Conditioned Schrödinger Bridge implementation.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure Analysis](#file-structure-analysis)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Model Architecture Details](#model-architecture-details)
5. [Training Pipeline](#training-pipeline)
6. [Autonomous Strategy Engine](#autonomous-strategy-engine)
7. [Inference & Multimodal Generation](#inference--generation)
8. [GUI Interfaces (MCP Design)](#gui-interfaces)
9. [Technical Innovations](#key-technical-innovations)

---

## Project Overview

This project implements a generative model based on the **Latent Schrödinger Bridge (LSB)** problem, enabling category-specific image generation by learning optimal stochastic transport between a Gaussian prior and a learned latent data distribution.

### Key Features:
- **Autonomous Intelligence**: Self-managing training phases and dynamic parameter scaling.
- **Bidirectional Multimodal Engine**: Supports Text $\leftrightarrow$ Image and Text $\leftrightarrow$ Text translation.
- **OU Reference Process**: Theoretically correct bridge velocity training targets.
- **Unified Hardware Engine**: Support for CUDA, MPS (Apple), and XPU (Intel).

---

## File Structure Analysis

### Core Modules
- **`main.py`**: Entry point with CLI and mode selection.
- **`config.py`**: Centralized configuration, hyperparameters, and device detection.
- **`models.py`**: Neural network architectures (VAE, U-Net Drift, Tokenizers).
- **`training.py`**: Core training logic, loss functions, and bridge mathematics.
- **`inference.py`**: High-level generation and translation APIs.
- **`data_management.py`**: Dataset handling (STL10, CIFAR10) and robust checkpoint loading.

### Application Layer (MCP Architecture)
- **`app_context.py`**: Central state container for progress tracking.
- **`app_processor.py`**: The **Intelligence Layer**. Houses the Autonomous Strategy Engine.
- **`app_streamlit.py` / `appmain_tk.py`**: Web and Desktop interfaces.

---

## Mathematical Foundations

### 1. Schrödinger Bridge Problem
The model solves for the SDE that maps $P_0$ (prior) to $P_1$ (data):
$$dz_t = f(z_t, t, y) dt + g(t) dw_t$$

### 2. Ornstein-Uhlenbeck Reference
Unlike standard Wiener processes, OU provides mean-reverting dynamics:
$$dx_t = -\theta x_t dt + \sigma dw_t$$
The `OUReference` class implements exact transition kernels and bridge sampling.

### 3. Classifier-Free Guidance (CFG)
Interpolates between conditional and unconditional drift for better label adherence:
$$f_{cfg} = f_{uncond} + s \cdot (f_{cond} - f_{uncond})$$

---

## Model Architecture Details

### 🧠 CNN Neural Tokenizer (Mathematical Formulation)
The system maps raw text prompts $T$ to a semantic vector $E_{text} \in \mathbb{R}^{512}$ through a four-stage neural pipeline:

1.  **Byte-Level Projection**: Text is converted to UTF-8 bytes $B \in \{0 \dots 255\}^L$ and projected to $X_0 \in \mathbb{R}^{L \times D}$.
2.  **1D-CNN Feature Extraction**: Captures local n-gram structures using a SiLU-activated 1D convolution:
    $$X_{conv} = \text{SiLU}(\text{Conv1d}(X_0))$$
3.  **Global Self-Attention**: Resolves long-range dependencies across the prompt:
    $$X_{attn} = \text{MHSA}(\text{LayerNorm}(X_{conv}))$$
4.  **Contrastive Alignment**: During Phase 1, the text vector is aligned with image latents $z$ using the **InfoNCE** objective:
    $$\mathcal{L}_{C} = -\mathbb{E} \left[ \log \frac{e^{sim(z, E_{text})/\tau}}{\sum e^{sim(z, E_{text, j})/\tau}} \right]$$

### Multimodal VAE
- **Encoder**: Maps 96x96 RGB to 12x12x8 latent space.
- **Decoder**: PixelShuffle-based upsampling with FiLM conditioning.
- **Diversity Loss**: Penalizes latent channel collapse via variance monitoring.

### Multimodal Drift Network (U-Net)
- **Time/Label Embeddings**: Combined Fourier features and learned embeddings.
- **Bottleneck Attention**: Captures long-range dependencies in latent space.
- **Adaptive Scaling**: Uses `PercentileRescale` for internal activation stability.

---

## Training Pipeline

### Three-Phase Curriculum
The training progresses through three distinct states to ensure stability:
1. **Phase 1: VAE Training**: Focuses on latent space reconstruction and organization.
2. **Phase 2: Drift Matching**: Freezes VAE and trains the U-Net to match transport dynamics.
3. **Phase 3: Joint Fine-tuning**: Unfreezes all components for final perceptual alignment.

---

## Autonomous Strategy Engine

Located in `app_processor.py`, this layer acts as a "virtual researcher" to manage training:

### 1. KPI-Driven Transitions
Transitions between phases are handled automatically based on performance metrics:
- **Phase 1 $\to$ 2**: Triggered when SNR > 20dB and SSIM < 0.26.
- **Phase 2 $\to$ 3**: Triggered when Drift Loss < 1.25 and SSIM < 0.21.

### 2. Dynamic Parameter Controller (Bidirectional)
Monitors multiple health signals to balance reconstruction, contrast, and smoothing:
- **Fade & Blurry Detection**: If SSIM loss is high (>0.25), the system boosts `CFG_SCALE` (up to 14.0) and `RECON_WEIGHT` (up to 25.0) to restore lost contrast and detail.
- **Anti-Artifact Control**: If latent chaos (`mu_std` > 1.2) or Drift loss is high, the system increases Langevin steps and lowers `LANGEVIN_SCORE_SCALE` to smooth out noise.
- **Panic Button (Baseline Restoration)**: If the composite KPI score collapses below -100, the system automatically resets all weights to known-good safe defaults to prevent total training divergence.

### 3. Training Type Jitter
To prevent network "forgetting," the system stochastically switches between three training focuses:
- **VAE Focus (10%)**: Forces the decoder to re-align with the current latent distribution.
- **Drift Focus (10%)**: Prioritizes trajectory accuracy.
- **Joint Fine-tuning (80%)**: The standard mode for global optimization.

---

## Inference & Generation

### Generation Logic
1. Sample $z_0 \sim \mathcal{N}(0, I)$.
2. Integrate $dz_t$ using Euler-Maruyama, Heun, or RK4 methods from $t=0 \to 1$.
3. Decode $z_1 \to \text{Image}$.

### Memory-Optimized Sampling
The sampling process uses a `torch.no_grad()` scoped integration loop. This allows for very high-step counts (RK4 with 100+ steps) with **constant memory overhead**, preventing the "CUDA Out of Memory" errors typically associated with long-trajectory ODE solvers.

### Multimodal Modes
- **Text-to-Image**: Guided generation from text prompts.
- **Image-to-Image**: Style/label translation with strength control.
- **Image-to-Text**: Latent-based classification and captioning.

---

## Technical Innovations

1. **Theoretically Correct OU Bridge**: Exact derivatives for velocity targets.
2. **Adaptive Diversity Loss**: Dynamic penalties for latent collapse prevention.
3. **Spectral Norm Baking**: Removing SN wrappers for efficient ONNX export.
4. **Flexible Loading**: Robust state-dict mapping for architecture evolution.

---
*Last Updated: April 19, 2026*
