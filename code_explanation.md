# Code Architecture & Implementation Deep-Dive

This document provides a comprehensive technical walkthrough of the Enhanced Label-Conditioned Schrödinger Bridge implementation.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure Analysis](#file-structure-analysis)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Model Architecture Details](#model-architecture-details)
5. [Training Pipeline](#training-pipeline)
6. [Inference & Multimodal Generation](#inference--generation)
7. [GUI Interfaces (MCP Design)](#gui-interfaces)
8. [Technical Innovations](#key-technical-innovations)

---

## Project Overview

This project implements a generative model based on the **Latent Schrödinger Bridge (LSB)** problem, enabling category-specific image generation by learning optimal stochastic transport between a Gaussian prior and a learned latent data distribution.

### Key Features:
- **Bidirectional Multimodal Engine**: Supports Text $\leftrightarrow$ Image and Text $\leftrightarrow$ Text translation.
- **OU Reference Process**: Theoretically correct bridge velocity training targets.
- **Three-Phase Curriculum**: Progresses from VAE training to Drift Matching and finally Joint Fine-tuning.
- **Unified Hardware Engine**: Support for CUDA, MPS (Apple), and XPU (Intel).

---

## File Structure Analysis

### Core Modules
- **`main.py`**: Entry point with CLI and mode selection.
- **`config.py`**: Centralized configuration, hyperparameters, and device detection.
- **`models.py`**: Neural network architectures (VAE, U-Net Drift, Tokenizers).
- **`training.py`**: Main training logic, loss functions, and phase management.
- **`inference.py`**: High-level generation and translation APIs.
- **`data_management.py`**: Dataset handling (STL10, CIFAR10) and robust checkpoint loading.

### Application Layer (MCP Architecture)
- **`app_context.py`**: Central state container for progress tracking.
- **`app_processor.py`**: Middleware bridging UI and training logic.
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
| Phase | Duration | Focus |
| :--- | :--- | :--- |
| **Phase 1: VAE** | 0-50 Epochs | Reconstruction quality and latent organization (KL/SSIM). |
| **Phase 2: Drift** | 50-100 Epochs | Learning transport dynamics between prior and data. |
| **Phase 3: Joint** | 100-200 Epochs | Global optimization and perceptual fine-tuning. |

---

## Inference & Generation

### Generation Logic
1. Sample $z_0 \sim \mathcal{N}(0, I)$.
2. Integrate $dz_t$ using Euler-Maruyama or Heun's method from $t=0 \to 1$.
3. Decode $z_1 \to \text{Image}$.

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
