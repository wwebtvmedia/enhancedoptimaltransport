# Enhanced Label-Conditioned Schrödinger Bridge (LSB)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art generative model implementation based on the **Latent Schrödinger Bridge** problem, optimized for high-performance generative research on consumer hardware.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Advancements (v2.1)](#-key-advancements-v21)
- [Installation & Setup](#%EF%B8%8F-installation-and-setup)
- [Pretrained Models](#-pretrained-models)
- [Mathematical Foundations](#-mathematical-foundations)
- [Model Architecture](#%EF%B8%8F-model-architecture)
- [Training & Monitoring](#-training--monitoring)
- [Edge Deployment (ONNX)](#-edge-deployment-onnx)
- [References](#-references)

---

## Overview
This project enables high-quality, category-specific image generation by learning the optimal stochastic transport between a Gaussian prior and a learned latent data distribution. It leverages the **Ornstein-Uhlenbeck (OU) Reference Process** for superior stability and theoretical consistency.

## 🚀 Key Advancements in v2.2 (Ultra-Sharp Refinement)
- **Exponential Moving Average (EMA):** Smoothed weight tracking to eliminate digital grain and enhance photorealism.
- **Refinement Decay:** Automatic 50% learning rate reduction in late Phase 3 to "lock in" micro-textures.
- **Autonomous Strategy Engine:** KPI-driven phase transitions and stochastic "quality nudges" scaled by training temperature.
- **Quadrupled Structural Weights:** `DRIFT_WEIGHT` and `SSIM_WEIGHT` boosted to 4.0 for razor-sharp object boundaries.
- **Subpixel Convolution:** Replaced bilinear upsampling with PixelShuffle + ICNR initialization for artifact-free upscaling.

---

## 🛠️ Installation and Setup

### 1. Prerequisites
- Python 3.8+
- GPU acceleration (Recommended: NVIDIA CUDA, Apple Silicon MPS, or Intel XPU)

### 2. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install core dependencies
pip install torch torchvision numpy scipy tqdm onnx onnxruntime pillow streamlit pygal cairosvg huggingface_hub
```

---

## 🤗 Pretrained Models
The latest model weights are hosted on Hugging Face. You can download the `.pt` file and place it in `enhanced_label_sb/checkpoints/`.

| Version | Checkpoint | Description |
| :--- | :--- | :--- |
| **v2.1 (SharpFlow)** | [Download on HF](https://huggingface.co/models) | Latest Phase 3 joint fine-tuned weights. |

### Uploading to Hugging Face
If you want to host your own version, use the provided helper script:
```bash
python upload_to_hf.py
```

---

## 🧬 Mathematical Foundations

### 1. Schrödinger Bridge Problem
The model solves for the most probable evolution between a prior $P_0 = \mathcal{N}(0, I)$ and the data distribution $P_1$ in a latent space.
$$dz_t = f(z_t, t, y) dt + g(t) dw_t$$
where $f$ is the learned drift network and $y$ is the class label.

### 2. Ornstein-Uhlenbeck Reference (mvOU-SBP)
Provides a mean-reverting prior for improved sampling stability:
$$dx_t = -\theta x_t dt + \sigma dw_t$$
Training target is the **bridge velocity**: $v_t = \frac{d}{dt} \mathbb{E}[z_t | z_0, z_1]$.

### 3. Classifier-Free Guidance (CFG)
Improves label adherence via dropout (10%) during training and guided drift during inference:
$$f_{cfg} = f(z, t, \text{null}) + s \cdot (f(z, t, y) - f(z, t, \text{null}))$$

---

## 🏗️ Model Architecture

### Label-Conditioned VAE
- **Encoder:** Maps images to an 8-channel latent space ($8 \times 12 \times 12$).
- **Decoder:** PixelShuffle-based reconstruction with FiLM-style conditioning.
- **Regularization:** KL annealing + Diversity Loss to prevent channel collapse.

### U-Net Drift Network
- **Structure:** Time-aware U-Net with residual blocks and bottleneck self-attention.
- **Conditioning:** Combined Fourier time embeddings and learned label embeddings.
- **Stability:** Adaptive Clipping tracks drift statistics to prevent exploding gradients.

---

## 📈 Training & Monitoring

### Three-Phase Training Lifecycle
1.  **Phase 1 (VAE):** Optimizes the latent space using KL-annealed ELBO.
2.  **Phase 2 (Drift):** Freezes VAE and trains the U-Net to match bridge dynamics.
3.  **Phase 3 (Joint):** Fine-tunes both networks for maximum sharpness and alignment.

### Live Monitoring
- **Latent Monitor:** Real-time visualization of channel standard deviations.
- **Hot-Swap:** Change loss weights (KL, Diversity, etc.) on-the-fly.
- **LoRA:** Efficient adaptation for custom styles or datasets.

---

## 🌐 Edge Deployment (ONNX)
Models can be exported to ONNX for browser-based inference:
- **Generator:** `generator.onnx` (Latent $\to$ Image)
- **Drift:** `drift.onnx` (Trajectory Prediction)
- **Web UI:** `onnx_generate_image.html` (Local browser-based generation)

---

## 📚 References
- De Bortoli, V., et al. (2021). *Diffusion Schrödinger Bridge Matching*.
- Chen, T., et al. (2022). *Optimal Transport and Schrödinger Bridges*.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through SDEs*.

---
*Developed for high-performance generative research on consumer hardware.*
