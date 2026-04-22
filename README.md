# Enhanced Label-Conditioned Schrödinger Bridge (LSB)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art generative model implementation based on the **Latent Schrödinger Bridge** problem, optimized for high-performance generative research on consumer hardware.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Advancements (v2.3)](#-key-advancements-v23)
- [Installation & Setup](#%EF%B8%8F-installation-and-setup)
- [Mathematical Foundations](#-mathematical-foundations)
- [Model Architecture](#%EF%B8%8F-model-architecture)
- [Autonomous Strategy](#-autonomous-strategy)
- [Edge Deployment (ONNX)](#-edge-deployment-onnx)

---

## Overview
This project enables high-quality, category-specific image generation by learning the optimal stochastic transport between a Gaussian prior and a learned latent data distribution. It leverages the **Ornstein-Uhlenbeck (OU) Reference Process** for superior stability and theoretical consistency.

## 🚀 Key Advancements in v2.3 (Pro-Active Stability)

- **KPI-Driven Autonomous Control:** Real-time monitoring of SSIM, Drift, and Latent Variance (`mu_std`) to dynamically adjust learning intensity and guidance.
- **"Panic Button" Restoration:** Automatic detection of training collapse with immediate restoration of stable model baselines.
- **Contrast & Detail Recovery:** Bidirectional adjustment logic that fights "fade and blurry" issues by boosting `CFG_SCALE` and `RECON_WEIGHT` when low contrast is detected.
- **Memory Optimization:** Fixed a critical memory leak in the ODE integration loop; 100+ step sampling now runs with constant memory overhead via `torch.no_grad()` scoping.
- **Phase Jittering:** Stochastic switching between VAE focus, Drift focus, and Joint training to prevent network "forgetting" and ensure long-term manifold stability.

---

## 🤗 Hugging Face Model Hub
The pre-trained weights and ONNX models are hosted on the Hugging Face Model Hub:
**[webtvmedianet/enhanced-schrodinger-bridge](https://huggingface.co/webtvmedianet/enhanced-schrodinger-bridge)**

### 📥 Using Pre-trained Models
You can download the `latest.pt` checkpoint and place it in `enhanced_label_sb/checkpoints/` to resume training or perform inference immediately.

### 📤 Uploading Your Own
To upload your current checkpoint and ONNX exports to your own HF account:
```bash
export HF_TOKEN="your_write_token"
python upload_to_hf.py
```

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

## 🧬 Mathematical Foundations

### 1. Schrödinger Bridge Problem
The model solves for the most probable evolution between a prior $P_0 = \mathcal{N}(0, I)$ and the data distribution $P_1$ in a latent space.
$$dz_t = f(z_t, t, y) dt + g(t) dw_t$$
where $f$ is the learned drift network and $y$ is the class label.

### 2. Ornstein-Uhlenbeck Reference (mvOU-SBP)
Provides a mean-reverting prior for improved sampling stability. The training target is the **bridge velocity**: $v_t = \frac{d}{dt} \mathbb{E}[z_t | z_0, z_1]$.

### 3. Classifier-Free Guidance (CFG)
Improves label adherence via dropout (10%) during training and guided drift during inference:
$$f_{cfg} = f(z, t, \text{null}) + s \cdot (f(z, t, y) - f(z, t, \text{null}))$$

---

## 🏗️ Model Architecture

### Label-Conditioned VAE
- **Encoder:** Maps images to an 8-channel latent space ($8 \times 12 \times 12$).
- **Decoder:** PixelShuffle-based reconstruction (Subpixel Convolution) with FiLM-style conditioning and ICNR initialization to eliminate grid artifacts.
- **Regularization:** KL annealing + Diversity Loss to prevent channel collapse.

### U-Net Drift Network
- **Structure:** Time-aware U-Net with residual blocks and spatial self-attention.
- **Conditioning:** Combined Fourier time embeddings and learned label/text embeddings.
- **Stability:** Adaptive Clipping tracks drift statistics to prevent exploding gradients.

---

## 🤖 Autonomous Strategy (App Control)

The system features an autonomous supervisor (`app_processor.py`) that monitors Key Performance Indicators (KPIs):

| Detection | Action | Goal |
| :--- | :--- | :--- |
| **High SSIM Loss** | ↑ CFG, ↑ Recon Weight | Restore detail and contrast (Fix "Fade"). |
| **High Latent Chaos** | ↑ Langevin Steps, ↓ Scale | Smooth out noise and prevent artifacts. |
| **Low Diversity** | ↑ Diversity Weight | Prevent mode collapse/identical samples. |
| **KPI Collapse** | Baseline Restore | Return to known-good safe parameters. |

---

## 🌐 Edge Deployment (ONNX)
Models can be exported to ONNX for browser-based inference:
- **Generator:** `generator.onnx` (Latent $\to$ Image)
- **Drift:** `drift.onnx` (Trajectory Prediction)
- **Security:** `run_web_server.py` implements **Cross-Origin Isolation** (COOP/COEP) to enable high-performance multi-threaded WASM.

---
*Developed for high-performance generative research on consumer hardware.*
