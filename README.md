# Enhanced Label-Conditioned Schrödinger Bridge with Ornstein-Uhlenbeck Reference

This project implements a state-of-the-art generative model based on the **Latent Schrödinger Bridge (LSB)** problem. It enables high-quality, category-specific image generation by learning the optimal stochastic transport between a Gaussian prior and a learned latent data distribution.

## 🚀 Key Advancements in v2.0
- **Theoretically Correct OU Bridge:** Uses the exact time-derivative of the OU bridge mean as the training target, replacing linear approximations.
- **Classifier-Free Guidance (CFG):** Integrated label dropout and guidance scaling for superior prompt/label alignment.
- **Three-Phase Lifecycle:** Robust progression from VAE training → Drift Matching → Joint Fine-tuning.
- **Unified Hardware Engine:** Centralized initialization for **NVIDIA (CUDA)**, **Apple Silicon (MPS)**, **Intel Arc (XPU)**, and **AMD (DirectML)**.
- **MCP Architecture:** Model-Context-Protocol design separating the core `TrainingProcessor` from Desktop (Tkinter), Web (Streamlit), and CLI interfaces.

---

## 🛠️ Installation and Setup

### 1. Prerequisites
- Python 3.8+
- `pip` (Python package installer)
- (Optional) GPU acceleration for faster training.

### 2. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install core dependencies
pip install torch torchvision numpy scipy tqdm onnx onnxruntime pillow streamlit pygal cairosvg
```

### 3. Launching the Interfaces
The project supports three distinct modes via `main.py`:

- **🖥️ Desktop Mode (Full GUI):** `python main.py --gui`
- **📱 Web Dashboard (Streamlit):** `python main.py --streamlit`
- **🚀 Terminal Mode (Headless):** `python main.py`

---

## 🧬 Mathematical Foundations

### 1. Schrödinger Bridge Problem
The model solves for the most probable evolution between a prior $P_0 = \mathcal{N}(0, I)$ and the data distribution $P_1$ in a latent space. This results in a Stochastic Differential Equation (SDE):
$$dz_t = f(z_t, t, y) dt + g(t) dw_t$$
where $f$ is the learned drift network and $y$ is the class label.

### 2. Ornstein-Uhlenbeck Reference (mvOU-SBP)
Unlike standard Wiener processes, the OU reference provides a mean-reverting prior:
$$dx_t = -\theta x_t dt + \sigma dw_t$$
The training target for the drift network is the **bridge velocity**:
$$v_t = \frac{d}{dt} \mathbb{E}[z_t | z_0, z_1]$$
Our implementation uses a stable numerical derivative of the exact OU bridge mean formula to ensure mathematical consistency.

### 3. Classifier-Free Guidance (CFG)
To improve label adherence, the drift network is trained with **Label Dropout** (10%). During inference, the drift is calculated as:
$$f_{cfg} = f(z, t, \text{null}) + s \cdot (f(z, t, y) - f(z, t, \text{null}))$$
where $s$ is the `CFG_SCALE`.

---

## 🏗️ Model Architecture

### Label-Conditioned VAE
- **Encoder:** Maps images to a 4-channel latent space ($4 \times 12 \times 12$ for $96 \times 96$ inputs).
- **Decoder:** Reconstructs images from latents.
- **Conditioning:** Uses scale-shift modulation (FiLM) based on label embeddings.
- **Regularization:** KL annealing and a **Diversity Loss** to prevent latent channel collapse.

### U-Net Drift Network
- **Structure:** A time-aware U-Net with residual blocks and self-attention at the bottleneck.
- **Embeddings:** Combined Fourier time embeddings and learned label embeddings.
- **Adaptive Clipping:** Automatically tracks drift statistics to prevent "exploding gradients" during the integration of the bridge.

---

## 📈 Training & Monitoring

### Three-Phase Training
1.  **Phase 1 (VAE):** Optimizes the latent space using KL-annealed ELBO.
2.  **Phase 2 (Drift):** Freezes the VAE and trains the U-Net to match the bridge dynamics.
3.  **Phase 3 (Joint):** Fine-tunes both networks for maximum sharpness and alignment.

### Live Monitoring Tools
- **Latent Monitor:** Real-time visualization of channel standard deviations to detect collapse early.
- **Visual Gallery:** Automatic preview of generated samples directly within the GUI.
- **Hot-Swap:** Change loss weights (KL, Diversity, etc.) on-the-fly without restarting training.

---

## 🌐 Edge Deployment (ONNX)
Models can be exported to ONNX for browser-based inference.
- **Generator:** `generator.onnx` (Latent $\to$ Image)
- **Drift:** `drift.onnx` (Predicts bridge trajectory)
- **Web UI:** Open `onnx_generate_image.html` in any modern browser to generate images locally using ONNX Runtime Web.

---

## 📚 References
- De Bortoli, V., et al. (2021). *Diffusion Schrödinger Bridge Matching*.
- Chen, T., et al. (2022). *Optimal Transport and Schrödinger Bridges*.
- Song, Y., et al. (2021). *Score-Based Generative Modeling through SDEs*.

---
*Developed for high-performance generative research on consumer hardware.*
