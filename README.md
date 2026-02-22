Enhanced Label-Conditioned Schrödinger Bridge with Ornstein-Uhlenbeck Reference
This project implements a generative model based on the Schrödinger Bridge (SB) problem, enabling high-quality image generation with explicit label conditioning. The core idea is to learn a stochastic process that transports a simple prior (standard Gaussian) to a complex data distribution over a fixed time interval. The model combines a conditional Variational Autoencoder (VAE) for latent representation with a drift network that learns the bridge dynamics. An optional Ornstein-Uhlenbeck (OU) reference process provides a theoretically grounded prior for the bridge (mvOU-SBP).

Table of Contents
Installation and Setup

Prerequisites

Creating a Virtual Environment

Installing Dependencies

Launching the Program

Mathematical Foundations

Key Features

Usage

Model Architecture

Results and Monitoring

References

Installation and Setup
Prerequisites
Python 3.8 or higher

pip (Python package installer)

(Optional) CUDA-capable GPU for faster training (NVIDIA), or Apple Silicon (MPS), or AMD with DirectML

Creating a Virtual Environment
It is strongly recommended to use a virtual environment to isolate dependencies.

On Windows (PowerShell or Command Prompt):

bash
python -m venv venv
venv\Scripts\activate
On macOS/Linux:

bash
python3 -m venv venv
source venv/bin/activate
After activation, your terminal prompt should show (venv).

Installing Dependencies
Install the required packages using pip:

bash
pip install torch torchvision numpy scipy tqdm onnx onnxruntime
For AMD GPU support on Windows (DirectML):
Replace the torch installation with:

bash
pip install torch-directml torchvision numpy scipy tqdm onnx onnxruntime
For Apple Silicon (MPS):
The standard PyTorch installation from pip includes MPS support. Ensure you have the latest version.

Verify Installation:
Run Python and check that PyTorch can see your device:

python
import torch
print(torch.__version__)
print(torch.cuda.is_available())      # For NVIDIA
print(torch.backends.mps.is_available())  # For Apple Silicon
Launching the Program
Once dependencies are installed and the virtual environment is active, simply run:

bash
python main.py
You will be presented with an interactive menu. Choose the desired option (e.g., 1 for training, 5 for inference). Follow the prompts to configure epochs, labels, etc.

Example: Quick training test

text
python main.py
> Enter choice (1-9): 2
This will run a 5-epoch test to verify everything works.

All output (checkpoints, logs, samples) will be saved in the enhanced_label_sb/ directory.

Mathematical Foundations
1. Schrödinger Bridge Problem
The Schrödinger Bridge problem seeks the most probable evolution between two probability distributions over a fixed time interval [0,1]. Given a reference process (e.g., Wiener or Ornstein-Uhlenbeck), we look for a process whose marginal distributions at t=0 and t=1 match given target distributions, minimizing the Kullback–Leibler divergence to the reference. In generative modeling, we set:

t=0: Prior distribution P0 = N(0, I) (in latent space)

t=1: Data distribution P1 = q_phi(z|x,y) (latent codes of real images)

The solution yields a stochastic differential equation (SDE) of the form:

dz_t = f(z_t, t) dt + g(t) dw_t

where f is the drift we aim to learn, and g is the diffusion coefficient (fixed).

2. Latent Space Representation
Images x in R^(3x64x64) are mapped to a latent space z in R^(4x8x8) via a conditional VAE:

Encoder q_phi(z|x,y) outputs mean mu_phi(x,y) and log-variance logvar_phi(x,y).

Decoder p_theta(x|z,y) reconstructs the image.

The VAE is trained with the Evidence Lower Bound (ELBO):

log p(x|y) >= E_{q_phi}[log p_theta(x|z,y)] - beta * KL(q_phi(z|x,y) || p(z))

where p(z) = N(0, I) is the prior. The KL term for spatial latents (CxHxW) is:

KL = -0.5 * sum_{c,h,w} (1 + log(sigma_{c,h,w}^2) - mu_{c,h,w}^2 - sigma_{c,h,w}^2)

3. Reference Process: Ornstein-Uhlenbeck
The Ornstein-Uhlenbeck process is a natural reference because it is the unique stationary Gauss–Markov process. Its SDE is:

dx_t = -theta x_t dt + sigma dw_t

with stationary distribution N(0, sigma^2/(2 theta)). In our implementation, we can use either:

Linear interpolation (simple, faster): z_t = (1-t) z_0 + t z_1

Exact OU bridge (mvOU-SBP): samples from the true bridge between z_0 and z_1:

z_t | z_0, z_1 ~ N( mu(t), sigma^2(t) )

where
mu(t) = (sinh(theta(1-t))/sinh(theta)) z_0 + (sinh(theta t)/sinh(theta)) z_1
sigma^2(t) = (sigma^2/(2 theta)) * (sinh(theta t) sinh(theta(1-t)) / sinh(theta))

(using exponential forms for numerical stability).

4. Bridge Dynamics and Drift Network
The drift network f_psi(z_t, t, y) learns the correction to the reference drift needed to match the data distribution. The target is the velocity:

v = (z_1 - z_0) (for linear interpolation) or the residual from the OU mean.

The network is conditioned on time t (via an MLP) and label y (via an embedding). It outputs the learned drift.

5. Loss Functions
5.1 VAE Loss (Phase 1)
L_vae = L_recon + beta * L_kl + gamma * L_diversity

L_recon = MSE between input and reconstruction

L_kl = KL divergence with dynamic weighting based on latent statistics (to prevent collapse)

L_diversity = channel diversity loss encouraging all latent channels to be used:

L_diversity = mean(ReLU(target_min_std - channel_std)) + 0.1 * std(channel_std)

5.2 Drift Loss (Phase 2)
L_drift = E_{t, z_0, z_1} [ || w(t) * (z_1 - z_0 - f_psi(z_t, t, y)) ||_Huber ]

where w(t) = 1 + 2t (higher weight on later timesteps). If using the OU reference, the target becomes (z_1 - z_0) - mu_OU, where mu_OU is the OU bridge mean.

Additionally, a consistency loss keeps the current encoder close to a frozen reference encoder (from Phase 1) to stabilize training:

L_consistency = MSE(mu, mu_ref)

The total loss in Phase 2 is L_total = L_drift + lambda * L_consistency.

6. Training Schedule
Training is split into two phases:

Phase 1 (epochs 0-49): Train VAE (encoder and decoder) with KL annealing.

Phase 2 (epochs 50-199): Train drift network, fine-tune encoder (decoder frozen). The drift loss uses time-weighted Huber loss.

The schedule can be customized via the configuration menu (auto, manual, custom, alternate).

Key Features
Label Conditioning: Full support for class labels (up to 6000 classes) via embedding and scale-shift modulation in every block.

Percentile Rescaling: Adaptive normalization that rescales activations based on running percentiles, improving stability.

KPI Tracking: Real-time monitoring of key metrics (loss, SNR, latent statistics) with convergence detection and early stopping.

Snapshot Ensemble: Automatic saving of model snapshots every N epochs; can revert to last good snapshot on NaN/inf.

Composite Score: Multi-metric score for model selection (SNR, KL, diversity, drift error) – best overall model saved separately.

Ornstein-Uhlenbeck Bridge: Optional exact OU bridge sampling for theoretically grounded training (mvOU-SBP).

Multi-device Support: Runs on CPU, CUDA (NVIDIA), MPS (Apple Silicon), DirectML (AMD).

Usage
After launching main.py, you will see an interactive menu. Options include:

Enhanced training (fresh start): Train the model from scratch with full epochs (default 200). You can specify the number of epochs.

Quick test (5 epochs): Run a short training loop to verify setup.

Export models to ONNX: Convert trained models to ONNX format for deployment.

Generate samples from checkpoint: Load the latest checkpoint and generate sample images.

Label-conditioned inference: Interactive inference with user-provided labels and temperature.

Snapshot management & recovery: List, inspect, compare, and load snapshots for continued training or restart.

Resume from latest checkpoint: Continue training from the most recent checkpoint.

Configure training schedule: Set phase switching mode (auto, manual, custom, alternate, vae_only, drift_only).

Toggle OU bridge: Enable/disable the exact OU bridge sampling (mvOU-SBP).

During training, checkpoints are saved in enhanced_label_sb/checkpoints/:

latest.pt: latest model (overwritten each epoch)

best.pt: model with lowest loss so far

best_overall_epoch_XXXX.pt: model with highest composite score

Snapshots are saved in enhanced_label_sb/snapshots/ every SNAPSHOT_INTERVAL (default 20) and kept up to SNAPSHOT_KEEP (default 5).

Logs are written to enhanced_label_sb/logs/ and also printed to console.

Model Architecture
LabelConditionedVAE:

Encoder: 3x64x64 -> 32x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8 -> mu/logvar (4x8x8)

Decoder: 4x8x8 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 32x64x64 -> 3x64x64

Label conditioning via embedding + scale-shift in each block.

LabelConditionedDrift:

Takes latent z (4x8x8), time t, label y.

Time and label embeddings combined, processed through U-Net-like architecture with spectral normalization.

Outputs drift of same shape as z.

Time-aware weighting and learned output scaling.

PercentileRescale: Adaptive normalization layer that tracks 1st and 99th percentiles and rescales to [-1,1] via tanh.

Results and Monitoring
During training, the console displays:

Current epoch, phase, loss, SNR, latent statistics.

KPI tracker monitors trends and can trigger early stopping if loss increases for EARLY_STOP_PATIENCE epochs (Phase 2 only).

Generated samples are saved every 10 epochs in enhanced_label_sb/samples/. Each generation includes:

A grid image

Individual images per sample

Raw latent tensors for debugging

The composite score combines multiple metrics (SNR, KL, diversity, drift error) to select the overall best model, saved separately.

References
Léonard, C. (2014). A survey of the Schrödinger problem and some of its connections with optimal transport. Discrete and Continuous Dynamical Systems - Series A, 34(4), 1533-1574.

De Bortoli, V., et al. (2021). Diffusion Schrödinger Bridge Matching. Advances in Neural Information Processing Systems.

Song, Y., et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. International Conference on Learning Representations.

Chen, T., et al. (2022). Optimal Transport and Schrödinger Bridges. arXiv preprint.

For any questions or issues, please open an issue on the repository.