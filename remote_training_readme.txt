# Remote Training Guide: Enhanced Schrödinger Bridge

This guide explains how to train the multimodal Schrödinger Bridge model on a remote GPU server (AWS, RunPod, Lambda Labs, etc.) while monitoring progress in real-time.

---

## 0. Choosing a GPU Provider

For Schrödinger Bridge training, you need at least **16GB VRAM** (RTX 3090/4090 or A10G/A100).

| Provider | Best For | Estimated Cost | Setup Complexity |
| :--- | :--- | :--- | :--- |
| **[RunPod](https://runpod.io)** | **Recommended** (Easy & Fast) | $0.40 - $0.80/hr | Very Low (1-Click PyTorch) |
| **[Lambda Labs](https://lambdalabs.com)** | High-Performance (A100/H100) | $1.10 - $2.50/hr | Low (Standard SSH) |
| **[Vast.ai](https://vast.ai)** | Lowest Cost (Marketplace) | $0.20 - $0.50/hr | Medium (Docker-based) |
| **[AWS / GCP](https://aws.amazon.com)** | Enterprise / Scalability | $1.00 - $12.00/hr | High (Quota & IAM setup) |

### Recommended Hardware for this Project:
- **Budget:** NVIDIA RTX 3090 / 4090 (24GB VRAM). Great for `BATCH_SIZE=4`.
- **Performance:** NVIDIA A100 (40GB/80GB). Allows `BATCH_SIZE=8+` and faster convergence.
- **Inference Only:** NVIDIA T4 or L4 (Cheaper, but slow for training).

---

## 1. Prerequisites & Setup

### Provider-Specific Tips
- **RunPod:** Use the "PyTorch" template. Use `/workspace` as your working directory for persistent storage.
- **Vast.ai:** Use the `pytorch/pytorch` image. Remember to install system dependencies (`apt-get install git tmux libgl1-mesa-glx`) inside the container.
- **AWS/GCP:** Use a "Deep Learning AMI" (Ubuntu 22.04) to have CUDA drivers pre-installed.

### Weights & Biases (Mandatory for Remote Monitoring)
1. Create a free account at [wandb.ai](https://wandb.ai).
2. Get your API Key from: `https://wandb.ai/authorize`

### Server Setup
On your remote server, run:
```bash
# Update and install system dependencies
sudo apt-get update && sudo apt-get install -y tmux

# Clone the repository
git clone -b multimodal https://github.com/wwebtvmedia/enhancedoptimaltransport.git
cd enhancedoptimaltransport

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Login to W&B
wandb login  # Paste your API key when prompted
```

---

## 2. GPU & WandB Configuration

To ensure WandB correctly monitors your GPU:

1. **Verify GPU Availability:**
   Run this command in your environment to ensure PyTorch sees the GPU:
   ```bash
   python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

2. **WandB System Metrics:**
   WandB automatically collects GPU usage, memory, and temperature if `nvidia-smi` is installed and the GPU is detected by PyTorch. You can view these under the **"System"** tab in your WandB run dashboard.

3. **Troubleshooting "No GPU" in WandB:**
   If WandB shows "CPU" but you have a GPU:
   - Ensure `nvidia-smi` is accessible in the terminal.
   - Re-install the correct version of PyTorch for your CUDA version:
     `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (adjust `cu121` to your CUDA version).

---

## 3. Running the Training (Persistent Session)

Use `tmux` to ensure your training continues even if your SSH connection drops.

1. **Start a tmux session:**
   ```bash
   tmux new -s sb_train
   ```

2. **Launch the training:**
   ```bash
   python main.py  # Automatically runs in Headless mode
   ```

3. **Detach from the session:**
   Press `Ctrl + B`, then press `D`. You can now safely close your terminal.

4. **Reattach later to check logs:**
   ```bash
   tmux attach -t sb_train
   ```

---

## 3. Real-Time Monitoring

Once training starts, you don't need the terminal to see progress.

1. Open [wandb.ai](https://wandb.ai) in your browser.
2. Select the project: `enhanced-schrodinger-bridge`.
3. **Live Curves:** View Recon Loss, KL, Alignment Loss, and SNR in real-time.
4. **Visual Gallery:** Every 10 epochs, a grid of generated samples is uploaded to the "Media" section of your W&B dashboard.
5. **System Health:** Monitor remote GPU temperature, memory usage, and CPU load.

---

## 4. Vast.ai: Advanced Persistence & Setup

Vast.ai runs inside Docker. If your instance is interrupted or destroyed, you might lose your progress unless you use `/workspace` or a backup mechanism.

### Quick Start on Vast.ai:
1.  **Rent Instance:** Choose a PyTorch image (e.g., `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`).
2.  **Clone & Initial Setup:**
    ```bash
    git clone -b multimodal https://github.com/wwebtvmedia/enhancedoptimaltransport.git
    cd enhancedoptimaltransport
    chmod +x *.sh
    ./vast_setup.sh  # Installs system dependencies and requirements.txt
    ```

### Automated Checkpoint Backup:
I have included a mechanism to automatically mirror your latest checkpoint to `/workspace/backup_checkpoints/` every 5 minutes. This ensures that even if you delete the code folder, the `latest_backup.pt` is safe.

1.  **Start Everything:**
    ```bash
    ./vast_start.sh
    ```
    *This starts the backup watcher in the background AND launches training in a `tmux` session.*

2.  **Monitoring Progress:**
    - **Training Logs:** `tmux attach -t sb_train`
    - **Backup Status:** `tail -f backup.log`
    - **WandB:** Check your dashboard as usual.

3.  **Restoring from Backup:**
    If your main training folder is corrupted or deleted:
    ```bash
    cp /workspace/backup_checkpoints/latest_backup.pt enhanced_label_sb/checkpoints/latest.pt
    ```

---

## 5. Transferring Results

After training is complete (or you have a "best" model), download the weights back to your local machine for GUI inference:

```bash
# On your LOCAL machine:
scp user@remote-ip:~/enhancedoptimaltransport/enhanced_label_sb/checkpoints/best.pt ./local_path/
```

---

## 5. Troubleshooting Remote Training
- **OOM (Out of Memory):** If the GPU runs out of memory, reduce `BATCH_SIZE` in `config.py`.
- **No GPU found:** Ensure `nvidia-smi` works on the remote server and that PyTorch is installed with CUDA support.
- **WandB Logs "CPU":** See Section 2 above to verify PyTorch and CUDA versions.
- **Multiple GPUs:** To use a specific GPU, prepend your command with:
  `CUDA_VISIBLE_DEVICES=0 python main.py`
- **Dataset missing:** If training doesn't start, ensure the `data/` directory is present or set `download=True` in `data_management.py`.
