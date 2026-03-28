# Remote Training Guide: Enhanced Schrödinger Bridge

This guide explains how to train the multimodal Schrödinger Bridge model on a remote GPU server (AWS, RunPod, Lambda Labs, etc.) while monitoring progress in real-time.

---

## 1. Prerequisites & Setup

### Weights & Biases (Mandatory for Remote Monitoring)
1. Create a free account at [wandb.ai](https://wandb.ai).
2. Get your API Key from: `https://wandb.ai/authorize`

### Server Setup
On your remote server, run:
```bash
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

## 2. Running the Training (Persistent Session)

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

## 4. Transferring Results

After training is complete (or you have a "best" model), download the weights back to your local machine for GUI inference:

```bash
# On your LOCAL machine:
scp user@remote-ip:~/enhancedoptimaltransport/enhanced_label_sb/checkpoints/best.pt ./local_path/
```

---

## 5. Troubleshooting Remote Training
- **OOM (Out of Memory):** If the GPU runs out of memory, reduce `BATCH_SIZE` in `config.py`.
- **No GPU found:** Ensure `nvidia-smi` works on the remote server and that PyTorch is installed with CUDA support.
- **Dataset missing:** If training doesn't start, ensure the `data/` directory is present or set `download=True` in `data_management.py`.
