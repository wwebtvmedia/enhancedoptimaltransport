import onnxruntime as ort
import numpy as np
import os
import json
import torch
import config
import data_management as dm
from tqdm import tqdm

def generate_calibration_data(num_samples=100, steps=20):
    """
    Generate realistic calibration data by running the FP32 ONNX models.
    This captures the actual distribution of latents along the ODE trajectory,
    which is essential for accurate quantization of the Drift and Generator models.
    """
    print(f"Generating calibration data using FP32 models...")
    onnx_dir = "enhanced_label_sb/onnx"
    drift_path = os.path.join(onnx_dir, "drift.onnx")
    gen_path = os.path.join(onnx_dir, "generator.onnx")

    if not os.path.exists(drift_path) or not os.path.exists(gen_path):
        print("Error: FP32 models not found in enhanced_label_sb/onnx/")
        return

    # Use CPU for deterministic and compatible results
    providers = ['CPUExecutionProvider']
    try:
        drift_sess = ort.InferenceSession(drift_path, providers=providers)
        gen_sess = ort.InferenceSession(gen_path, providers=providers)
    except Exception as e:
        print(f"Error loading ONNX models: {e}")
        return

    # Load real embeddings to use real prompts
    label_embeddings = {}
    embed_path = os.path.join(onnx_dir, "label_embeddings.json")
    if os.path.exists(embed_path):
        with open(embed_path, 'r') as f:
            label_embeddings = json.load(f)

    drift_inputs = []
    gen_inputs = []

    # Target directory
    calib_dir = "calibration_data"
    os.makedirs(calib_dir, exist_ok=True)

    for i in tqdm(range(num_samples)):
        # Random label from available classes
        label_idx = np.random.randint(0, 10)
        l_str = str(label_idx)
        
        if l_str in label_embeddings:
            text_emb = np.array(label_embeddings[l_str], dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to normalized random
            temp = np.random.randn(1, config.TEXT_EMBEDDING_DIM).astype(np.float32)
            text_emb = temp / (np.linalg.norm(temp) + 1e-8)

        # Initial z from the Gaussian prior
        z = np.random.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W).astype(np.float32)
        z *= config.CST_COEF_GAUSSIAN_PRIO
        
        cfg_scale = np.array([config.CFG_SCALE], dtype=np.float32)
        dt = 1.0 / steps

        # Drift loop: integrate the ODE to find the path
        for s in range(steps):
            t = np.array([[s / steps]], dtype=np.float32)
            
            # Save drift input for calibration
            drift_inputs.append({
                'z': z.copy(),
                't': t.copy(),
                'text_embedding': text_emb.copy(),
                'cfg_scale': cfg_scale.copy()
            })
            
            # Take one ODE step
            drift_out = drift_sess.run(None, {
                'z': z, 't': t, 'text_embedding': text_emb, 'cfg_scale': cfg_scale
            })[0]
            z = z + drift_out * dt
            
        # Save generator input (the final latent z at t=1)
        gen_inputs.append({
            'z': z.copy(),
            'text_embedding': text_emb.copy()
        })

    # Save to disk as .npy files
    np.save(os.path.join(calib_dir, "drift_inputs.npy"), drift_inputs, allow_pickle=True)
    np.save(os.path.join(calib_dir, "gen_inputs.npy"), gen_inputs, allow_pickle=True)
    print(f"Success: Saved {len(drift_inputs)} drift and {len(gen_inputs)} generator samples to {calib_dir}/")

if __name__ == "__main__":
    generate_calibration_data(num_samples=100, steps=20)
