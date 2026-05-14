import onnxruntime as ort
import numpy as np
import torch
import data_management as dm
import config
from PIL import Image
import os
import time

def run_stm32_inference_test(label_idx=0, steps=20):
    print(f"🚀 Starting STM32-compatible ONNX Inference Test")
    print(f"Target Label: {dm.CLASS_DESCRIPTIONS[label_idx]} (Index {label_idx})")
    
    onnx_dir = "enhanced_label_sb/onnx"
    drift_path = os.path.join(onnx_dir, "drift_static_int8.onnx")
    gen_path = os.path.join(onnx_dir, "generator_static_int8.onnx")

    if not os.path.exists(drift_path) or not os.path.exists(gen_path):
        print("❌ Error: Quantized models not found. Run quantization first.")
        return

    # 1. Initialize Sessions
    print(f"📦 Loading ONNX sessions...")
    # Use CPU for testing to match MCU behavior (no GPU specific kernels)
    providers = ['CPUExecutionProvider']
    drift_sess = ort.InferenceSession(drift_path, providers=providers)
    gen_sess = ort.InferenceSession(gen_path, providers=providers)

    # 2. Prepare Inputs
    batch_size = 1
    # Start with Gaussian Noise as the Prior
    z = np.random.randn(batch_size, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W).astype(np.float32)
    z = z * config.CST_COEF_GAUSSIAN_PRIO
    
    # Prepare text embedding
    text_desc = dm.CLASS_DESCRIPTIONS[label_idx]
    text_bytes = np.array([dm.text_to_bytes(text_desc)], dtype=np.int64)
    labels = np.array([label_idx], dtype=np.int64)
    source_id = np.array([0], dtype=np.int64)
    cfg_scale = np.array([config.CFG_SCALE], dtype=np.float32)

    # 3. ODE Integration Loop (Simulating MCU Loop)
    print(f"🔄 Running {steps} Drift steps (ODE Integration)...")
    dt = 1.0 / steps
    start_time = time.time()
    
    for i in range(steps):
        t = np.array([[i / steps]], dtype=np.float32)
        
        # Run Drift Model
        drift_inputs = {
            'z': z,
            't': t,
            'text_bytes': text_bytes,
            'source_id': source_id,
            'cfg_scale': cfg_scale
        }
        
        drift_out = drift_sess.run(None, drift_inputs)[0]
        
        # Euler Step: z_{t+1} = z_t + drift * dt
        z = z + drift_out * dt
        
        if (i+1) % 5 == 0:
            print(f"  Step {i+1}/{steps} complete...")

    drift_time = time.time() - start_time
    print(f"⏱️ Drift loop took {drift_time:.2f}s ({drift_time/steps:.4f}s per step)")

    # 4. Final Decoding
    print(f"🎨 Decoding latent to image...")
    gen_inputs = {
        'z': z,
        'text_bytes': text_bytes,
        'source_id': source_id
    }
    
    decode_start = time.time()
    generated_img = gen_sess.run(None, gen_inputs)[0]
    decode_time = time.time() - decode_start
    print(f"⏱️ Decoding took {decode_time:.2f}s")

    # 5. Post-process and Save
    # Convert [-1, 1] to [0, 255]
    img_np = ((generated_img[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    output_path = "stm32_test_output.png"
    Image.fromarray(img_np).save(output_path)
    print(f"✅ Success! Test image saved to: {output_path}")
    print(f"📊 Total latency (simulated): {drift_time + decode_time:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=0, help="Label index to generate")
    parser.add_argument("--steps", type=int, default=20, help="Number of ODE steps")
    args = parser.parse_args()
    
    run_stm32_inference_test(label_idx=args.label, steps=args.steps)
