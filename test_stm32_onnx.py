import onnxruntime as ort
import numpy as np
import data_management as dm
import config
from PIL import Image
import os
import time


def compute_text_embedding(label_idx):
    """Run NeuralTokenizer from checkpoint to get a real text embedding."""
    try:
        import torch
        import training
        from torch.utils.data import DataLoader, TensorDataset
        dummy = DataLoader(TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)), batch_size=1)
        trainer = training.EnhancedLabelTrainer(dummy)
        if trainer.load_for_inference():
            trainer.vae.eval()
            text_desc = dm.CLASS_DESCRIPTIONS[label_idx]
            text_bytes = torch.tensor([dm.text_to_bytes(text_desc)], dtype=torch.long)
            with torch.no_grad():
                emb = trainer.vae.text_encoder(text_bytes).cpu().numpy().astype(np.float32)
            print(f"  Text embedding computed from checkpoint (shape {emb.shape})")
            return emb
    except Exception as e:
        print(f"  Warning: checkpoint unavailable ({e}); using normalized random embedding")
    emb = np.random.randn(1, config.TEXT_EMBEDDING_DIM).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


def run_stm32_inference_test(label_idx=0, steps=20):
    print(f"Starting STM32-compatible ONNX Inference Test")
    print(f"Target Label: {dm.CLASS_DESCRIPTIONS[label_idx]} (Index {label_idx})")

    onnx_dir = "enhanced_label_sb/onnx"
    drift_path = os.path.join(onnx_dir, "drift_static_int8.onnx")
    gen_path = os.path.join(onnx_dir, "generator_static_int8.onnx")

    if not os.path.exists(drift_path) or not os.path.exists(gen_path):
        print("Error: Quantized models not found. Run quantization first.")
        return

    print("Loading ONNX sessions...")
    providers = ['CPUExecutionProvider']
    drift_sess = ort.InferenceSession(drift_path, providers=providers)
    gen_sess = ort.InferenceSession(gen_path, providers=providers)

    # Log actual model inputs so we can verify the headless signature
    print("Drift inputs:", [i.name for i in drift_sess.get_inputs()])
    print("Generator inputs:", [i.name for i in gen_sess.get_inputs()])

    z = np.random.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W).astype(np.float32)
    z *= config.CST_COEF_GAUSSIAN_PRIO

    text_emb = compute_text_embedding(label_idx)
    cfg_scale = np.array([config.CFG_SCALE], dtype=np.float32)

    print(f"Running {steps} Drift steps (ODE Integration)...")
    dt = 1.0 / steps
    start_time = time.time()

    for i in range(steps):
        t = np.array([[i / steps]], dtype=np.float32)
        drift_out = drift_sess.run(None, {
            'z': z, 't': t, 'text_embedding': text_emb, 'cfg_scale': cfg_scale
        })[0]
        z = z + drift_out * dt
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}/{steps} complete...")

    drift_time = time.time() - start_time
    print(f"Drift loop: {drift_time:.2f}s ({drift_time/steps:.4f}s/step)")

    print("Decoding latent to image...")
    decode_start = time.time()
    generated_img = gen_sess.run(None, {'z': z, 'text_embedding': text_emb})[0]
    decode_time = time.time() - decode_start
    print(f"Decoding: {decode_time:.2f}s")

    img_np = ((generated_img[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    output_path = "stm32_test_output.png"
    Image.fromarray(img_np).save(output_path)
    print(f"Success! Test image saved to: {output_path}")
    print(f"Total latency (simulated): {drift_time + decode_time:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=0, help="Label index to generate")
    parser.add_argument("--steps", type=int, default=20, help="Number of ODE steps")
    args = parser.parse_args()
    run_stm32_inference_test(label_idx=args.label, steps=args.steps)
