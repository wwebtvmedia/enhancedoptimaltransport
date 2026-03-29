# ============================================================================
# LABEL-CONDITIONED INFERENCE FOR SCHRÖDINGER BRIDGE
# ============================================================================

import torch
import torchvision.utils as vutils
from pathlib import Path
from typing import List, Optional, Union
import training
import data_management as dm
import config
import models
from datetime import datetime
from PIL import Image
import torchvision.transforms as T

def image_to_text(image_path: Union[str, Path]) -> str:
    """
    Takes an image, encodes it, and decodes the latent vector back to text/labels.
    """
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        return "No model found."

    # Load model
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    trainer.load_for_inference()

    # Preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3)
    ])
    img_tensor = transform(img).unsqueeze(0).to(config.DEVICE)

    # Infer
    trainer.vae.eval()
    with torch.no_grad():
        mu, _ = trainer.vae.encode(img_tensor)
        _, logits = trainer.vae.context_decoder(mu)
        label_idx = torch.argmax(logits, dim=1).item()

    class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    return class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"

def image_to_image(image_path: Union[str, Path], 
                   target_label: Optional[int] = None,
                   target_text: Optional[str] = None,
                   strength: float = 0.5,
                   method: str = 'heun',
                   cfg_scale: float = 1.5) -> Optional[Path]:
    """
    Translates an input image to a target label/prompt.
    strength: 0.0 = original image, 1.0 = completely new image.
    """
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        return None
    
    # Load trainer/models
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    trainer.load_for_inference()
    
    # 1. Preprocess and Encode Source Image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3)
    ])
    img_tensor = transform(img).unsqueeze(0).to(config.DEVICE)
    
    trainer.vae.eval()
    trainer.drift.eval()
    
    with torch.no_grad():
        # Get source latent
        z_src, _ = trainer.vae.encode(img_tensor)
        
        # 2. Add Noise based on strength
        # z_init = (1-strength) * z_src + strength * noise
        noise = torch.randn_like(z_src)
        z = (1.0 - strength) * z_src + strength * noise
        
        # 3. Prepare target condition
        label_tensor = None
        text_tensor = None
        if target_label is not None:
            label_tensor = torch.tensor([target_label], dtype=torch.long, device=config.DEVICE)
        elif target_text:
            # Simple mapping for this prototype
            class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
            idx = 0
            for i, name in enumerate(class_names):
                if name.lower() in target_text.lower():
                    idx = i; break
            text_tensor = trainer.text_encoder(torch.tensor([idx], device=config.DEVICE))
            
        # 4. Run Bridge (Shorter path based on strength)
        steps = int(config.DEFAULT_STEPS * strength)
        steps = max(steps, 10)
        dt = 1.0 / steps
        
        # Start at t = (1 - strength)
        t_start = 1.0 - strength
        
        for i in range(steps):
            t_val = t_start + (i * dt * strength)
            t_cur = torch.full((1, 1), t_val, device=config.DEVICE)
            
            if method == 'euler':
                drift = trainer.drift(z, t_cur, label_tensor, text_tensor, cfg_scale=cfg_scale)
                z = z + drift * (dt * strength)
            elif method == 'heun':
                k1 = trainer.drift(z, t_cur, label_tensor, text_tensor, cfg_scale=cfg_scale)
                z_pred = z + (dt * strength) * k1
                t_next = torch.full((1, 1), min(1.0, t_val + dt * strength), device=config.DEVICE)
                k2 = trainer.drift(z_pred, t_next, label_tensor, text_tensor, cfg_scale=cfg_scale)
                z = z + ((dt * strength) / 2.0) * (k1 + k2)
        
        # 5. Decode
        res_img = trainer.vae.decode(z, label_tensor, text_tensor)
        
    # Save comparison grid
    comparison = torch.cat([img_tensor, res_img], dim=0)
    grid = vutils.make_grid((comparison + 1) / 2, nrow=2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.DIRS["samples"] / f"i2i_{timestamp}.png"
    vutils.save_image(grid, out_path)
    return out_path

def run_inference(labels: Optional[List[int]] = None,
                  text_prompts: Optional[List[str]] = None,
                  samples_per_prompt: int = 1,
                  temperature: float = config.INFERENCE_TEMPERATURE,
                  method: str = 'heun',
                  cfg_scale: float = 1.0) -> Optional[Path]:
    """
    Run inference with multimodal conditioning.
    """
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        config.logger.error("No trained model found! Train a model first.")
        return None
    
    # Create trainer
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    
    if not trainer.load_for_inference():
        config.logger.error("Failed to load checkpoint")
        return None
    
    config.logger.info("\n" + "="*50)
    config.logger.info("MULTIMODAL INFERENCE")
    config.logger.info("="*50)
    
    # Interactive selection if nothing provided
    if labels is None and text_prompts is None:
        print("\nChoose input mode:")
        print("  1. Discrete Labels (0-9)")
        print("  2. Text Prompts")
        mode = input("\nSelect mode [1]: ").strip()
        
        if mode == '2':
            prompt_input = input("\nEnter text prompts (comma-separated): ").strip()
            if prompt_input:
                text_prompts = [x.strip() for x in prompt_input.split(',')]
            else:
                text_prompts = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        else:
            print("\nAvailable labels: 0-9 for STL-10 classes")
            label_input = input("\nEnter labels (comma-separated) [default: 0,1,2,3]: ").strip()
            if label_input:
                labels = [int(x.strip()) for x in label_input.split(',')]
            else:
                labels = [0, 1, 2, 3]

    # Build context
    all_labels = None
    text_emb = None
    
    if text_prompts:
        config.logger.info(f"Encoding text prompts: {text_prompts}")
        # Use simple label-to-text mapping for now if real text encoder is not trained
        # or use the trained text_encoder
        trainer.text_encoder.eval()
        
        # We need to map text strings to something the encoder understands.
        # Since our TextEncoder uses an Embedding(1000, dim), we'll map class names to indices.
        class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        indices = []
        for prompt in text_prompts:
            found = False
            for i, name in enumerate(class_names):
                if name.lower() in prompt.lower():
                    indices.append(i)
                    found = True
                    break
            if not found:
                indices.append(0) # Default
        
        indices_expanded = []
        for idx in indices:
            indices_expanded.extend([idx] * samples_per_prompt)
            
        with torch.no_grad():
            # Build padded tokens [B, MAX_TEXT_LENGTH]
            batch_size = len(indices_expanded)
            tokens = torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long, device=config.DEVICE)
            for i, idx in enumerate(indices_expanded):
                tokens[i, 0] = min(idx + 1, config.CLIP_VOCAB_SIZE - 2)
                tokens[i, 1] = config.CLIP_VOCAB_SIZE - 1 # EOS
            
            text_emb = trainer.text_encoder(tokens)
    else:
        all_labels = []
        for label in labels:
            all_labels.extend([label] * samples_per_prompt)
    
    # Generate samples
    grid_path = trainer.generate_samples(
        labels=all_labels,
        text_emb=text_emb,
        temperature=temperature,
        method=method,
        cfg_scale=cfg_scale
    )
    
    config.logger.info(f"Generated samples saved to: {grid_path}")
    return grid_path

def text_to_text(source_text: str, 
                 target_text: str, 
                 steps: int = 50) -> str:
    """
    Translates one text concept to another via the latent space.
    """
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        return "No model found."

    # Load trainer/models
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    trainer.load_for_inference()

    class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

    def get_idx(text):
        for i, name in enumerate(class_names):
            if name.lower() in text.lower(): return i
        return 0

    src_idx = get_idx(source_text)
    tgt_idx = get_idx(target_text)

    trainer.vae.eval()
    trainer.drift.eval()
    trainer.text_encoder.eval()

    with torch.no_grad():
        # Start with a small noise vector
        z = torch.randn(1, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W, device=config.DEVICE) * 0.1

        # Target Context
        tgt_tokens = torch.tensor([tgt_idx], device=config.DEVICE)
        tgt_emb = trainer.text_encoder(tgt_tokens)

        # Run Bridge
        dt = 1.0 / steps
        for i in range(steps):
            t_cur = torch.full((1, 1), i * dt, device=config.DEVICE)
            drift = trainer.drift(z, t_cur, None, tgt_emb)
            z = z + drift * dt

        # Decode z back to Text
        _, logits = trainer.vae.context_decoder(z)
        pred_idx = torch.argmax(logits, dim=1).item()

    return class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"

if __name__ == "__main__":
    run_inference()