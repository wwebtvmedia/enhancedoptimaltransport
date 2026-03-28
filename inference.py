# ============================================================================
# LABEL-CONDITIONED INFERENCE FOR SCHRÖDINGER BRIDGE
# ============================================================================

import torch
from pathlib import Path
from typing import List, Optional
import training
import data_management as dm
import config

def run_inference(labels: Optional[List[int]] = None,
                  text_prompts: Optional[List[str]] = None,
                  samples_per_prompt: Optional[int] = None,
                  temperature: Optional[float] = None,
                  method: str = 'rk4',
                  langevin_steps: Optional[int] = None,
                  langevin_step_size: Optional[float] = None,
                  langevin_score_scale: Optional[float] = None,
                  cfg_scale: Optional[float] = None) -> None:
    """
    Run inference with multimodal conditioning.
    
    Args:
        labels: List of class labels.
        text_prompts: List of text prompts.
        samples_per_prompt: Number of samples per prompt/label.
        temperature: Sampling temperature.
        method: Integration method ('euler', 'heun', or 'rk4').
        langevin_steps: Number of Langevin refinement steps.
        langevin_step_size: Step size for Langevin dynamics.
        langevin_score_scale: Scaling factor for the approximate score.
        cfg_scale: Scale for classifier-free guidance.
    """
    checkpoint_path = config.DIRS["ckpt"] / "latest.pt"
    if not checkpoint_path.exists():
        config.logger.error("No trained model found! Train a model first.")
        return
    
    # Create dummy loader for trainer initialization
    from torch.utils.data import DataLoader, TensorDataset
    dummy_dataset = TensorDataset(torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE))
    dummy_loader = DataLoader(dummy_dataset, batch_size=1)
    trainer = training.EnhancedLabelTrainer(dummy_loader)
    
    if not trainer.load_for_inference():
        config.logger.error("Failed to load checkpoint")
        return
    
    config.logger.info("\n" + "="*50)
    config.logger.info("MULTIMODAL INFERENCE")
    config.logger.info("="*50)
    
    # Interactive selection of mode
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
                text_prompts = ["a small cat", "a fast airplane"]
        else:
            print("\nAvailable labels: 0-9 for STL-10 classes")
            label_input = input("\nEnter labels (comma-separated) [default: 0,1,2,3]: ").strip()
            if label_input:
                labels = [int(x.strip()) for x in label_input.split(',')]
            else:
                labels = [0, 1, 2, 3]
    
    if samples_per_prompt is None:
        samples_input = input(f"Samples per prompt/label [default: 2]: ").strip()
        samples_per_prompt = int(samples_input) if samples_input else 2
    
    if temperature is None:
        temperature = config.INFERENCE_TEMPERATURE

    # Build context
    all_labels = None
    text_emb = None
    
    if text_prompts:
        print(f"\nEncoding text prompts: {text_prompts}")
        # Placeholder for real text encoding (e.g. CLIP)
        # For now, we use zero embeddings as we haven't integrated a pre-trained encoder yet
        # in this turn. In a real setup, you'd call a CLIP model here.
        text_emb = torch.zeros(len(text_prompts) * samples_per_prompt, config.TEXT_EMBEDDING_DIM)
        print("⚠️ Using zero embeddings (Text Encoder not yet integrated).")
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
        cfg_scale=cfg_scale or 1.0
    )
    
    print(f"\n Generated samples saved to: {grid_path}")

if __name__ == "__main__":
    run_inference()