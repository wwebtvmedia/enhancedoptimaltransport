
import torch
from pathlib import Path
import models
import config
import training
import data_management as dm
from torchvision.utils import save_image

def verify_with_data():
    print("🔍 Verifying Migrated Model with real data...")
    config.initialize_hardware()
    
    # Load real data
    loader = dm.load_data()
    batch = next(iter(loader))
    real_images = batch['image'][:8].to(config.DEVICE)
    labels = batch['label'][:8].to(config.DEVICE)
    
    # Load trainer and model
    trainer = training.EnhancedLabelTrainer(loader)
    if not trainer.load_for_inference():
        print("❌ Failed to load checkpoint.")
        return

    trainer.vae.eval()
    
    with torch.no_grad():
        # Encode and Decode
        mu, logvar = trainer.vae.encode(real_images, labels)
        recon = trainer.vae.decode(mu, labels)
        
        print(f"Latent mu - min: {mu.min():.3f}, max: {mu.max():.3f}, mean: {mu.mean():.3f}, std: {mu.std():.3f}")
        print(f"Reconstruction - min: {recon.min():.3f}, max: {recon.max():.3f}, mean: {recon.mean():.3f}, std: {recon.std():.3f}")
        
        # Prepare display
        real_display = (real_images + 1) / 2
        recon_display = (recon + 1) / 2
        
        # Combine real and recon for comparison
        comparison = torch.cat([real_display, recon_display], dim=0)
        
        output_path = Path("migration_comparison.png")
        save_image(comparison, output_path, nrow=8)
        print(f"✅ Comparison image saved to {output_path}")
        print("Top row: Real Images | Bottom row: Migrated Model Reconstruction")

if __name__ == "__main__":
    verify_with_data()
