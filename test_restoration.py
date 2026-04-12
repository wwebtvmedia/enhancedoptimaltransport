
import training
import data_management as dm
import config
import torch

# Initialize hardware
config.initialize_hardware()

loader = dm.load_data()
trainer = training.EnhancedLabelTrainer(loader)

# Load the restored checkpoint
if trainer.load_for_inference():
    print(f"✅ Loaded restored model from epoch {trainer.epoch}")
    # Run generation
    trainer.generate_samples(num_samples=4)
else:
    print("❌ Failed to load restored checkpoint")
