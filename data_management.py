# ============================================================================
# DATA MANAGEMENT AND SNAPSHOT HANDLING FOR SCHRÖDINGER BRIDGE
# ============================================================================

import os
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.utils as vutils
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, List, Dict, Union, Tuple

import config

# ============================================================
# CREATE ALL NECESSARY DIRECTORIES
# ============================================================
for d in config.DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# SNAPSHOT MANAGER
# ============================================================
class SnapshotManager:
    """Manage model snapshots for snapshot ensemble learning."""
    
    def __init__(self, model, optimizer, name="model", interval=config.SNAPSHOT_INTERVAL, keep=config.SNAPSHOT_KEEP):
        self.model = model
        self.optimizer = optimizer
        self.name = name
        self.interval = interval
        self.keep = keep
        self.snapshots = []
        self.snapshot_dir = config.DIRS["snaps"]

    
    def step(self) -> None:
        """Update snapshot scheduler."""
        return
    
    def should_save(self, epoch: int) -> bool:
        """Check if we should save a snapshot at this epoch."""
        return (epoch + 1) % self.interval == 0

    def save_snapshot(self, epoch: int, loss: float) -> Path:
        """Save model snapshot."""
        snapshot_path = self.snapshot_dir / f"{self.name}_snapshot_epoch_{epoch:04d}.pt"
        
        snapshot = {
            'epoch': epoch,
            'loss': loss,
            'model_type': self.name,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.name == "drift":
            snapshot['drift_state'] = self.model.state_dict()
            snapshot['opt_drift_state'] = self.optimizer.state_dict()
        else:
            snapshot['model_state'] = self.model.state_dict()
            snapshot['optimizer_state'] = self.optimizer.state_dict()
        
        torch.save(snapshot, snapshot_path)
        self.snapshots.append(snapshot_path)
        
        # Keep only the most recent 'keep' snapshots
        if len(self.snapshots) > self.keep:
            old_snapshot = self.snapshots.pop(0)
            if os.path.exists(old_snapshot):
                os.remove(old_snapshot)
        
        config.logger.info(f"Saved {self.name} snapshot at epoch {epoch}")
        return snapshot_path

    def revert(self) -> bool:
        """Emergency revert to last good snapshot."""
        if not self.snapshots:
            config.logger.error(f"❌ No snapshots available for {self.name} to revert to!")
            return False
        
        latest_snapshot = self.snapshots[-1]
        try:
            snapshot = torch.load(latest_snapshot, map_location=config.DEVICE, weights_only=False)
            
            if self.name == "drift" and 'drift_state' in snapshot:
                self.model.load_state_dict(snapshot['drift_state'])
                if 'opt_drift_state' in snapshot:
                    self.optimizer.load_state_dict(snapshot['opt_drift_state'])
            elif 'model_state' in snapshot:
                self.model.load_state_dict(snapshot['model_state'])
                if 'optimizer_state' in snapshot:
                    self.optimizer.load_state_dict(snapshot['optimizer_state'])
            else:
                config.logger.error(f" Invalid snapshot format for {self.name}")
                return False
            
            config.logger.info(f" Emergency revert successful for {self.name} to epoch {snapshot.get('epoch', 'unknown')}")
            return True
            
        except Exception as e:
            config.logger.error(f" Emergency revert failed: {e}")
            return False

# ============================================================
# CHECKPOINT SAVE/LOAD FUNCTIONS
# ============================================================
def save_checkpoint(trainer, is_best: bool = False, is_best_overall: bool = False) -> Path:
    """Save training checkpoint."""
    config.DIRS["ckpt"].mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': trainer.epoch,
        'step': trainer.step,
        'phase': trainer.phase,
        'phase2_start_epoch': getattr(trainer, 'phase2_start_epoch', None),
        'vae_state': trainer.vae.state_dict(),
        'drift_state': trainer.drift.state_dict(),
        'text_encoder_state': trainer.text_encoder.state_dict() if hasattr(trainer, 'text_encoder') else None,
        'opt_vae_state': trainer.opt_vae.state_dict(),
        'opt_drift_state': trainer.opt_drift.state_dict(),
        'opt_text_state': trainer.opt_text.state_dict() if hasattr(trainer, 'opt_text') else None,
        'scheduler_vae_state': trainer.scheduler_vae.state_dict(),
        'scheduler_drift_state': trainer.scheduler_drift.state_dict(),
        'best_loss': trainer.best_loss,
        'best_composite_score': trainer.best_composite_score,
        'training_schedule': config.TRAINING_SCHEDULE,
        'config': {
            'DATASET_NAME': config.DATASET_NAME,
            'IMG_SIZE': config.IMG_SIZE,
            'GEN_SIZE': config.GEN_SIZE
        }
    }
    
    if hasattr(trainer, 'kpi_tracker') and hasattr(trainer.kpi_tracker, 'metrics'):
        checkpoint['kpi_metrics'] = trainer.kpi_tracker.metrics
    
    # Save reference VAE if it exists
    if hasattr(trainer, 'vae_ref') and trainer.vae_ref is not None:
        checkpoint['vae_ref_state'] = trainer.vae_ref.state_dict()
    
    latest_path = config.DIRS["ckpt"] / "latest.pt"
    torch.save(checkpoint, latest_path)
    config.logger.info(f"Checkpoint saved to {latest_path}")
    
    if is_best:
        best_path = config.DIRS["ckpt"] / "best.pt"
        torch.save(checkpoint, best_path)
        config.logger.info(f"New best model saved (loss: {trainer.best_loss:.4f})")
    
    if is_best_overall:
        overall_best_path = config.DIRS["best"] / f"best_overall_epoch_{trainer.epoch+1:04d}.pt"
        overall_best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, overall_best_path)
        config.logger.info(f"New overall best model saved (composite score: {trainer.best_composite_score:.4f})")
    
    return latest_path

def load_checkpoint(trainer, path: Optional[Path] = None) -> bool:
    """Load training checkpoint into trainer."""
    if path is None:
        path = config.DIRS["ckpt"] / "latest.pt"
    
    if not os.path.exists(path):
        config.logger.warning(f"No checkpoint found at {path}")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
        
        # Check for size/config mismatch
        if 'config' in checkpoint:
            saved_cfg = checkpoint['config']
            if saved_cfg.get('IMG_SIZE') != config.IMG_SIZE:
                config.logger.warning(f"IMG_SIZE mismatch: checkpoint={saved_cfg.get('IMG_SIZE')}, current={config.IMG_SIZE}")
        
        trainer.vae.load_state_dict(checkpoint['vae_state'], strict=False)
        trainer.drift.load_state_dict(checkpoint['drift_state'], strict=False)
        if 'text_encoder_state' in checkpoint and hasattr(trainer, 'text_encoder') and checkpoint['text_encoder_state'] is not None:
            trainer.text_encoder.load_state_dict(checkpoint['text_encoder_state'], strict=False)
        
        try:
            trainer.opt_vae.load_state_dict(checkpoint['opt_vae_state'])
            trainer.opt_drift.load_state_dict(checkpoint['opt_drift_state'])
            if 'opt_text_state' in checkpoint and hasattr(trainer, 'opt_text') and checkpoint['opt_text_state'] is not None:
                trainer.opt_text.load_state_dict(checkpoint['opt_text_state'])
            trainer.scheduler_vae.load_state_dict(checkpoint['scheduler_vae_state'])
            trainer.scheduler_drift.load_state_dict(checkpoint['scheduler_drift_state'])
        except Exception as opt_e:
            config.logger.warning(f"Failed to load optimizer/scheduler states (likely due to architecture change): {opt_e}")
            config.logger.info("Continuing with fresh optimizer states.")
        
        trainer.epoch = checkpoint['epoch']
        trainer.step = checkpoint['step']
        trainer.phase = checkpoint.get('phase', 1)
        trainer.phase2_start_epoch = checkpoint.get('phase2_start_epoch', None)
        
        # Load or recreate reference VAE
        if trainer.phase >= 2:
            from models import LabelConditionedVAE
            trainer.vae_ref = LabelConditionedVAE().to(config.DEVICE)
            
            if 'vae_ref_state' in checkpoint:
                trainer.vae_ref.load_state_dict(checkpoint['vae_ref_state'])
                config.logger.info("Reference anchor loaded from checkpoint.")
            else:
                trainer.vae_ref.load_state_dict(trainer.vae.state_dict())
                config.logger.info("Reference anchor recreated from VAE state (no saved anchor found).")
                
            trainer.vae_ref.eval()
            for param in trainer.vae_ref.parameters():
                param.requires_grad = False
        
        if 'training_schedule' in checkpoint:
            config.TRAINING_SCHEDULE.update(checkpoint['training_schedule'])
        
        trainer.best_loss = checkpoint.get('best_loss', float('inf'))
        trainer.best_composite_score = checkpoint.get('best_composite_score', float('-inf'))
        
        if 'kpi_metrics' in checkpoint and hasattr(trainer, 'kpi_tracker'):
            trainer.kpi_tracker.metrics = checkpoint['kpi_metrics']
        
        config.logger.info(f"Loaded checkpoint from epoch {trainer.epoch}, step {trainer.step}")
        return True
        
    except Exception as e:
        config.logger.error(f"Failed to load checkpoint: {e}")
        return False

def load_for_inference(trainer, path: Optional[Path] = None) -> bool:
    """Load only model weights for inference (ignore optimizer states)."""
    if path is None:
        path = config.DIRS["ckpt"] / "latest.pt"
    
    if not os.path.exists(path):
        config.logger.error(f"No checkpoint found at {path}")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)
        trainer.vae.load_state_dict(checkpoint['vae_state'])
        trainer.drift.load_state_dict(checkpoint['drift_state'])
        trainer.epoch = checkpoint.get('epoch', 0)
        trainer.vae.eval()
        trainer.drift.eval()
        config.logger.info(f" Successfully loaded model from epoch {trainer.epoch}")
        return True
    except Exception as e:
        config.logger.error(f"Failed to load checkpoint: {e}")
        return False

# ============================================================
# PNG GENERATION UTILITIES
# ============================================================
def save_image_grid(
    images: torch.Tensor, 
    path: Path, 
    nrow: int = 4, 
    normalize: bool = True
) -> None:
    """Save a grid of images to PNG, respecting config.GEN_SIZE."""
    if normalize:
        images_display = (images + 1) / 2
        images_display = torch.clamp(images_display, 0, 1)
    else:
        images_display = images
        
    if config.GEN_SIZE != config.IMG_SIZE:
        images_display = torch.nn.functional.interpolate(
            images_display, size=(config.GEN_SIZE, config.GEN_SIZE), mode='bilinear', align_corners=False
        )
        
    grid = vutils.make_grid(images_display, nrow=nrow, padding=2)
    vutils.save_image(grid, path)

def save_individual_images(
    images: torch.Tensor, 
    labels: List[int], 
    epoch: int, 
    base_dir: Optional[Path] = None
) -> None:
    """Save each image individually, respecting config.GEN_SIZE."""
    if base_dir is None:
        base_dir = config.DIRS["samples"]
        
    if config.GEN_SIZE != config.IMG_SIZE:
        images = torch.nn.functional.interpolate(
            images, size=(config.GEN_SIZE, config.GEN_SIZE), mode='bilinear', align_corners=False
        )
        
    for idx, img in enumerate(images):
        individual_path = base_dir / f"gen_{idx}_label{labels[idx]}_epoch{epoch+1}.png"
        vutils.save_image(img, individual_path)

def save_raw_tensors(
    z: torch.Tensor, 
    images: torch.Tensor, 
    labels: List[int], 
    epoch: int, 
    base_dir: Optional[Path] = None
) -> Path:
    """Save raw latent and image tensors for debugging."""
    if base_dir is None:
        base_dir = config.DIRS["samples"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"raw_epoch{epoch+1}_{timestamp}.pt"
    torch.save({
        'z': z.cpu(),
        'images': images.cpu(),
        'labels': labels
    }, path)
    return path

# ============================================================
# DATASET WITH LABELS
# ============================================================
class LabeledImageDataset(Dataset):
    """Dataset wrapper that provides labels."""
    
    def __init__(self, base_dataset, transform=None):
        self.dataset = base_dataset
        self.transform = transform
        
        if hasattr(base_dataset, 'classes'):
            self.classes = base_dataset.classes
        else:
            self.classes = [f"class_{i}" for i in range(10)]
        
        self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
        self.reverse_map = {idx: cls for cls, idx in self.label_map.items()}
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        
        if isinstance(item, tuple):
            img, label_idx = item
            label_text = self.classes[label_idx] if label_idx < len(self.classes) else f"class_{label_idx}"
        else:
            img = item
            label_idx = idx % len(self.classes)
            label_text = self.classes[label_idx]
        
        if self.transform:
            img = self.transform(img)
        
        data = {
            'image': img,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'label_text': label_text,
            'index': idx
        }
        
        # In this prototype, we use the label index as a token ID for the TextEncoder
        if config.USE_MULTIMODAL:
            # Shift labels by 1 so 0 is padding, 1-10 are labels, VOCAB_SIZE-1 is EOS
            tokens = torch.zeros(config.MAX_TEXT_LENGTH, dtype=torch.long)
            # Ensure label fits within vocab (minus EOS)
            safe_label = min(label_idx + 1, config.CLIP_VOCAB_SIZE - 2)
            tokens[0] = safe_label
            tokens[1] = config.CLIP_VOCAB_SIZE - 1 # EOS token
            data['text_tokens'] = tokens
            
        return data

# ============================================================
# config.DEVICE-COMPATIBLE DATALOADER SETTINGS
# ============================================================
def get_dataloader_config() -> Dict:
    """Get dataloader configuration optimized for current device."""
    config_dict = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'drop_last': True,
    }
    
    if config.DEVICE.type == 'cuda':
        config_dict.update({
            'num_workers': 4 if os.cpu_count() > 4 else 2,
            'pin_memory': True,
            'persistent_workers': True,
        })
    elif config.DEVICE.type == 'mps':
        config_dict.update({
            'num_workers': 0,
            'pin_memory': False,
        })
    elif config.DEVICE.type in ['xpu', 'directml', 'privateuseone']:
        config_dict.update({
            'num_workers': 2,
            'pin_memory': True,
        })
    else:  # CPU
        config_dict.update({
            'num_workers': 0,
            'pin_memory': False,
        })
    
    return config_dict

# ============================================================
# DATA LOADING FUNCTION
# ============================================================
def load_data() -> DataLoader:
    """Load dataset based on config.DATASET_NAME with device-optimized settings."""
    
    # ========== ENHANCED DATA AUGMENTATION ==========
    transform = T.Compose([
        T.Resize((config.IMG_SIZE + 16, config.IMG_SIZE + 16)),  # Slightly larger
        T.RandomCrop(config.IMG_SIZE),  # Random crop for variation
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,hue=0.05),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Occlusion augmentation
    ])
    
    config.logger.info(f"Loading dataset: {config.DATASET_NAME} from {config.DATASET_PATH}...")
    
    try:
        if config.DATASET_NAME == "STL10":
            base_ds = datasets.STL10(root=str(config.DATASET_PATH), split='train', download=True, transform=transform)
            if not hasattr(base_ds, 'classes'):
                base_ds.classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        
        elif config.DATASET_NAME == "CIFAR10":
            base_ds = datasets.CIFAR10(root=str(config.DATASET_PATH), train=True, download=True, transform=transform)
            
        elif config.DATASET_NAME == "CUSTOM":
            if not config.DATASET_PATH.exists():
                raise FileNotFoundError(f"Custom data path {config.DATASET_PATH} does not exist!")
            base_ds = datasets.ImageFolder(root=str(config.DATASET_PATH), transform=transform)
            
        else:
            raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")
            
        dataset = LabeledImageDataset(base_ds)
        config.logger.info(f"Loaded {len(dataset)} images from {config.DATASET_NAME}")
        
    except Exception as e:
        config.logger.error(f"Failed to load dataset {config.DATASET_NAME}: {e}")
        raise e
    
    dataloader_config = get_dataloader_config()
    return DataLoader(dataset, **dataloader_config)
