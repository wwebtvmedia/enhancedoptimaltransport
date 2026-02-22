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
# USE CENTRALIZED CONFIGURATION
# ============================================================
IMG_SIZE = config.IMG_SIZE
LATENT_CHANNELS = config.LATENT_CHANNELS
LATENT_H = config.LATENT_H
LATENT_W = config.LATENT_W
BATCH_SIZE = config.BATCH_SIZE
DEVICE = config.DEVICE
LR = config.LR
EPOCHS = config.EPOCHS
SNAPSHOT_INTERVAL = config.SNAPSHOT_INTERVAL
SNAPSHOT_KEEP = config.SNAPSHOT_KEEP
NUM_CLASSES = config.NUM_CLASSES
LABEL_EMB_DIM = config.LABEL_EMB_DIM
DIRS = config.DIRS
logger = config.logger

# ============================================================
# SNAPSHOT MANAGER
# ============================================================
class SnapshotManager:
    """Manage model snapshots for snapshot ensemble learning."""
    
    def __init__(self, model, optimizer, name="model", interval=SNAPSHOT_INTERVAL, keep=SNAPSHOT_KEEP):
        self.model = model
        self.optimizer = optimizer
        self.name = name
        self.interval = interval
        self.keep = keep
        self.snapshots = []
        self.snapshot_dir = DIRS["snaps"]
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=interval // 2,
            T_mult=1,
            eta_min=LR * 0.1
        )
    
    def step(self) -> None:
        """Update snapshot scheduler."""
        self.scheduler.step()
    
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
        
        logger.info(f"Saved {self.name} snapshot at epoch {epoch}")
        return snapshot_path

    def revert(self) -> bool:
        """Emergency revert to last good snapshot."""
        if not self.snapshots:
            logger.error(f"❌ No snapshots available for {self.name} to revert to!")
            return False
        
        latest_snapshot = self.snapshots[-1]
        try:
            snapshot = torch.load(latest_snapshot, map_location=DEVICE, weights_only=False)
            
            if self.name == "drift" and 'drift_state' in snapshot:
                self.model.load_state_dict(snapshot['drift_state'])
                if 'opt_drift_state' in snapshot:
                    self.optimizer.load_state_dict(snapshot['opt_drift_state'])
            elif 'model_state' in snapshot:
                self.model.load_state_dict(snapshot['model_state'])
                if 'optimizer_state' in snapshot:
                    self.optimizer.load_state_dict(snapshot['optimizer_state'])
            else:
                logger.error(f" Invalid snapshot format for {self.name}")
                return False
            
            logger.info(f" Emergency revert successful for {self.name} to epoch {snapshot.get('epoch', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f" Emergency revert failed: {e}")
            return False

# ============================================================
# CHECKPOINT SAVE/LOAD FUNCTIONS
# ============================================================
def save_checkpoint(trainer, is_best: bool = False, is_best_overall: bool = False) -> Path:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': trainer.epoch,
        'step': trainer.step,
        'phase': trainer.phase,
        'vae_state': trainer.vae.state_dict(),
        'drift_state': trainer.drift.state_dict(),
        'opt_vae_state': trainer.opt_vae.state_dict(),
        'opt_drift_state': trainer.opt_drift.state_dict(),
        'scheduler_vae_state': trainer.scheduler_vae.state_dict(),
        'scheduler_drift_state': trainer.scheduler_drift.state_dict(),
        'best_loss': trainer.best_loss,
        'best_composite_score': trainer.best_composite_score,
        'kpi_metrics': trainer.kpi_tracker.metrics,
        'training_schedule': config.TRAINING_SCHEDULE
    }
    
    latest_path = DIRS["ckpt"] / "latest.pt"
    torch.save(checkpoint, latest_path)
    logger.info(f"Checkpoint saved to {latest_path}")
    
    if is_best:
        best_path = DIRS["ckpt"] / "best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved (loss: {trainer.best_loss:.4f})")
    
    if is_best_overall:
        overall_best_path = DIRS["best"] / f"best_overall_epoch_{trainer.epoch+1:04d}.pt"
        torch.save(checkpoint, overall_best_path)
        logger.info(f"New overall best model saved (composite score: {trainer.best_composite_score:.4f})")
    
    return latest_path

def load_checkpoint(trainer, path: Optional[Path] = None) -> bool:
    """Load training checkpoint into trainer."""
    if path is None:
        path = DIRS["ckpt"] / "latest.pt"
    
    if not os.path.exists(path):
        logger.warning(f"No checkpoint found at {path}")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        trainer.vae.load_state_dict(checkpoint['vae_state'])
        trainer.drift.load_state_dict(checkpoint['drift_state'])
        trainer.opt_vae.load_state_dict(checkpoint['opt_vae_state'])
        trainer.opt_drift.load_state_dict(checkpoint['opt_drift_state'])
        trainer.scheduler_vae.load_state_dict(checkpoint['scheduler_vae_state'])
        trainer.scheduler_drift.load_state_dict(checkpoint['scheduler_drift_state'])
        
        trainer.epoch = checkpoint['epoch']
        trainer.step = checkpoint['step']
        trainer.phase = checkpoint.get('phase', 1)
        
        # Handle Phase 2 reference model
        if trainer.phase == 2 and not hasattr(trainer, 'vae_ref'):
            from models import LabelConditionedVAE
            trainer.vae_ref = LabelConditionedVAE().to(DEVICE)
            trainer.vae_ref.load_state_dict(trainer.vae.state_dict())
            trainer.vae_ref.eval()
            for param in trainer.vae_ref.parameters():
                param.requires_grad = False
            logger.info("Reference anchor created from loaded Phase 2 VAE.")
        
        if 'training_schedule' in checkpoint:
            config.TRAINING_SCHEDULE.update(checkpoint['training_schedule'])
        
        trainer.best_loss = checkpoint.get('best_loss', float('inf'))
        trainer.best_composite_score = checkpoint.get('best_composite_score', float('-inf'))
        trainer.kpi_tracker.metrics = checkpoint.get('kpi_metrics', {})
        
        logger.info(f"Loaded checkpoint from epoch {trainer.epoch}, step {trainer.step}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False

def load_for_inference(trainer, path: Optional[Path] = None) -> bool:
    """Load only model weights for inference (ignore optimizer states)."""
    if path is None:
        path = DIRS["ckpt"] / "latest.pt"
    
    if not os.path.exists(path):
        logger.error(f"No checkpoint found at {path}")
        return False
    
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        trainer.vae.load_state_dict(checkpoint['vae_state'])
        trainer.drift.load_state_dict(checkpoint['drift_state'])
        
        trainer.epoch = checkpoint.get('epoch', 0)
        
        trainer.vae.eval()
        trainer.drift.eval()
        
        logger.info(f"✓ Successfully loaded model from epoch {trainer.epoch}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
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
    """Save a grid of images to PNG.
    
    Args:
        images: Tensor of shape (B, C, H, W) in range [-1, 1] if normalize=True
        path: Path to save PNG
        nrow: Number of images per row
        normalize: If True, map [-1,1] to [0,1]
    """
    if normalize:
        images_display = (images + 1) / 2
        images_display = torch.clamp(images_display, 0, 1)
    else:
        images_display = images
    grid = vutils.make_grid(images_display, nrow=nrow, padding=2)
    vutils.save_image(grid, path)

def save_individual_images(
    images: torch.Tensor, 
    labels: List[int], 
    epoch: int, 
    base_dir: Optional[Path] = None
) -> None:
    """Save each image individually with label in filename.
    
    Args:
        images: Tensor of shape (B, C, H, W) in [0, 1] range
        labels: List of integer labels
        epoch: Current epoch (for filename)
        base_dir: Directory to save (default: DIRS["samples"])
    """
    if base_dir is None:
        base_dir = DIRS["samples"]
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
        base_dir = DIRS["samples"]
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
        
        return {
            'image': img,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'label_text': label_text,
            'index': idx
        }

# ============================================================
# DEVICE-COMPATIBLE DATALOADER SETTINGS
# ============================================================
def get_dataloader_config() -> Dict:
    """Get dataloader configuration optimized for current device."""
    config_dict = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'drop_last': True,
    }
    
    if DEVICE.type == 'cuda':
        config_dict.update({
            'num_workers': 4 if os.cpu_count() > 4 else 2,
            'pin_memory': True,
        })
    elif DEVICE.type == 'mps':
        config_dict.update({
            'num_workers': 0,
            'pin_memory': False,
        })
    elif DEVICE.type == 'directml':
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
    """Load dataset with device-optimized settings."""
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,)*3, (0.5,)*3)
    ])
    
    datasets_list = []
    
    local_path = Path("./data/images")
    if local_path.exists():
        logger.info("Loading local images...")
        try:
            local_ds = datasets.ImageFolder(str(local_path), transform=transform)
            datasets_list.append(LabeledImageDataset(local_ds))
            logger.info(f"Loaded {len(local_ds)} local images with {len(local_ds.classes)} classes")
        except Exception as e:
            logger.warning(f"Failed to load local images: {e}")
    
    logger.info("Loading CIFAR-10...")
    try:
        cifar_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        labeled_cifar = LabeledImageDataset(cifar_ds)
        datasets_list.append(labeled_cifar)
        logger.info(f"Loaded {len(cifar_ds)} CIFAR-10 images")
    except Exception as e:
        logger.error(f"Failed to load CIFAR-10: {e}")
        if not datasets_list:
            raise RuntimeError("No datasets could be loaded!")
    
    if len(datasets_list) > 1:
        combined_ds = ConcatDataset(datasets_list)
    else:
        combined_ds = datasets_list[0]
    
    logger.info(f"Total dataset size: {len(combined_ds)} images")
    
    dataloader_config = get_dataloader_config()
    
    return DataLoader(combined_ds, **dataloader_config)