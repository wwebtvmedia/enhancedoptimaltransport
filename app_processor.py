# ============================================================================
# PROCESSING LAYER - Core Training and Model Logic
# ============================================================================

import os
import sys
import torch
import threading
from typing import Callable, Dict, Any, Optional

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training
import data_management as dm
import config

class TrainingProcessor:
    """The Engine that performs all the heavy computations."""
    
    def __init__(self, context):
        self.ctx = context
        self.trainer: Optional[training.EnhancedLabelTrainer] = None
        self._thread: Optional[threading.Thread] = None

    def initialize_hardware(self) -> str:
        """Determines best available hardware and updates context."""
        info = config.initialize_hardware()
        self.ctx.device_info = info
        return info

    def start_training(self, on_epoch_done: Optional[Callable] = None):
        """Launches the training thread."""
        if self.ctx.is_training:
            return
            
        self.ctx.is_training = True
        self.ctx.stop_signal = False
        self._thread = threading.Thread(target=self._run_loop, args=(on_epoch_done,), daemon=True)
        self._thread.start()

    def stop_training(self):
        self.ctx.stop_signal = True

    def _run_loop(self, on_epoch_done):
        try:
            loader = dm.load_data()
            self.trainer = training.EnhancedLabelTrainer(loader)
            
            # Auto-resume from checkpoint
            latest = self.ctx.config.DIRS["ckpt"] / "latest.pt"
            if latest.exists():
                self.trainer.load_checkpoint()

            for epoch in range(self.trainer.epoch, self.ctx.config.EPOCHS):
                if self.ctx.stop_signal:
                    break
                
                self.trainer.epoch = epoch
                losses = self.trainer.train_epoch()
                
                # Update context
                self.ctx.update_metric(epoch, losses)
                
                if on_epoch_done:
                    on_epoch_done(epoch, losses)
                
                # Periodically save/generate
                if (epoch+1) % 5 == 0:
                    self.trainer.save_checkpoint()
                if (epoch+1) % 10 == 0:
                    self.trainer.generate_samples()
                    # Signal to UI that new samples are available
                    self.ctx.log_queue.put("UPDATE_GALLERY")
                    
        except Exception as e:
            config.logger.error(f"Engine failure: {e}")
        finally:
            self.ctx.is_training = False
            self.ctx.stop_signal = False
