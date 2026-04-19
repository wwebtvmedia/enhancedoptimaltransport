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

    def start_training(self, on_epoch_done: Optional[Callable] = None, force_fresh: bool = False):
        """Launches the training thread."""
        if self.ctx.is_training:
            return
            
        self.ctx.is_training = True
        self.ctx.stop_signal = False
        self._thread = threading.Thread(target=self._run_loop, args=(on_epoch_done, force_fresh), daemon=True)
        self._thread.start()

    def stop_training(self):
        self.ctx.stop_signal = True

    def _adjust_dynamic_parameters(self, epoch: int, losses: Dict[str, Any]):
        """
        Application-layer logic to adjust model parameters based on training health.
        Monitors loss trends to push for sharpness while ensuring stability.
        """
        # Ensure we are in Phase 2 or 3 (where drift is active)
        if not self.trainer or self.trainer.phase < 2:
            return

        # Target metric for Schrödinger Bridge (Phase 2/3)
        current_loss = losses.get('drift', losses.get('total', 10.0))
        
        # requested 5.0 stability threshold
        STABILITY_LIMIT = 5.0
        
        old_drift = config.DRIFT_WEIGHT
        old_cfg = config.CFG_SCALE
        old_div = config.DIVERSITY_WEIGHT
        old_ssim = config.SSIM_WEIGHT
        
        # 1. DRIFT_WEIGHT adjustment for learning sharpness
        if current_loss < STABILITY_LIMIT * 0.4:
            # Very stable, push for even sharper gradients (max 2.5)
            config.DRIFT_WEIGHT = min(2.5, config.DRIFT_WEIGHT * 1.05)
        elif current_loss > STABILITY_LIMIT * 0.8:
            # Approaching danger zone, pull back for safety (min 0.5)
            config.DRIFT_WEIGHT = max(0.5, config.DRIFT_WEIGHT * 0.8)
            
        # 2. CFG_SCALE adjustment for generation sharpness
        if current_loss < STABILITY_LIMIT * 0.3:
            # Low loss implies good manifold learning, we can push guidance (max 12.0)
            config.CFG_SCALE = min(12.0, config.CFG_SCALE + 0.2)
        elif current_loss > STABILITY_LIMIT * 0.9:
            # High loss suggests unstable integration, dial down guidance
            config.CFG_SCALE = max(1.0, config.CFG_SCALE - 0.5)

        # 3. DIVERSITY_WEIGHT (Monitor Latent Variance)
        mu_std = losses.get('mu_std', 0.8)
        if mu_std < 0.5:
            # Low diversity (risk of collapse), boost diversity loss
            config.DIVERSITY_WEIGHT = min(5.0, config.DIVERSITY_WEIGHT * 1.1)
        elif mu_std > 1.2:
            # Too much chaos, relax diversity to favor reconstruction
            config.DIVERSITY_WEIGHT = max(0.1, config.DIVERSITY_WEIGHT * 0.9)

        # 4. SSIM_WEIGHT (Monitor Structural Integrity)
        ssim_loss = losses.get('ssim_loss', 0.3)
        if ssim_loss > 0.4:
            # Too blurry, increase importance of structural loss
            config.SSIM_WEIGHT = min(5.0, config.SSIM_WEIGHT * 1.05)
        elif ssim_loss < 0.15:
            # Already very sharp, can slightly relax
            config.SSIM_WEIGHT = max(0.5, config.SSIM_WEIGHT * 0.98)
            
        # Log significant changes to the terminal
        if (abs(config.DRIFT_WEIGHT - old_drift) > 0.01 or 
            abs(config.CFG_SCALE - old_cfg) > 0.1 or
            abs(config.DIVERSITY_WEIGHT - old_div) > 0.05 or
            abs(config.SSIM_WEIGHT - old_ssim) > 0.05):
            
            config.logger.info(f"📊 [App Control] Dynamic Update: DriftW={config.DRIFT_WEIGHT:.2f}, CFG={config.CFG_SCALE:.1f}, DivW={config.DIVERSITY_WEIGHT:.2f}, SSIMW={config.SSIM_WEIGHT:.2f}")

    def _run_loop(self, on_epoch_done, force_fresh=False):
        try:
            loader = dm.load_data()
            self.trainer = training.EnhancedLabelTrainer(loader)
            
            # Auto-resume from checkpoint unless force_fresh is True
            if not force_fresh:
                latest = self.ctx.config.DIRS["ckpt"] / "latest.pt"
                if latest.exists():
                    try:
                        self.trainer.load_checkpoint()
                    except Exception as e:
                        config.logger.error(f"Failed to auto-resume from checkpoint: {e}")
                        config.logger.info("Starting fresh training instead.")
            else:
                config.logger.info("Force fresh start requested. Skipping checkpoint load.")

            consecutive_failures = 0
            for epoch in range(self.trainer.epoch, self.ctx.config.EPOCHS):
                if self.ctx.stop_signal:
                    break
                
                self.trainer.epoch = epoch
                losses = self.trainer.train_epoch()
                
                # Check for critical failure
                total_loss = losses.get('total', 0)
                if total_loss >= 1e8:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        config.logger.error("🛑 Training stopped: 3 consecutive failed epochs (NaN/Inf).")
                        break
                else:
                    consecutive_failures = 0
                
                # Update context
                self.ctx.update_metric(epoch, losses)
                
                if on_epoch_done:
                    on_epoch_done(epoch, losses)
                
                # Periodically save/generate
                if (epoch+1) % 5 == 0:
                    self.trainer.save_checkpoint()
                if (epoch+1) % 10 == 0:
                    self.trainer.generate_reconstructions()
                    self.trainer.generate_samples()
                    # Signal to UI that new samples are available
                    self.ctx.log_queue.put("UPDATE_GALLERY")
                    
        except Exception as e:
            config.logger.error(f"Engine failure: {e}")
        finally:
            self.ctx.is_training = False
            self.ctx.stop_signal = False
