# ============================================================================
# PROCESSING LAYER - Core Training and Model Logic
# ============================================================================

import os
import sys
import torch
import threading
import random
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
        Refined KPI-driven logic to adjust parameters and training mode.
        Fixes "Fade and Blurry" issues by balancing reconstruction vs smoothing.
        """
        if not self.trainer: return

        # 1. KPI EXTRACTION
        drift_loss = losses.get('drift', 5.0)
        ssim_loss = losses.get('ssim_loss', 0.3)
        mu_std = losses.get('mu_std', 0.8)
        comp_score = losses.get('composite_score', -50)
        
        # Stability Baseline
        STABILITY_LIMIT = 5.0
        
        # Store old values for logging
        prev = {
            'dw': config.DRIFT_WEIGHT, 'rw': config.RECON_WEIGHT,
            'cfg': config.CFG_SCALE, 'sw': config.SSIM_WEIGHT,
            'ls': config.DEFAULT_LANGEVIN_STEPS, 'lc': config.LANGEVIN_SCORE_SCALE
        }

        # 2. FIX FADE & BLURRY (The "Contrast & Detail" Block)
        if ssim_loss > 0.25 or comp_score < -40:
            # BLURRY/FADE: Increase intensity and guidance
            # Pro: Restores lost details and contrast. Con: Can increase noise.
            config.CFG_SCALE = min(14.0, config.CFG_SCALE + 0.4)
            config.RECON_WEIGHT = min(25.0, config.RECON_WEIGHT * 1.1)
            config.LANGEVIN_SCORE_SCALE = min(0.8, config.LANGEVIN_SCORE_SCALE * 1.1)
            
            # If weight is too high, it causes "averaging" blur, so we pull back slightly
            if config.SSIM_WEIGHT > 5.0:
                 config.SSIM_WEIGHT *= 0.9
        
        elif ssim_loss < 0.15:
            # SHARP: Can afford to relax and increase diversity
            # Pro: Faster training, more varied samples. Con: Risk of losing focus.
            config.CFG_SCALE = max(2.5, config.CFG_SCALE - 0.2)
            config.RECON_WEIGHT = max(2.0, config.RECON_WEIGHT * 0.95)
            config.SSIM_WEIGHT = max(0.5, config.SSIM_WEIGHT * 0.98)

        # 3. ANTI-ARTIFACT & STABILITY (The "Noise & Chaos" Block)
        if mu_std > 1.2 or drift_loss > STABILITY_LIMIT * 0.8:
            # NOISY/INSTABLE: Increase smoothing and lower intensity
            # Pro: Prevents NaN and "grainy" images. Con: Makes images softer.
            config.DRIFT_WEIGHT = max(0.5, config.DRIFT_WEIGHT * 0.85)
            config.DEFAULT_LANGEVIN_STEPS = min(150, config.DEFAULT_LANGEVIN_STEPS + 10)
            config.LANGEVIN_SCORE_SCALE *= 0.9
        
        elif mu_std < 0.7 and drift_loss < 1.0:
            # STABLE: Push for faster learning
            # Pro: Quick convergence. Con: Risk of overshooting.
            config.DRIFT_WEIGHT = min(3.0, config.DRIFT_WEIGHT * 1.05)
            config.DEFAULT_LANGEVIN_STEPS = max(40, config.DEFAULT_LANGEVIN_STEPS - 5)

        # 4. TRAINING TYPE JITTER (VAE vs DRIFT vs BOTH)
        # Randomly switch focus to prevent the model from "forgetting" one part
        # Pro: Keeps both networks fresh. Con: Short-term loss fluctuations.
        if random.random() < 0.03: # 3% chance per epoch
            current_mode = config.TRAINING_SCHEDULE.get('force_phase', 3)
            # Weights: 10% VAE only, 10% Drift only, 80% Joint
            r = random.random()
            if r < 0.10:
                new_mode = 1 # VAE Focus (Fixes "Fade" by retraining decoder)
                config.logger.info("🎲 [App Control] Jitter: VAE Focus mode enabled.")
            elif r < 0.20:
                new_mode = 2 # Drift Focus (Fixes "Structural error" in bridge)
                config.logger.info("🎲 [App Control] Jitter: Drift Focus mode enabled.")
            else:
                new_mode = 3 # Joint Fine-tuning
                config.logger.info("🎲 [App Control] Jitter: Joint Fine-tuning restored.")
            
            config.TRAINING_SCHEDULE['force_phase'] = new_mode

        # 5. BASELINE RESTORATION (The "Panic" Button)
        # If score is catastrophically low, reset to safe defaults
        if comp_score < -100:
            config.logger.warning("🚨 [App Control] KPI COLLAPSE: Restoring stable baselines.")
            config.CFG_SCALE = 5.0
            config.DRIFT_WEIGHT = 1.0
            config.RECON_WEIGHT = 5.0
            config.SSIM_WEIGHT = 1.0
            config.DEFAULT_LANGEVIN_STEPS = 60
            config.LANGEVIN_SCORE_SCALE = 0.4

        # Log changes
        if any(abs(getattr(config, k.upper()) - prev[k.lower()[:2]]) > 0.01 for k in prev):
             config.logger.info(f"📊 [App Control] Update: DW={config.DRIFT_WEIGHT:.2f}, RW={config.RECON_WEIGHT:.1f}, "
                                f"CFG={config.CFG_SCALE:.1f}, SSIMW={config.SSIM_WEIGHT:.2f}, LSteps={config.DEFAULT_LANGEVIN_STEPS}")

    def _run_autonomous_strategy(self, epoch: int, losses: Dict[str, Any]):
        """
        Autonomous training management:
        - KPI-based phase transitions (1 -> 2 -> 3).
        - Stochastic nudges/restarts to avoid local minima.
        """
        if not self.trainer:
            return

        current_phase = self.trainer.phase
        temp = losses.get('temperature', 0.5)
        
        # --- 1. PHASE TRANSITION LOGIC ---
        if current_phase == 1:
            snr = losses.get('snr', 0)
            ssim = losses.get('ssim_loss', 1.0)
            if snr > 20.0 and ssim < 0.26 and epoch >= 80:
                config.logger.info(f"✨ [Auto-Strategy] VAE Sharpness reached. Transitioning to Phase 2.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 2
                
        elif current_phase == 2:
            drift = losses.get('drift', 10.0)
            ssim = losses.get('ssim_loss', 1.0)
            if drift < 1.25 and ssim < 0.21 and epoch >= 160:
                config.logger.info(f"✨ [Auto-Strategy] Drift Stability reached. Transitioning to Phase 3.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 3

        # --- 2. STOCHASTIC RESTART capability (Back-switching) ---
        # Very small proba to go back a phase if fine-tuning is failing
        if current_phase == 3 and losses.get('composite_score', 0) < -60:
            if random.random() < 0.05:
                config.logger.info("↩️ [Auto-Strategy] Fine-tuning regression: Reverting to Phase 2 for trajectory fix.")
                config.TRAINING_SCHEDULE['force_phase'] = 2


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
                
                # --- STRATEGY ENGINE ---
                # Dynamic parameter scaling
                self._adjust_dynamic_parameters(epoch, losses)
                # Autonomous phase switching and stochastic nudges
                self._run_autonomous_strategy(epoch, losses)
                
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
