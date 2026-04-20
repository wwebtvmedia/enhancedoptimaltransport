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
        if mu_std < 0.6:
            # Low diversity (risk of collapse), boost diversity loss
            config.DIVERSITY_WEIGHT = min(3.0, config.DIVERSITY_WEIGHT * 1.1)
        elif mu_std > 1.1:
            # Too much chaos (potential periodic motifs), relax diversity
            config.DIVERSITY_WEIGHT = max(0.1, config.DIVERSITY_WEIGHT * 0.9)

        # 4. SSIM_WEIGHT (Monitor Structural Integrity)
        ssim_loss = losses.get('ssim_loss', 0.3)
        if ssim_loss > 0.35:
            # Blurry detected, aggressively increase importance of structural loss
            config.SSIM_WEIGHT = min(6.0, config.SSIM_WEIGHT * 1.1)
        elif ssim_loss < 0.2:
            # High quality, can slightly relax to favor other metrics
            config.SSIM_WEIGHT = max(0.5, config.SSIM_WEIGHT * 0.95)

        # 5. ANTI-ARTIFACT CONTROL (NEW: Automatic Langevin Tuning)
        # Uses SSIM and Drift as proxies for reconstruction artifacts
        old_l_steps = config.DEFAULT_LANGEVIN_STEPS
        old_l_scale = config.LANGEVIN_SCORE_SCALE
        
        if ssim_loss > 0.28:
            # Structural noise detected: Increase smoothing refinement
            config.DEFAULT_LANGEVIN_STEPS = min(120, config.DEFAULT_LANGEVIN_STEPS + 5)
            config.LANGEVIN_SCORE_SCALE = max(0.15, config.LANGEVIN_SCORE_SCALE * 0.9)
            # Also lower CFG to reduce artifact amplification
            config.CFG_SCALE = max(2.0, config.CFG_SCALE - 0.2)
        elif ssim_loss < 0.18 and current_loss < 1.0:
            # High fidelity: Can afford to reduce refinement steps for speed
            config.DEFAULT_LANGEVIN_STEPS = max(30, config.DEFAULT_LANGEVIN_STEPS - 2)
            config.LANGEVIN_SCORE_SCALE = min(0.6, config.LANGEVIN_SCORE_SCALE * 1.05)
            
        # Log significant changes to the terminal
        if (abs(config.DRIFT_WEIGHT - old_drift) > 0.01 or 
            abs(config.CFG_SCALE - old_cfg) > 0.1 or
            abs(config.DIVERSITY_WEIGHT - old_div) > 0.05 or
            abs(config.SSIM_WEIGHT - old_ssim) > 0.05 or
            abs(config.DEFAULT_LANGEVIN_STEPS - old_l_steps) >= 5 or
            abs(config.LANGEVIN_SCORE_SCALE - old_l_scale) > 0.02):
            
            config.logger.info(f"📊 [App Control] Dynamic Update: DriftW={config.DRIFT_WEIGHT:.2f}, CFG={config.CFG_SCALE:.1f}, "
                               f"SSIMW={config.SSIM_WEIGHT:.2f}, LangevinSteps={config.DEFAULT_LANGEVIN_STEPS}, "
                               f"LangevinScale={config.LANGEVIN_SCORE_SCALE:.2f}")

    def _run_autonomous_strategy(self, epoch: int, losses: Dict[str, Any]):
        """
        Autonomous training management:
        - KPI-based phase transitions (1 -> 2 -> 3).
        - Stochastic nudges/restarts to avoid local minima, scaled by temperature.
        """
        if not self.trainer:
            return

        current_phase = self.trainer.phase
        # Use temperature as a proxy for 'settledness'
        temp = losses.get('temperature', 0.5)
        
        # --- 1. PHASE TRANSITION LOGIC ---
        # Goal: Optimize 1 -> 2 -> 3 transition based on quality targets
        
        if current_phase == 1:
            snr = losses.get('snr', 0)
            ssim = losses.get('ssim_loss', 1.0)
            # Transition to Drift Matching if VAE is sharp and stable
            if snr > 20.0 and ssim < 0.26 and epoch >= 80:
                config.logger.info(f"✨ [Auto-Strategy] VAE Quality Target Reached (SNR: {snr:.1f}dB, SSIM: {ssim:.3f}).")
                config.logger.info("🚀 Transitioning to Phase 2: Drift Matching.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 2
                
        elif current_phase == 2:
            drift = losses.get('drift', 10.0)
            ssim = losses.get('ssim_loss', 1.0)
            # Transition to Joint Fine-tuning if Bridge is accurate
            if drift < 1.25 and ssim < 0.21 and epoch >= 160:
                config.logger.info(f"✨ [Auto-Strategy] Drift Stability Target Reached (Loss: {drift:.3f}).")
                config.logger.info("🚀 Transitioning to Phase 3: Joint Fine-tuning.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 3

        # --- 2. STOCHASTIC RESTART / NUDGE ---
        # Probability peaks when temperature is high (searching harder when hot)
        nudge_prob = 0.03 * (temp + 0.5) 
        
        if random.random() < nudge_prob:
            config.logger.info(f"🎲 [Stochastic Control] Triggering quality nudge (p={nudge_prob:.3f}, temp={temp:.2f})")
            
            # "Increase a little bit" - temporary boost to learning intensity
            config.DRIFT_WEIGHT = min(2.8, config.DRIFT_WEIGHT * 1.12)
            config.RECON_WEIGHT = min(16.0, config.RECON_WEIGHT * 1.08)
            
            # "Restart" capability: small chance to go back a phase if in Phase 3
            if current_phase == 3 and random.random() < 0.15:
                config.logger.info("↩️ [Stochastic Control] Momentum Reset: Back-switching to Phase 2 for trajectory correction.")
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
