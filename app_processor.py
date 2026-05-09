# ============================================================================
# PROCESSING LAYER - Core Training and Model Logic
# ============================================================================

import os
import sys
import torch
import threading
import random
import json
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training
import data_management as dm
import config

class TrainingProcessor:
    """The Engine that performs all the heavy computations with Swarm Protocol support."""
    
    def __init__(self, context):
        self.ctx = context
        self.trainer: Optional[training.EnhancedLabelTrainer] = None
        self._thread: Optional[threading.Thread] = None
        # Unique ID for Swarm Protocol
        self.node_id = f"node_{os.getpid()}_{random.randint(1000, 9999)}"
        self.swarm_dir = Path("enhanced_label_sb/swarm")
        self.swarm_dir.mkdir(parents=True, exist_ok=True)
        self._bad_score_count = 0

    def _broadcast_swarm_status(self, epoch: int, losses: Dict[str, Any]):
        """Broadcast current node state to the swarm."""
        status = {
            "id": self.node_id,
            "last_seen": time.time(),
            "epoch": epoch,
            "phase": getattr(self.trainer, 'phase', 1),
            "total": float(losses.get('total', 0)),
            "drift": float(losses.get('drift', 0)),
            "ssim": float(losses.get('ssim_loss', 0)),
            "mu_std": float(losses.get('mu_std', 0)),
            "score": float(losses.get('composite_score', -100)),
            "params": {
                "dw": config.DRIFT_WEIGHT,
                "cfg": config.CFG_SCALE,
                "sw": config.SSIM_WEIGHT
            }
        }
        try:
            with open(self.swarm_dir / f"{self.node_id}.json", 'w') as f:
                json.dump(status, f)
        except Exception as e:
            config.logger.error(f"Swarm broadcast failed: {e}")

    def _listen_for_swarm_commands(self):
        """Check for global overrides from the swarm supervisor."""
        override_file = self.swarm_dir / "global_override.json"
        if override_file.exists():
            try:
                with open(override_file, 'r') as f:
                    cmd = json.load(f)
                
                # Only apply if recent (within TTL)
                if time.time() - cmd.get('timestamp', 0) < config.SWARM_COMMAND_TTL:
                    config.CFG_SCALE = cmd.get('cfg', config.CFG_SCALE)
                    config.DRIFT_WEIGHT = cmd.get('dw', config.DRIFT_WEIGHT)
                    config.logger.info(f"🛰️ [Swarm] Global Override applied: CFG={config.CFG_SCALE}, DW={config.DRIFT_WEIGHT}")
                
                # Delete after reading to prevent infinite loops (one-time command)
                # Note: In a real multi-node setup, this would be a shared pub/sub
            except:
                pass

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
        Bi-directional: Can decrease values if performance is already high.
        """
        if not self.trainer: return

        # 1. KPI EXTRACTION
        drift_loss = losses.get('drift', 5.0)
        ssim_loss = losses.get('ssim_loss', 0.6)
        mu_std = losses.get('mu_std', 0.8)
        comp_score = losses.get('composite_score', -30)
        sharpness = losses.get('sharpness', 0.08)
        
        # Stability Baseline
        STABILITY_LIMIT = config.KPI_STABILITY_LIMIT
        
        # Store old values for logging
        prev = {
            'dw': config.DRIFT_WEIGHT, 'rw': config.RECON_WEIGHT,
            'cfg': config.CFG_SCALE, 'sw': config.SSIM_WEIGHT,
            'ls': config.DEFAULT_LANGEVIN_STEPS, 'lc': config.LANGEVIN_SCORE_SCALE
        }

        # 2. SSIM-BASED STRUCTURAL CONTROL (Realistic Thresholds for Phase 3)
        # SSIM around 0.6 is common in Phase 3.
        if ssim_loss > config.SSIM_BLUR_THRESHOLD:
            # BLURRY: Increase intensity
            # Pro: Sharpens boundaries. Con: Risk of artifacts.
            config.RECON_WEIGHT = min(config.MAX_RECON_WEIGHT, config.RECON_WEIGHT * 1.05)
            config.CFG_SCALE = min(config.MAX_CFG_SCALE, config.CFG_SCALE + 0.2)
        elif ssim_loss < config.SSIM_SHARP_THRESHOLD:
            # SHARP: Can relax to favor diversity
            # Pro: More creative generations. Con: Risk of getting "soft".
            config.RECON_WEIGHT = max(config.MIN_RECON_WEIGHT, config.RECON_WEIGHT * 0.95)
            config.CFG_SCALE = max(config.MIN_CFG_SCALE, config.CFG_SCALE - 0.1)

        # 3. SHARPNESS-DRIVEN CONTROL (Direct gradient monitoring)
        # Target sharpness range: 0.08 - 0.15
        if sharpness < config.SHARPNESS_BLUR_THRESHOLD:
            # BLURRY: Increase CFG and Langevin influence
            config.CFG_SCALE = min(config.MAX_CFG_SCALE, config.CFG_SCALE + 0.3)
            config.LANGEVIN_SCORE_SCALE = min(config.MAX_LANGEVIN_SCORE_SCALE, config.LANGEVIN_SCORE_SCALE + 0.05)
            config.logger.info(f"🔍 [App Control] Detected Blur (Sharp={sharpness:.4f}): Boosting CFG/LScale")
        elif sharpness > config.SHARPNESS_SHARP_THRESHOLD:
            # FRIED/ARTIFACTS: Pull back to restore structural integrity
            config.CFG_SCALE = max(config.MIN_CFG_SCALE, config.CFG_SCALE - 0.4)
            config.LANGEVIN_SCORE_SCALE = max(config.MIN_LANGEVIN_SCORE_SCALE, config.LANGEVIN_SCORE_SCALE - 0.05)
            config.logger.info(f"🔍 [App Control] Detected Over-sharpening (Sharp={sharpness:.4f}): Reducing CFG/LScale")

        # 4. CONTRAST & DETAIL (Based on Score Trend)
        # If score is dropping, we might be "over-cooking"
        if comp_score < -45:
            # QUALITY DROP: Pull back slightly
            # Pro: Prevents burn-in artifacts. Con: Temporary lower contrast.
            config.CFG_SCALE = max(config.MIN_CFG_SCALE, config.CFG_SCALE * 0.9)
            config.RECON_WEIGHT = max(config.MIN_RECON_WEIGHT, config.RECON_WEIGHT * 0.9)
            config.LANGEVIN_SCORE_SCALE = max(config.MIN_LANGEVIN_SCORE_SCALE, config.LANGEVIN_SCORE_SCALE * 0.9)
        elif comp_score > -25:
            # HIGH QUALITY: Can push for extra detail
            # Pro: Professional-level micro-textures. Con: GPU heat/time.
            config.LANGEVIN_SCORE_SCALE = min(0.6, config.LANGEVIN_SCORE_SCALE * 1.05)

        # 4. ANTI-ARTIFACT & STABILITY (The "Noise & Chaos" Block)
        if mu_std > config.MU_STD_NOISE_CEILING or drift_loss > STABILITY_LIMIT * 0.8:
            # NOISY/INSTABLE: Increase smoothing
            # Pro: Cleaner backgrounds. Con: Less high-freq detail.
            config.DRIFT_WEIGHT = max(config.MIN_DRIFT_WEIGHT_DYNAMIC, config.DRIFT_WEIGHT * 0.9)
            config.DEFAULT_LANGEVIN_STEPS = min(40, config.DEFAULT_LANGEVIN_STEPS + 2) # Reduced cap and step
        elif mu_std < config.MU_STD_STABLE_FLOOR and drift_loss < 1.0:
            # STABLE: Push for faster learning
            config.DRIFT_WEIGHT = min(config.MAX_DRIFT_WEIGHT_DYNAMIC, config.DRIFT_WEIGHT * 1.02)
            config.DEFAULT_LANGEVIN_STEPS = max(5, config.DEFAULT_LANGEVIN_STEPS - 2) # Lower floor

        # 5. TRAINING TYPE JITTER (DISABLED for stability)
        # Jitter logic removed to prevent random phase switches that destabilize convergence.
        pass

        # 6. PANIC BUTTON (Catastrophic divergence)
        if comp_score < config.PANIC_SCORE_THRESHOLD or drift_loss > config.PANIC_DRIFT_THRESHOLD:
            config.logger.warning("🚨 [App Control] EMERGENCY RESTORE: Training diverging.")
            config.CFG_SCALE = 4.0
            config.DRIFT_WEIGHT = 1.5
            config.RECON_WEIGHT = 6.0
            config.LANGEVIN_SCORE_SCALE = 0.3
            config.DEFAULT_LANGEVIN_STEPS = 60

        # Log changes if any significant parameter was adjusted
        has_changed = (
            abs(config.DRIFT_WEIGHT - prev['dw']) > 0.01 or
            abs(config.RECON_WEIGHT - prev['rw']) > 0.1 or
            abs(config.CFG_SCALE - prev['cfg']) > 0.1 or
            abs(config.DEFAULT_LANGEVIN_STEPS - prev['ls']) >= 2 or
            abs(config.LANGEVIN_SCORE_SCALE - prev['lc']) > 0.02
        )

        if has_changed:
             config.logger.info(f"📊 [App Control] Update: DW={config.DRIFT_WEIGHT:.2f}, RW={config.RECON_WEIGHT:.1f}, "
                                f"CFG={config.CFG_SCALE:.1f}, LSteps={config.DEFAULT_LANGEVIN_STEPS}, LScale={config.LANGEVIN_SCORE_SCALE:.2f}")


    def _run_autonomous_strategy(self, epoch: int, losses: Dict[str, Any]):
        """
        Autonomous training management:
        - KPI-based phase transitions (1 -> 2 -> 3).
        - Stochastic nudges/restarts to avoid local minima.
        - Recovery Mode: Forces Phase 1 to repair latents.
        """
        if not self.trainer:
            return

        # --- 0. RECOVERY MODE HANDLING ---
        if config.IN_RECOVERY_MODE:
            if config.RECOVERY_START_EPOCH is None:
                config.RECOVERY_START_EPOCH = epoch
                config.logger.info(f"🩹 [Recovery] Starting 20-epoch VAE repair at epoch {epoch}.")
            
            elapsed = epoch - config.RECOVERY_START_EPOCH
            if elapsed < config.RECOVERY_EPOCHS:
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 1
                config.logger.info(f"🩹 [Recovery] VAE repair progress: {elapsed+1}/{config.RECOVERY_EPOCHS} epochs.")
                return # Stay in Phase 1
            else:
                config.IN_RECOVERY_MODE = False
                config.TRAINING_SCHEDULE['force_phase'] = 2 # Transition to Drift
                config.logger.info("✅ [Recovery] VAE repair complete. Transitioning to Phase 2 (Drift).")

        current_phase = self.trainer.phase
        temp = losses.get('temperature', 0.5)
        
        # --- 1. PHASE TRANSITION LOGIC ---
        if current_phase == 1:
            snr = losses.get('snr', 0)
            ssim = losses.get('ssim_loss', 1.0)
            if snr > config.PHASE1_SNR_THRESHOLD and ssim < config.PHASE1_SSIM_MAX and epoch >= config.PHASE1_MIN_EPOCH:
                config.logger.info(f"✨ [Auto-Strategy] VAE Sharpness reached. Transitioning to Phase 2.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 2

        elif current_phase == 2:
            drift = losses.get('drift', 10.0)
            ssim = losses.get('ssim_loss', 1.0)
            if drift < config.PHASE2_DRIFT_MAX and ssim < config.PHASE2_SSIM_MAX and epoch >= config.PHASE2_MIN_EPOCH:
                config.logger.info(f"✨ [Auto-Strategy] Drift Stability reached. Transitioning to Phase 3.")
                config.TRAINING_SCHEDULE['mode'] = 'manual'
                config.TRAINING_SCHEDULE['force_phase'] = 3

        # --- 2. STOCHASTIC RESTART capability (Back-switching) ---
        # Use a consecutive-failures counter to avoid random phase regressions
        elif current_phase == 3:
            if losses.get('composite_score', 0) < config.STOCHASTIC_RESTART_SCORE_THRESHOLD:
                self._bad_score_count += 1
                if self._bad_score_count >= config.CONSECUTIVE_BAD_SCORES_MAX:
                    config.logger.info("↩️ [Auto-Strategy] Fine-tuning regression: Reverting to Phase 2 for trajectory fix.")
                    config.TRAINING_SCHEDULE['force_phase'] = 2
                    self._bad_score_count = 0
            else:
                self._bad_score_count = 0


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
                if total_loss >= config.NAN_LOSS_THRESHOLD:
                    consecutive_failures += 1
                    if consecutive_failures >= config.CONSECUTIVE_FAILURES_MAX:
                        config.logger.error("🛑 Training stopped: consecutive failed epochs (NaN/Inf).")
                        break
                else:
                    consecutive_failures = 0
                
                # Update context
                self.ctx.update_metric(epoch, losses)
                
                # --- STRATEGY ENGINE ---
                self._broadcast_swarm_status(epoch, losses)
                self._listen_for_swarm_commands()
                
                # Dynamic parameter scaling
                self._adjust_dynamic_parameters(epoch, losses)
                # Autonomous phase switching and stochastic nudges
                self._run_autonomous_strategy(epoch, losses)
                
                if on_epoch_done:
                    on_epoch_done(epoch, losses)
                
                # Periodically save/generate
                if (epoch+1) % config.CHECKPOINT_SAVE_INTERVAL == 0:
                    self.trainer.save_checkpoint()
                if (epoch+1) % config.GENERATE_SAMPLE_INTERVAL == 0:
                    self.trainer.generate_reconstructions()
                    self.trainer.generate_samples()
                    # Signal to UI that new samples are available
                    self.ctx.log_queue.put("UPDATE_GALLERY")
                    
        except Exception as e:
            config.logger.error(f"Engine failure: {e}")
        finally:
            self.ctx.is_training = False
            self.ctx.stop_signal = False
