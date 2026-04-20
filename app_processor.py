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
    """The Engine that performs all the heavy computations with Swarm Protocol support."""
    
    def __init__(self, context):
        self.ctx = context
        self.trainer: Optional[training.EnhancedLabelTrainer] = None
        self._thread: Optional[threading.Thread] = None
        # Unique ID for Swarm Protocol
        self.node_id = f"node_{os.getpid()}_{random.randint(1000, 9999)}"
        self.swarm_dir = Path("enhanced_label_sb/swarm")
        self.swarm_dir.mkdir(parents=True, exist_ok=True)

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
                
                # Only apply if recent (last 30 seconds)
                if time.time() - cmd.get('timestamp', 0) < 30:
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
        
        # Stability Baseline
        STABILITY_LIMIT = 5.0
        
        # Store old values for logging
        prev = {
            'dw': config.DRIFT_WEIGHT, 'rw': config.RECON_WEIGHT,
            'cfg': config.CFG_SCALE, 'sw': config.SSIM_WEIGHT,
            'ls': config.DEFAULT_LANGEVIN_STEPS, 'lc': config.LANGEVIN_SCORE_SCALE
        }

        # 2. SSIM-BASED STRUCTURAL CONTROL (Realistic Thresholds for Phase 3)
        # SSIM around 0.6 is common in Phase 3. 
        if ssim_loss > 0.65:
            # BLURRY: Increase intensity
            # Pro: Sharpens boundaries. Con: Risk of artifacts.
            config.RECON_WEIGHT = min(20.0, config.RECON_WEIGHT * 1.05)
            config.CFG_SCALE = min(12.0, config.CFG_SCALE + 0.2)
        elif ssim_loss < 0.55:
            # SHARP: Can relax to favor diversity
            # Pro: More creative generations. Con: Risk of getting "soft".
            config.RECON_WEIGHT = max(4.0, config.RECON_WEIGHT * 0.95)
            config.CFG_SCALE = max(2.5, config.CFG_SCALE - 0.1)

        # 3. CONTRAST & DETAIL (Based on Score Trend)
        # If score is dropping, we might be "over-cooking"
        if comp_score < -45:
            # QUALITY DROP: Pull back slightly
            # Pro: Prevents burn-in artifacts. Con: Temporary lower contrast.
            config.CFG_SCALE = max(3.0, config.CFG_SCALE * 0.9)
            config.RECON_WEIGHT = max(5.0, config.RECON_WEIGHT * 0.9)
            config.LANGEVIN_SCORE_SCALE = max(0.1, config.LANGEVIN_SCORE_SCALE * 0.9)
        elif comp_score > -25:
            # HIGH QUALITY: Can push for extra detail
            # Pro: Professional-level micro-textures. Con: GPU heat/time.
            config.LANGEVIN_SCORE_SCALE = min(0.6, config.LANGEVIN_SCORE_SCALE * 1.05)

        # 4. ANTI-ARTIFACT & STABILITY (The "Noise & Chaos" Block)
        if mu_std > 1.1 or drift_loss > STABILITY_LIMIT * 0.8:
            # NOISY/INSTABLE: Increase smoothing
            # Pro: Cleaner backgrounds. Con: Less high-freq detail.
            config.DRIFT_WEIGHT = max(0.8, config.DRIFT_WEIGHT * 0.9)
            config.DEFAULT_LANGEVIN_STEPS = min(120, config.DEFAULT_LANGEVIN_STEPS + 5)
        elif mu_std < 0.7 and drift_loss < 1.0:
            # STABLE: Push for faster learning
            config.DRIFT_WEIGHT = min(3.0, config.DRIFT_WEIGHT * 1.02)
            config.DEFAULT_LANGEVIN_STEPS = max(30, config.DEFAULT_LANGEVIN_STEPS - 2)

        # 5. TRAINING TYPE JITTER (VAE vs DRIFT vs BOTH)
        # Pro: Prevents network "laziness". Con: Slight epoch jitter.
        if random.random() < 0.04: # 4% chance per epoch
            current_mode = config.TRAINING_SCHEDULE.get('force_phase', 3)
            r = random.random()
            if r < 0.15:
                new_mode = 1 # VAE Focus (Repair Fade)
                config.logger.info("🎲 [App Control] Jitter: VAE focus (Repair Fade).")
            elif r < 0.30:
                new_mode = 2 # Drift Focus (Repair Structure)
                config.logger.info("🎲 [App Control] Jitter: Drift focus (Repair Structure).")
            else:
                new_mode = 3 # Joint Fine-tuning
                config.logger.info("🎲 [App Control] Jitter: Joint fine-tuning.")
            config.TRAINING_SCHEDULE['force_phase'] = new_mode

        # 6. PANIC BUTTON (Catastrophic divergence)
        if comp_score < -120 or drift_loss > 15.0:
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
                self._broadcast_swarm_status(epoch, losses)
                self._listen_for_swarm_commands()
                
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
