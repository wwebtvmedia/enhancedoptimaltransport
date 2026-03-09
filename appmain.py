#!/usr/bin/env python3
"""
appmain.py – Graphical interface for Schrödinger Bridge training and visualization.

Replaces the original console main.py. Provides:
- Interactive configuration of all training hyperparameters.
- Real‑time log display and convergence plots during training.
- Loading of existing log files for post‑training analysis.
- Multiple pygal charts (loss, SNR, latent stats, etc.) in a tabbed window.
"""

import os
import sys
import re
import json
import queue
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ===== Third‑party imports =====
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
except ImportError:
    print("Tkinter not available. Please install python3-tk.")
    sys.exit(1)

try:
    import pygal
    from pygal.style import DarkStyle
    PYGL_AVAILABLE = True
except ImportError:
    PYGL_AVAILABLE = False
    print("Pygal not installed. Charts will be disabled. Install with: pip install pygal")

try:
    import cairosvg
    from PIL import Image, ImageTk
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("Cairosvg or PIL not installed. PNG embedding disabled; will open SVG in browser.")

# ===== Local modules (the existing codebase) =====
# Ensure the current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import training
import data_management as dm
import models
import inference

# ============================================================
# Custom logging handler that sends messages to a queue
# ============================================================
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# ============================================================
# Main GUI Application
# ============================================================
class SchrödingerBridgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schrödinger Bridge Trainer – Label‑Conditioned")
        self.root.geometry("1200x800")

        # Control flags
        self.training_running = False
        self.stop_training_flag = False
        self.log_queue = queue.Queue()
        self.metrics = {}          # Stores all parsed metrics: epoch -> dict
        self.current_epoch = 0

        # Build UI
        self.create_menu()
        self.create_notebook()
        self.create_status_bar()

        # Start periodic queue processing
        self.process_log_queue()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def create_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load log file...", command=self.load_log_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.root.config(menu=menubar)

    # ------------------------------------------------------------------
    # Notebook (tabs)
    # ------------------------------------------------------------------
    def create_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Configuration
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.build_config_tab()

        # Tab 2: Training control & log
        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text="Training")
        self.build_control_tab()

        # Tab 3: Visualization (multi‑chart)
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        self.build_viz_tab()

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ------------------------------------------------------------------
    # Configuration tab
    # ------------------------------------------------------------------
    def build_config_tab(self):
        # Use a canvas with scrollbar for many parameters
        canvas = tk.Canvas(self.config_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Organise parameters in groups
        row = 0
        groups = [
            ("Paths", ["BASE_DIR"]),
            ("Model dimensions", ["IMG_SIZE", "LATENT_CHANNELS", "LATENT_H", "LATENT_W", "LATENT_DIM"]),
            ("Label conditioning", ["NUM_CLASSES", "LABEL_EMB_DIM"]),
            ("Training hyperparameters", ["LR", "EPOCHS", "WEIGHT_DECAY", "GRAD_CLIP", "BATCH_SIZE"]),
            ("Loss weights", ["KL_WEIGHT", "RECON_WEIGHT", "DRIFT_WEIGHT", "DIVERSITY_WEIGHT", "CONSISTENCY_WEIGHT"]),
            ("VAE specific", ["LATENT_SCALE", "FREE_BITS", "DIVERSITY_TARGET_STD", "DIVERSITY_BALANCE_WEIGHT",
                              "KL_ANNEALING_EPOCHS", "LOGVAR_CLAMP_MIN", "LOGVAR_CLAMP_MAX", "MU_NOISE_SCALE"]),
            ("Channel dropout", ["CHANNEL_DROPOUT_PROB", "CHANNEL_DROPOUT_SURVIVAL"]),
            ("Drift network", ["DRIFT_LR_MULTIPLIER", "DRIFT_GRAD_CLIP_FACTOR", "PHASE2_VAE_LR_FACTOR",
                               "PHASE3_VAE_LR_FACTOR", "TEMPERATURE_START", "TEMPERATURE_END",
                               "DRIFT_TARGET_NOISE_SCALE", "TIME_WEIGHT_FACTOR"]),
            ("Inference", ["DEFAULT_STEPS", "DEFAULT_SEED", "INFERENCE_TEMPERATURE",
                           "LANGEVIN_STEP_SIZE", "LANGEVIN_SCORE_SCALE"]),
            ("Fourier features", ["USE_FOURIER_FEATURES", "FOURIER_FREQS"]),
            ("Enhanced features", ["USE_PERCENTILE", "USE_SNAPSHOTS", "USE_KPI_TRACKING",
                                   "TARGET_SNR", "SNAPSHOT_INTERVAL", "SNAPSHOT_KEEP",
                                   "KPI_WINDOW_SIZE", "EARLY_STOP_PATIENCE"]),
            ("OU Bridge", ["USE_OU_BRIDGE", "OU_THETA", "OU_SIGMA", "USE_AMP"]),
            ("Training schedule", ["PHASE1_EPOCHS", "PHASE2_EPOCHS"]),
        ]

        self.config_vars = {}   # stores tk.StringVar for each parameter

        for group_name, param_list in groups:
            # Group label
            ttk.Label(scrollable_frame, text=group_name, font=('Arial', 10, 'bold')).grid(
                row=row, column=0, columnspan=2, sticky='w', pady=(10,0))
            row += 1

            for param in param_list:
                # Get current value from config module
                default = getattr(config, param, "")
                # For boolean, use Checkbutton
                if isinstance(default, bool):
                    var = tk.BooleanVar(value=default)
                    cb = ttk.Checkbutton(scrollable_frame, text=param, variable=var)
                    cb.grid(row=row, column=0, sticky='w', padx=20)
                    self.config_vars[param] = var
                else:
                    # For other types, use entry
                    ttk.Label(scrollable_frame, text=param).grid(row=row, column=0, sticky='w', padx=20)
                    var = tk.StringVar(value=str(default))
                    entry = ttk.Entry(scrollable_frame, textvariable=var, width=20)
                    entry.grid(row=row, column=1, sticky='w', padx=5)
                    self.config_vars[param] = var
                row += 1

        # Buttons to apply/save config
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Apply to config", command=self.apply_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save config to file", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load config from file", command=self.load_config).pack(side=tk.LEFT, padx=5)

    def apply_config(self):
        """Update the global config module with values from the GUI."""
        for param, var in self.config_vars.items():
            val = var.get()
            # Convert to appropriate type
            default = getattr(config, param)
            if isinstance(default, bool):
                setattr(config, param, var.get())
            elif isinstance(default, int):
                try:
                    setattr(config, param, int(val))
                except:
                    pass
            elif isinstance(default, float):
                try:
                    setattr(config, param, float(val))
                except:
                    pass
            elif isinstance(default, str):
                setattr(config, param, val)
            elif isinstance(default, list):
                # For lists like FOURIER_FREQS, expect comma-separated
                try:
                    items = [int(x.strip()) for x in val.split(',') if x.strip()]
                    setattr(config, param, items)
                except:
                    pass
        self.status_var.set("Configuration applied.")
        # Also update derived constants that depend on these
        config.LATENT_H = config.IMG_SIZE // 8
        config.LATENT_W = config.IMG_SIZE // 8
        config.LATENT_DIM = config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W

    def save_config(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filename:
            return
        self.apply_config()  # ensure latest values
        # Gather all config variables
        cfg_dict = {}
        for param in dir(config):
            if param.isupper() and not param.startswith('_'):
                try:
                    # make it JSON serializable
                    val = getattr(config, param)
                    if isinstance(val, Path):
                        val = str(val)
                    cfg_dict[param] = val
                except:
                    pass
        with open(filename, 'w') as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        self.status_var.set(f"Config saved to {filename}")

    def load_config(self):
        filename = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not filename:
            return
        with open(filename, 'r') as f:
            cfg_dict = json.load(f)
        for param, val in cfg_dict.items():
            if param in self.config_vars:
                self.config_vars[param].set(str(val))
            # Also update config module immediately
            setattr(config, param, val)
        self.status_var.set(f"Config loaded from {filename}")

    # ------------------------------------------------------------------
    # Training control tab
    # ------------------------------------------------------------------
    def build_control_tab(self):
        # Top frame with buttons
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(fill='x', padx=5, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Generate Samples (from latest)", command=self.generate_samples).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export to ONNX", command=self.export_onnx).pack(side=tk.LEFT, padx=5)

        # Progress bar and epoch info
        prog_frame = ttk.Frame(self.control_frame)
        prog_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(prog_frame, text="Epoch:").pack(side=tk.LEFT)
        self.epoch_var = tk.StringVar(value="0/???")
        ttk.Label(prog_frame, textvariable=self.epoch_var).pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(prog_frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=5, expand=True, fill='x')

        # Log output
        ttk.Label(self.control_frame, text="Training Log:").pack(anchor='w', padx=5)
        self.log_text = scrolledtext.ScrolledText(self.control_frame, height=15, wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

    def start_training(self):
        if self.training_running:
            return
        self.training_running = True
        self.stop_training_flag = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Training started...")

        # Apply current config to global module
        self.apply_config()

        # Clear previous metrics and log
        self.metrics.clear()
        self.current_epoch = 0
        self.log_text.delete(1.0, tk.END)

        # Start training thread
        self.train_thread = threading.Thread(target=self.run_training, daemon=True)
        self.train_thread.start()

        # Start periodic progress update
        self.update_progress()

    def stop_training(self):
        if self.training_running:
            self.stop_training_flag = True
            self.status_var.set("Stopping after current epoch...")

    def run_training(self):
        """This runs in a separate thread."""
        # Redirect logging to queue
        import logging
        qh = QueueHandler(self.log_queue)
        qh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        config.logger.addHandler(qh)

        # Override EPOCHS if needed? We'll just run train_model with whatever is in config
        # But train_model runs for config.EPOCHS; we might want to allow early stop.
        # We'll create a custom loop that respects stop flag.
        try:
            loader = dm.load_data()
            trainer = training.EnhancedLabelTrainer(loader)

            # Optionally load latest checkpoint if exists
            latest = config.DIRS["ckpt"] / "latest.pt"
            if latest.exists():
                trainer.load_checkpoint()

            total_epochs = config.EPOCHS
            for epoch in range(trainer.epoch, total_epochs):
                if self.stop_training_flag:
                    config.logger.info("Training stopped by user.")
                    break
                trainer.epoch = epoch
                epoch_losses = trainer.train_epoch()
                self.log_queue.put(f"EPOCH_DONE:{epoch}:{epoch_losses}")

                # Update progress via queue
                self.log_queue.put(f"PROGRESS:{epoch+1}/{total_epochs}")

                # Save checkpoint periodically
                if (epoch+1) % 5 == 0:
                    trainer.save_checkpoint()

                # Generate samples every 10 epochs
                if (epoch+1) % 10 == 0:
                    trainer.generate_samples()

            config.logger.info("Training finished.")
        except Exception as e:
            config.logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Remove queue handler
            config.logger.removeHandler(qh)
            self.log_queue.put("TRAINING_DONE")

    def update_progress(self):
        """Periodically called to update UI based on queue."""
        # This method is called via after()
        self.process_log_queue()
        if self.training_running:
            self.root.after(500, self.update_progress)

    def process_log_queue(self):
        """Process all messages currently in the log queue."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "TRAINING_DONE":
                    self.training_running = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.status_var.set("Training completed.")
                    self.update_charts()
                elif msg.startswith("PROGRESS:"):
                    prog = msg[9:]
                    self.epoch_var.set(prog)
                    try:
                        cur, tot = map(int, prog.split('/'))
                        self.progress['maximum'] = tot
                        self.progress['value'] = cur
                    except:
                        pass
                elif msg.startswith("EPOCH_DONE:"):
                    parts = msg.split(':', 2)
                    epoch = int(parts[1])
                    loss_dict = eval(parts[2])   # safe because controlled output
                    self.metrics[epoch] = loss_dict
                    self.current_epoch = epoch
                    self.update_charts()
                else:
                    # Normal log line
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Visualization tab
    # ------------------------------------------------------------------
    def build_viz_tab(self):
        # Create a notebook inside for multiple charts
        self.chart_notebook = ttk.Notebook(self.viz_frame)
        self.chart_notebook.pack(fill='both', expand=True)

        # We'll create placeholders for charts; they will be filled later
        self.chart_frames = {}
        self.chart_labels = {}

        chart_types = [
            ("Loss", "total_loss", "recon", "kl", "diversity"),
            ("Drift & Consistency", "drift", "consistency"),
            ("SNR & Latent Std", "snr", "latent_std"),
            ("Channel Std", "min_channel_std", "max_channel_std"),
        ]
        for title, *keys in chart_types:
            frame = ttk.Frame(self.chart_notebook)
            self.chart_notebook.add(frame, text=title)
            self.chart_frames[title] = frame
            # Will later place an image label here
            lbl = ttk.Label(frame)
            lbl.pack(fill='both', expand=True)
            self.chart_labels[title] = lbl

    def update_charts(self):
        """Regenerate all charts from current metrics and display."""
        if not PYGL_AVAILABLE:
            return
        if not self.metrics:
            return

        # Prepare data series
        epochs = sorted(self.metrics.keys())
        series = {}
        for ep in epochs:
            d = self.metrics[ep]
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    series.setdefault(k, []).append((ep, v))

        # For each chart tab, create a line chart
        for title, lbl in self.chart_labels.items():
            if title == "Loss":
                keys = ['total', 'recon', 'kl', 'diversity']
            elif title == "Drift & Consistency":
                keys = ['drift', 'consistency']
            elif title == "SNR & Latent Std":
                keys = ['snr', 'latent_std']
            elif title == "Channel Std":
                keys = ['min_channel_std', 'max_channel_std']
            else:
                continue

            # Create pygal chart
            line_chart = pygal.Line(style=DarkStyle, x_label_rotation=20, show_minor_x_labels=False)
            line_chart.title = title
            line_chart.x_labels = [str(ep) for ep in epochs]

            for key in keys:
                if key in series:
                    # values aligned with epochs
                    vals = []
                    for ep in epochs:
                        found = next((v for e, v in series[key] if e == ep), None)
                        vals.append(found if found is not None else 0)
                    line_chart.add(key, vals)

            # Render to SVG and then to PNG if cairo available
            svg = line_chart.render()
            if CAIRO_AVAILABLE:
                png_bytes = cairosvg.svg2png(bytestring=svg)
                from PIL import Image, ImageTk
                import io
                img = Image.open(io.BytesIO(png_bytes))
                imgtk = ImageTk.PhotoImage(img)
                lbl.config(image=imgtk)
                lbl.image = imgtk   # keep reference
            else:
                # Fallback: save to temp file and open in browser
                import tempfile, webbrowser
                tmp = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
                tmp.write(svg)
                tmp.close()
                webbrowser.open('file://' + tmp.name)

    # ------------------------------------------------------------------
    # Load log file for post‑training visualization
    # ------------------------------------------------------------------
    def load_log_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Log files", "*.log"), ("All files", "*.*")])
        if not filename:
            return
        self.status_var.set(f"Loading log {filename}...")
        self.metrics.clear()
        # Parse log file
        epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) complete:')
        loss_pattern = re.compile(r'  (\w+ loss): ([\d\.eE+-]+)')
        snr_pattern = re.compile(r'  SNR: ([\d\.]+)dB')
        latent_std_pattern = re.compile(r'  Latent std: ([\d\.]+)')

        with open(filename, 'r') as f:
            lines = f.readlines()
        current_epoch = None
        for line in lines:
            m = epoch_pattern.search(line)
            if m:
                current_epoch = int(m.group(1))
                self.metrics[current_epoch] = {}
                continue
            if current_epoch is not None:
                m = loss_pattern.search(line)
                if m:
                    key = m.group(1).replace(' ', '_')
                    val = float(m.group(2))
                    self.metrics[current_epoch][key] = val
                m = snr_pattern.search(line)
                if m:
                    self.metrics[current_epoch]['snr'] = float(m.group(1))
                m = latent_std_pattern.search(line)
                if m:
                    self.metrics[current_epoch]['latent_std'] = float(m.group(1))
        self.status_var.set(f"Loaded {len(self.metrics)} epochs from {filename}")
        self.update_charts()
        # Switch to visualization tab
        self.notebook.select(self.viz_frame)

    # ------------------------------------------------------------------
    # Utility actions
    # ------------------------------------------------------------------
    def generate_samples(self):
        """Run inference using latest checkpoint."""
        try:
            # Dummy loader for trainer init
            from torch.utils.data import DataLoader, TensorDataset
            dummy = DataLoader(TensorDataset(torch.randn(1,3,64,64)), batch_size=1)
            trainer = training.EnhancedLabelTrainer(dummy)
            if trainer.load_for_inference():
                # Ask for parameters
                dialog = tk.Toplevel(self.root)
                dialog.title("Generate Samples")
                ttk.Label(dialog, text="Labels (comma-separated, e.g. 0,1,2):").grid(row=0, column=0, padx=5, pady=5)
                labels_var = tk.StringVar(value="0,1,2,3")
                ttk.Entry(dialog, textvariable=labels_var).grid(row=0, column=1, padx=5, pady=5)
                ttk.Label(dialog, text="Samples per label:").grid(row=1, column=0, padx=5, pady=5)
                samples_var = tk.StringVar(value="2")
                ttk.Entry(dialog, textvariable=samples_var).grid(row=1, column=1, padx=5, pady=5)
                ttk.Button(dialog, text="Generate", command=lambda: self.do_generate(trainer, labels_var.get(), samples_var.get(), dialog)).grid(row=2, column=0, columnspan=2, pady=10)
            else:
                messagebox.showerror("Error", "No checkpoint found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def do_generate(self, trainer, labels_str, samples_str, dialog):
        try:
            labels = [int(x.strip()) for x in labels_str.split(',')]
            samples_per_label = int(samples_str)
            all_labels = []
            for l in labels:
                all_labels.extend([l] * samples_per_label)
            trainer.generate_samples(labels=all_labels, num_samples=len(all_labels))
            messagebox.showinfo("Success", f"Generated {len(all_labels)} samples in {config.DIRS['samples']}")
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_onnx(self):
        try:
            from torch.utils.data import DataLoader, TensorDataset
            dummy = DataLoader(TensorDataset(torch.randn(1,3,64,64)), batch_size=1)
            trainer = training.EnhancedLabelTrainer(dummy)
            if trainer.load_for_inference():
                trainer.export_onnx()
                messagebox.showinfo("ONNX Export", f"Models exported to {config.DIRS['onnx']}")
            else:
                messagebox.showerror("Error", "No checkpoint found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_about(self):
        messagebox.showinfo("About", "Schrödinger Bridge Trainer GUI\nUsing Pygal for visualizations.")

# ============================================================
# Main entry point
# ============================================================
def main():
    # Ensure required directories exist
    for d in config.DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Set up logging to console only (GUI will also capture)
    import logging
    config.logger.handlers.clear()
    config.logger.addHandler(logging.StreamHandler())
    config.logger.setLevel(logging.INFO)

    root = tk.Tk()
    app = SchrödingerBridgeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()