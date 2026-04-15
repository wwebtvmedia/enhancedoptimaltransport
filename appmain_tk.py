#!/usr/bin/env python3
"""
appmain_tk.py – Beautiful Tkinter GUI for Schrödinger Bridge Training
Uses shared logic from appmain_display.py and follows MCP pattern.
"""

import os
import sys
import re
import json
import queue
import threading
import time
import importlib
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ===== Third‑party imports =====
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
    from tkinter import font as tkfont
except ImportError:
    print("Tkinter not available. Please install python3-tk.")
    sys.exit(1)

try:
    import pygal
    from pygal.style import DarkStyle
    PYGL_AVAILABLE = True
except ImportError:
    PYGL_AVAILABLE = False

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False

# ===== Local modules =====
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import training
import data_management as dm
import models
from app_context import AppContext
from app_processor import TrainingProcessor
from appmain_display import Colors, get_param_description, parse_training_log, format_metrics_text, get_latest_log_path

# ============================================================
# Custom logging handler
# ============================================================
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# ============================================================
# Modern Tooltip Class
# ============================================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.enter)
        widget.bind('<Leave>', self.leave)
        widget.bind('<ButtonPress>', self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#3c4043", foreground="#ffffff",
                         relief=tk.SOLID, borderwidth=0,
                         font=("Roboto", 9), padx=8, pady=4)
        label.pack()

    def leave(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# ============================================================
# Modern Card Frame - Google Style
# ============================================================
class CardFrame(ttk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="Card.TFrame")
        
        if title:
            header = ttk.Frame(self, style="Card.TFrame")
            header.pack(fill='x', padx=16, pady=(16, 8))
            
            title_label = ttk.Label(header, text=title, style="CardTitle.TLabel")
            title_label.pack(side='left')
            
        self.content = ttk.Frame(self, style="Card.TFrame")
        self.content.pack(fill='both', expand=True, padx=16, pady=(0, 16))

# ============================================================
# Main GUI Application
# ============================================================
class SchrödingerBridgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schrödinger Bridge Trainer v2.0")
        self.root.geometry("1450x950")
        self.root.configure(bg=Colors.BG_DARK)
        
        # Initialize App State
        self.ctx = AppContext()
        self.engine = TrainingProcessor(self.ctx)
        
        # Link context queue to GUI's log_queue
        # Add QueueHandler to config.logger so we see all logs in the UI
        qh = QueueHandler(self.ctx.log_queue)
        qh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        config.logger.addHandler(qh)
        
        self.setup_styling()
        
        self.metrics = {}
        self.chart_labels = {}
        self.training_images_cache = None
        self.training_labels_cache = None
        
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Start HW init
        self.device_var.set(self.engine.initialize_hardware())
        
        # Start periodic UI polling
        self.process_log_queue()
        
        # Load training data preview automatically
        self.root.after(1000, self.load_training_data_preview)

    def setup_styling(self):
        style = ttk.Style()
        style.theme_use('clam')
        main_font = ("Roboto", 10) if "Roboto" in tkfont.families() else ("Segoe UI", 10)
        header_font = ("Roboto", 18, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 18, "bold")
        tab_font = ("Roboto", 10, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 10, "bold")
        
        style.configure(".", background=Colors.BG_DARK, foreground=Colors.FG, fieldbackground=Colors.BG_LIGHT, font=main_font)
        style.configure("TNotebook", background=Colors.BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=Colors.BG_DARK, foreground=Colors.FG_SECONDARY, padding=[24, 12], font=tab_font, borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", Colors.BG_DARK)], foreground=[("selected", Colors.ACCENT)])
        style.configure("Accent.TButton", background=Colors.ACCENT, foreground="#ffffff", borderwidth=0, font=tab_font)
        style.configure("TButton", background=Colors.BG_MEDIUM, foreground=Colors.ACCENT, borderwidth=1, bordercolor=Colors.BORDER, font=tab_font)
        style.configure("Header.TLabel", font=header_font, foreground=Colors.ACCENT, background=Colors.BG_DARK)
        style.configure("CardTitle.TLabel", font=("Roboto", 12, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 12, "bold"), foreground=Colors.FG, background=Colors.BG_MEDIUM)
        style.configure("Card.TFrame", background=Colors.BG_MEDIUM, relief="flat", borderwidth=1)
        style.configure("TSeparator", background=Colors.BORDER)

    def create_header(self):
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill='x', padx=32, pady=(32, 16))
        ttk.Label(header_frame, text="Schrödinger Bridge Trainer", style="Header.TLabel").pack(side='left')
        self.device_var = tk.StringVar(value="Initializing...")
        ttk.Label(header_frame, textvariable=self.device_var, foreground=Colors.FG_SECONDARY, background=Colors.BG_DARK).pack(side='right')

    def create_main_content(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        self.create_config_tab()
        self.create_training_tab()
        self.create_training_data_tab()
        self.create_gallery_tab()
        self.create_visualization_tab()

    def create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=20, pady=10)
        self.status_var = tk.StringVar(value="🚀 Ready to train")
        ttk.Label(status_frame, textvariable=self.status_var, foreground=Colors.FG_SECONDARY).pack(side='left')

    def create_config_tab(self):
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="⚙️ Configuration")
        
        canvas = tk.Canvas(self.config_frame, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.config_vars = {}
        groups = [
            ("📁 Dataset Settings", ["DATASET_NAME", "DATASET_PATH", "IMG_SIZE", "GEN_SIZE"]),
            ("⚡ Hyperparameters", ["LR", "EPOCHS", "BATCH_SIZE", "CST_COEF_GAUSSIAN_PRIO"]),
            ("⚖️ Loss Weights", ["KL_WEIGHT", "RECON_WEIGHT", "DIVERSITY_WEIGHT", "PERCEPTUAL_WEIGHT", "SSIM_WEIGHT"]),
            ("📅 Training Schedule", ["PHASE1_EPOCHS", "PHASE2_EPOCHS"]),
            ("🧠 Neural Tokenizer", ["USE_NEURAL_TOKENIZER", "TEXT_EMBEDDING_DIM", "CONTRASTIVE_WEIGHT"]),
            ("🎨 Features", ["USE_SUBPIXEL_CONV", "USE_AMP", "USE_SNAPSHOTS"])
        ]
        
        for name, params in groups:
            card = CardFrame(scrollable_frame, title=name)
            card.pack(fill='x', padx=10, pady=5)
            for p in params:
                self.create_param_row(card.content, p)

    def create_param_row(self, parent, param):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill='x', pady=4)
        lbl = ttk.Label(frame, text=param, width=25, background=Colors.BG_MEDIUM)
        lbl.pack(side='left', padx=10)
        ToolTip(lbl, get_param_description(param))
        
        # Handle dict/list nested configs (simplified for UI)
        val = getattr(config, param, "")
        if isinstance(val, dict) and param == "TRAINING_SCHEDULE":
            val = val.get('mode', 'auto')
            
        var = tk.StringVar(value=str(val))
        ttk.Entry(frame, textvariable=var, width=20).pack(side='left')
        self.config_vars[param] = var

    def create_training_tab(self):
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🎮 Training")
        
        left = ttk.Frame(self.training_frame, width=350)
        left.pack(side='left', fill='y', padx=10)
        
        ctrl = CardFrame(left, title="Controls")
        ctrl.pack(fill='x')
        self.start_btn = tk.Button(ctrl.content, text="Start Training", command=self.start_training, bg=Colors.ACCENT, fg="white", relief="flat", padx=20, pady=10)
        self.start_btn.pack(fill='x', pady=5)
        
        self.stop_btn = tk.Button(ctrl.content, text="Stop", command=self.engine.stop_training, bg=Colors.BG_LIGHT, fg=Colors.ERROR, relief="flat", state='disabled')
        self.stop_btn.pack(fill='x', pady=5)

        self.log_text = scrolledtext.ScrolledText(self.training_frame, wrap='word', bg="#ffffff", fg=Colors.FG, font=("Consolas", 10))
        self.log_text.pack(side='right', fill='both', expand=True, padx=10, pady=10)

    # --- RESTORED: Training Data Tab ---
    def create_training_data_tab(self):
        self.training_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_data_frame, text="📚 Training Data")
        
        canvas = tk.Canvas(self.training_data_frame, bg=Colors.BG_DARK, highlightthickness=0)
        scrollable = ttk.Frame(canvas)
        canvas.create_window((0,0), window=scrollable, anchor='nw')
        canvas.pack(fill='both', expand=True)
        
        ctrl = CardFrame(scrollable, title="Data Inspector")
        ctrl.pack(fill='x', padx=10, pady=5)
        
        self.data_info_var = tk.StringVar(value="Data logic restored.")
        ttk.Label(ctrl.content, textvariable=self.data_info_var).pack(side='left')
        
        tk.Button(ctrl.content, text="🔄 Refresh Preview", command=self.load_training_data_preview, bg=Colors.ACCENT, fg="white", relief='flat', padx=10).pack(side='right')
        
        self.training_display_frame = ttk.Frame(scrollable)
        self.training_display_frame.pack(fill='both', expand=True, padx=10, pady=10)

    def load_training_data_preview(self):
        try:
            import torchvision.utils as vutils
            loader = dm.load_data()
            batch = next(iter(loader))
            imgs = batch['image'] if isinstance(batch, dict) else batch[0]
            self.training_images_cache = imgs
            
            grid = vutils.make_grid(imgs[:32], nrow=8, normalize=True, value_range=(-1, 1))
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            pil_img = Image.fromarray(ndarr)
            self.training_photo = ImageTk.PhotoImage(pil_img)
            
            for w in self.training_display_frame.winfo_children(): w.destroy()
            tk.Label(self.training_display_frame, image=self.training_photo, bg=Colors.BG_DARK).pack()
            self.data_info_var.set(f"Loaded {len(imgs)} sample images from dataset")
        except Exception as e:
            self.data_info_var.set(f"Error: {e}")

    # --- RESTORED: Visualization Tab ---
    def create_visualization_tab(self):
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📊 Visualization")
        
        bar = ttk.Frame(self.viz_frame)
        bar.pack(fill='x', pady=5)
        
        tk.Button(bar, text="🔄 Load Latest Log", command=self.load_latest_log, bg=Colors.ACCENT, fg="white", relief='flat', padx=15).pack(side='left', padx=10)
        tk.Button(bar, text="📂 Select Log File", command=self.select_log_file, bg=Colors.BG_MEDIUM, fg=Colors.ACCENT, relief='flat', padx=15).pack(side='left', padx=10)
        self.viz_status = tk.StringVar(value="Ready to plot.")
        ttk.Label(bar, textvariable=self.viz_status).pack(side='left')

        self.chart_notebook = ttk.Notebook(self.viz_frame)
        self.chart_notebook.pack(fill='both', expand=True)
        
        for title in ["📉 Losses", "📊 SNR & Quality"]:
            frame = ttk.Frame(self.chart_notebook)
            self.chart_notebook.add(frame, text=title)
            canvas = tk.Canvas(frame, bg=Colors.BG_MEDIUM)
            canvas.pack(fill='both', expand=True)
            self.chart_labels[title] = canvas

    def load_latest_log(self):
        path = get_latest_log_path(config.DIRS["logs"])
        if path:
            self._load_log_from_path(path)

    def select_log_file(self):
        initial_dir = config.DIRS["logs"]
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Training Log File",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            path = Path(file_path)
            self._load_log_from_path(path)

    def _load_log_from_path(self, path: Path):
        self.metrics = parse_training_log(str(path))
        if self.metrics:
            self.viz_status.set(f"Loaded {len(self.metrics)} epochs from {path.name}")
            self.update_charts()
        else:
            self.viz_status.set(f"❌ Failed to parse or empty log: {path.name}")

    def update_charts(self):
        if not PYGL_AVAILABLE or not self.metrics: return
        epochs = sorted(self.metrics.keys())
        
        # Simple loss chart for Losses tab
        chart = pygal.Line(style=DarkStyle, width=800, height=400)
        chart.title = "Training Losses"
        chart.x_labels = [str(e+1) for e in epochs]
        
        for k in ['total', 'recon', 'kl']:
            vals = [self.metrics[e].get(k, 0) for e in epochs]
            chart.add(k.capitalize(), vals)
            
        if CAIRO_AVAILABLE:
            import io
            png = cairosvg.svg2png(bytestring=chart.render())
            img = Image.open(io.BytesIO(png)).resize((800, 400))
            self.chart_photo = ImageTk.PhotoImage(img)
            canvas = self.chart_labels["📉 Losses"]
            canvas.delete("all")
            canvas.create_image(10, 10, anchor='nw', image=self.chart_photo)

    def create_gallery_tab(self):
        self.gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gallery_frame, text="🖼️ Gallery")
        
        bar = ttk.Frame(self.gallery_frame)
        bar.pack(fill='x', pady=5)
        tk.Button(bar, text="🔄 Refresh Samples", command=self.refresh_gallery, bg=Colors.ACCENT, fg="white", relief='flat', padx=15).pack(side='left', padx=10)
        
        self.gallery_container = ttk.Frame(self.gallery_frame)
        self.gallery_container.pack(fill='both', expand=True, padx=10, pady=10)
        self.refresh_gallery()

    def refresh_gallery(self):
        for w in self.gallery_container.winfo_children(): w.destroy()
        
        samples_dir = config.DIRS["samples"]
        png_files = sorted(list(samples_dir.glob("gen_*.png")), key=os.path.getmtime, reverse=True)
        
        if png_files:
            try:
                # Load latest sample
                img = Image.open(str(png_files[0]))
                # Resize if too large for display
                img.thumbnail((800, 800))
                self.gallery_photo = ImageTk.PhotoImage(img)
                tk.Label(self.gallery_container, image=self.gallery_photo, text=f"Latest: {png_files[0].name}", compound='top', bg=Colors.BG_DARK).pack()
            except Exception as e:
                tk.Label(self.gallery_container, text=f"Error loading image: {e}").pack()
        else:
            tk.Label(self.gallery_container, text="No generated samples found.").pack()

    def start_training(self):
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.engine.start_training(on_epoch_done=self._on_epoch)

    def _on_epoch(self, epoch, losses):
        self.ctx.log_queue.put(format_metrics_text(epoch, losses))

    def process_log_queue(self):
        try:
            while True:
                msg = self.ctx.log_queue.get_nowait()
                self.log_text.insert(tk.END, str(msg) + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(500, self.process_log_queue)

def main():
    root = tk.Tk()
    app = SchrödingerBridgeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
