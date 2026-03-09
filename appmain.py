#!/usr/bin/env python3
"""
appmain.py – Beautiful GUI for Schrödinger Bridge Training
A modern, dark-themed interface with real-time visualization
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

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
    from pygal.style import DarkStyle, LightStyle, DarkGreenStyle
    PYGL_AVAILABLE = True
except ImportError:
    PYGL_AVAILABLE = False
    print("Pygal not installed. Charts will be disabled. Install with: pip install pygal")

try:
    import cairosvg
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    print("Cairosvg or PIL not installed. PNG embedding disabled.")

# ===== Local modules =====
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import training
import data_management as dm
import models
import inference

# ============================================================
# Modern Dark Theme Colors
# ============================================================
class Colors:
    BG_DARK = "#1e1e2e"      # Main background
    BG_MEDIUM = "#2d2d3f"     # Secondary background
    BG_LIGHT = "#3d3d5a"      # Input fields
    FG = "#ffffff"            # Foreground text
    FG_SECONDARY = "#b4b4c0"  # Secondary text
    ACCENT = "#89b4fa"        # Blue accent
    ACCENT2 = "#cba6f7"       # Purple accent
    SUCCESS = "#a6e3a1"       # Green
    WARNING = "#f9e2af"       # Yellow
    ERROR = "#f38ba8"         # Red
    BORDER = "#45475a"        # Border color

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
                         background=Colors.BG_LIGHT, foreground=Colors.FG,
                         relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", 9))
        label.pack(padx=5, pady=5)

    def leave(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# ============================================================
# Modern Card Frame
# ============================================================
class CardFrame(ttk.Frame):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="Card.TFrame")
        
        if title:
            title_label = ttk.Label(self, text=title, style="CardTitle.TLabel")
            title_label.pack(anchor='w', padx=10, pady=(5, 0))
            
            separator = ttk.Separator(self, orient='horizontal')
            separator.pack(fill='x', padx=10, pady=5)
        
        self.content = ttk.Frame(self, style="Card.TFrame")
        self.content.pack(fill='both', expand=True, padx=10, pady=10)

# ============================================================
# Main GUI Application
# ============================================================
class SchrödingerBridgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Schrödinger Bridge Trainer")
        self.root.geometry("1400x900")
        
        # Set up modern styling
        self.setup_styling()
        
        # Control flags
        self.training_running = False
        self.stop_training_flag = False
        self.log_queue = queue.Queue()
        self.metrics = {}
        self.current_epoch = 0

        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Initialize device
        self.initialize_device()
        
        # Start queue processing
        self.process_log_queue()

    def setup_styling(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        
        # Configure colors
        style.theme_use('clam')
        
        # Configure colors for all elements
        style.configure(".", 
                       background=Colors.BG_DARK,
                       foreground=Colors.FG,
                       fieldbackground=Colors.BG_LIGHT,
                       troughcolor=Colors.BG_MEDIUM,
                       selectbackground=Colors.ACCENT,
                       selectforeground=Colors.BG_DARK,
                       font=("Segoe UI", 10))
        
        # Notebook styling
        style.configure("TNotebook", background=Colors.BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", 
                       background=Colors.BG_MEDIUM,
                       foreground=Colors.FG,
                       padding=[15, 5],
                       font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab",
                 background=[("selected", Colors.ACCENT)],
                 foreground=[("selected", Colors.BG_DARK)])
        
        # Button styling
        style.configure("Accent.TButton",
                       background=Colors.ACCENT,
                       foreground=Colors.BG_DARK,
                       borderwidth=0,
                       focuscolor="none",
                       font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
                 background=[("active", "#9bb9f0"), ("pressed", "#6d91d0")])
        
        style.configure("TButton",
                       background=Colors.BG_LIGHT,
                       foreground=Colors.FG,
                       borderwidth=1,
                       focuscolor="none",
                       font=("Segoe UI", 10))
        style.map("TButton",
                 background=[("active", Colors.BG_MEDIUM)],
                 relief=[("pressed", "sunken")])
        
        # Label styling
        style.configure("TLabel", 
                       background=Colors.BG_DARK,
                       foreground=Colors.FG,
                       font=("Segoe UI", 10))
        
        style.configure("Header.TLabel",
                       font=("Segoe UI", 16, "bold"),
                       foreground=Colors.ACCENT)
        
        style.configure("CardTitle.TLabel",
                       font=("Segoe UI", 12, "bold"),
                       foreground=Colors.ACCENT2)
        
        # Entry styling
        style.configure("TEntry",
                       fieldbackground=Colors.BG_LIGHT,
                       foreground=Colors.FG,
                       insertcolor=Colors.FG,
                       borderwidth=1,
                       relief="solid")
        
        style.map("TEntry",
                 fieldbackground=[("focus", Colors.BG_MEDIUM)],
                 bordercolor=[("focus", Colors.ACCENT)])
        
        # Combobox styling
        style.configure("TCombobox",
                       fieldbackground=Colors.BG_LIGHT,
                       foreground=Colors.FG,
                       arrowcolor=Colors.FG,
                       borderwidth=1)
        style.map("TCombobox",
                 fieldbackground=[("focus", Colors.BG_MEDIUM)],
                 bordercolor=[("focus", Colors.ACCENT)])
        
        # Checkbutton styling
        style.configure("TCheckbutton",
                       background=Colors.BG_DARK,
                       foreground=Colors.FG,
                       font=("Segoe UI", 10))
        style.map("TCheckbutton",
                 background=[("active", Colors.BG_DARK)],
                 foreground=[("active", Colors.ACCENT)])
        
        # Progress bar styling
        style.configure("Horizontal.TProgressbar",
                       background=Colors.ACCENT,
                       troughcolor=Colors.BG_MEDIUM,
                       borderwidth=0)
        
        # Scrollbar styling
        style.configure("Vertical.TScrollbar",
                       background=Colors.BG_LIGHT,
                       troughcolor=Colors.BG_DARK,
                       arrowcolor=Colors.FG,
                       borderwidth=0)
        
        # Separator
        style.configure("TSeparator",
                       background=Colors.BORDER)
        
        # Frame styling
        style.configure("Card.TFrame",
                       background=Colors.BG_MEDIUM,
                       relief="solid",
                       borderwidth=1)

    def create_header(self):
        """Create modern header with title and description"""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        # Title
        title_label = ttk.Label(header_frame, text="🧠 Schrödinger Bridge Trainer", 
                                style="Header.TLabel")
        title_label.pack(side='left')
        
        # Version and info
        info_frame = ttk.Frame(header_frame)
        info_frame.pack(side='right')
        
        self.device_var = tk.StringVar(value="Initializing...")
        device_label = ttk.Label(info_frame, textvariable=self.device_var, 
                                foreground=Colors.FG_SECONDARY)
        device_label.pack()
        
        # Separator
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', padx=20, pady=10)

    def create_main_content(self):
        """Create main notebook and content"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Create tabs
        self.create_config_tab()
        self.create_training_tab()
        self.create_visualization_tab()

    def create_status_bar(self):
        """Create modern status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom', padx=20, pady=10)
        
        self.status_var = tk.StringVar(value="🚀 Ready to train")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                foreground=Colors.FG_SECONDARY)
        status_label.pack(side='left')
        
        # Add small indicators
        indicators_frame = ttk.Frame(status_frame)
        indicators_frame.pack(side='right')
        
        self.indicator_gpu = tk.Canvas(indicators_frame, width=16, height=16,
                                      bg=Colors.BG_DARK, highlightthickness=0)
        self.indicator_gpu.pack(side='left', padx=5)
        self.draw_indicator(self.indicator_gpu, Colors.SUCCESS if torch.cuda.is_available() else Colors.BG_LIGHT)
        
        self.indicator_mem = tk.Canvas(indicators_frame, width=16, height=16,
                                      bg=Colors.BG_DARK, highlightthickness=0)
        self.indicator_mem.pack(side='left', padx=5)
        self.draw_indicator(self.indicator_mem, Colors.SUCCESS)

    def draw_indicator(self, canvas, color):
        """Draw a circular indicator"""
        canvas.create_oval(2, 2, 14, 14, fill=color, outline=Colors.BORDER, width=1)

    def initialize_device(self):
        """Initialize device and update config"""
        try:
            import torch
            
            # Device detection
            if torch.cuda.is_available():
                config.DEVICE = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
                self.device_var.set(f"🎮 CUDA: {device_name[:30]}...")
                self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                config.DEVICE = torch.device("mps")
                self.device_var.set("🍎 Apple Silicon (MPS)")
                self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
            else:
                config.DEVICE = torch.device("cpu")
                self.device_var.set("💻 CPU (No GPU)")
                self.draw_indicator(self.indicator_gpu, Colors.WARNING)
            
            # Configure device-specific settings
            if config.DEVICE.type == 'cpu':
                config.BATCH_SIZE = 32
                config.LR = 1e-4
            elif config.DEVICE.type == 'mps':
                config.BATCH_SIZE = 48
                config.LR = 1.5e-4
            else:  # CUDA
                config.BATCH_SIZE = 64
                config.LR = 2e-4
            
            config.DTYPE = torch.float32
            config.logger.info(f"Device initialized: {config.DEVICE}")
            
        except Exception as e:
            self.device_var.set("⚠️ Device init failed")
            self.draw_indicator(self.indicator_gpu, Colors.ERROR)
            config.logger.error(f"Device initialization failed: {e}")

    # ============================================================
    # Configuration Tab (Improved)
    # ============================================================
    def create_config_tab(self):
        """Create beautiful configuration tab with cards"""
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="⚙️ Configuration")

        # Create canvas with scrollbar
        canvas = tk.Canvas(self.config_frame, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar.pack(side="right", fill="y")

        # Organize parameters in cards
        param_groups = [
            ("📁 Paths", ["BASE_DIR"]),
            ("📐 Model Dimensions", ["IMG_SIZE", "LATENT_CHANNELS", "LATENT_H", "LATENT_W", "LATENT_DIM"]),
            ("🏷️ Label Conditioning", ["NUM_CLASSES", "LABEL_EMB_DIM"]),
            ("⚡ Training Hyperparameters", ["LR", "EPOCHS", "WEIGHT_DECAY", "GRAD_CLIP", "BATCH_SIZE"]),
            ("⚖️ Loss Weights", ["KL_WEIGHT", "RECON_WEIGHT", "DRIFT_WEIGHT", "DIVERSITY_WEIGHT", "CONSISTENCY_WEIGHT"]),
            ("🎨 VAE Specific", ["LATENT_SCALE", "FREE_BITS", "DIVERSITY_TARGET_STD", "DIVERSITY_BALANCE_WEIGHT",
                                "KL_ANNEALING_EPOCHS", "LOGVAR_CLAMP_MIN", "LOGVAR_CLAMP_MAX", "MU_NOISE_SCALE"]),
            ("📊 Channel Dropout", ["CHANNEL_DROPOUT_PROB", "CHANNEL_DROPOUT_SURVIVAL"]),
            ("🔄 Drift Network", ["DRIFT_LR_MULTIPLIER", "DRIFT_GRAD_CLIP_FACTOR", "PHASE2_VAE_LR_FACTOR",
                                 "PHASE3_VAE_LR_FACTOR", "TEMPERATURE_START", "TEMPERATURE_END",
                                 "DRIFT_TARGET_NOISE_SCALE", "TIME_WEIGHT_FACTOR"]),
            ("🔮 Inference", ["DEFAULT_STEPS", "DEFAULT_SEED", "INFERENCE_TEMPERATURE",
                             "LANGEVIN_STEP_SIZE", "LANGEVIN_SCORE_SCALE"]),
            ("🌈 Fourier Features", ["USE_FOURIER_FEATURES", "FOURIER_FREQS"]),
            ("✨ Enhanced Features", ["USE_PERCENTILE", "USE_SNAPSHOTS", "USE_KPI_TRACKING",
                                    "TARGET_SNR", "SNAPSHOT_INTERVAL", "SNAPSHOT_KEEP",
                                    "KPI_WINDOW_SIZE", "EARLY_STOP_PATIENCE"]),
            ("🎯 OU Bridge", ["USE_OU_BRIDGE", "OU_THETA", "OU_SIGMA", "USE_AMP"]),
            ("📅 Training Schedule", ["PHASE1_EPOCHS", "PHASE2_EPOCHS"]),
        ]

        self.config_vars = {}

        for group_name, param_list in param_groups:
            # Create card for each group
            card = CardFrame(scrollable_frame, title=group_name)
            card.pack(fill='x', padx=10, pady=5)

            # Create parameter grid
            for i, param in enumerate(param_list):
                self.create_param_row(card.content, param, i)

        # Action buttons
        actions_card = CardFrame(scrollable_frame, title="🎮 Actions")
        actions_card.pack(fill='x', padx=10, pady=5)

        btn_frame = ttk.Frame(actions_card.content)
        btn_frame.pack(fill='x', pady=5)

        buttons = [
            ("💾 Apply to Config", self.apply_config, Colors.ACCENT),
            ("📂 Save to JSON", self.save_config_json, Colors.BG_LIGHT),
            ("📂 Load from JSON", self.load_config_json, Colors.BG_LIGHT),
            ("🔄 Load from config.py", self.load_from_config_py, Colors.BG_LIGHT),
            ("✏️ Edit config.py", self.edit_config_file, Colors.ACCENT2),
        ]

        for text, command, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=command,
                          bg=color, fg=Colors.BG_DARK if color == Colors.ACCENT else Colors.FG,
                          font=("Segoe UI", 10, "bold" if color == Colors.ACCENT else "normal"),
                          relief="flat", padx=15, pady=8, cursor="hand2")
            btn.pack(side='left', padx=5)
            
            # Add hover effect
            btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=self.lighten_color(c)))
            btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))

    def lighten_color(self, color):
        """Lighten a color for hover effect"""
        if color == Colors.ACCENT:
            return "#9bb9f0"
        elif color == Colors.ACCENT2:
            return "#d5b9fc"
        else:
            return Colors.BG_MEDIUM

    def create_param_row(self, parent, param, row):
        """Create a parameter row with tooltip"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)

        # Get current value
        default = getattr(config, param, "")
        
        # Label with tooltip
        label = ttk.Label(frame, text=param, width=25, anchor='w')
        label.pack(side='left', padx=(0, 10))
        
        # Add tooltip with description if available
        tooltip_text = self.get_param_description(param)
        ToolTip(label, tooltip_text)

        # Input field based on type
        if isinstance(default, bool):
            var = tk.BooleanVar(value=default)
            cb = ttk.Checkbutton(frame, variable=var)
            cb.pack(side='left')
            self.config_vars[param] = var
        else:
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=20)
            entry.pack(side='left')
            self.config_vars[param] = var

    def get_param_description(self, param):
        """Get description for parameter tooltip"""
        descriptions = {
            "IMG_SIZE": "Input image size (pixels)",
            "LATENT_CHANNELS": "Number of channels in latent space",
            "NUM_CLASSES": "Number of classes for label conditioning",
            "LR": "Learning rate",
            "EPOCHS": "Number of training epochs",
            "BATCH_SIZE": "Batch size (auto-adjusted for device)",
            "KL_WEIGHT": "Weight for KL divergence loss",
            "RECON_WEIGHT": "Weight for reconstruction loss",
            "DRIFT_WEIGHT": "Weight for drift network loss",
            "USE_OU_BRIDGE": "Use Ornstein-Uhlenbeck bridge",
            "USE_FOURIER_FEATURES": "Add Fourier features to input",
        }
        return descriptions.get(param, f"Configure {param} parameter")

    # ============================================================
    # Training Tab (Improved)
    # ============================================================
    def create_training_tab(self):
        """Create beautiful training control tab"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🎮 Training")

        # Left panel - Controls
        left_panel = ttk.Frame(self.training_frame, width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)

        # Control card
        control_card = CardFrame(left_panel, title="🎛️ Controls")
        control_card.pack(fill='x', pady=(0, 10))

        # Buttons with icons
        btn_frame = ttk.Frame(control_card.content)
        btn_frame.pack(fill='x', pady=5)

        self.start_btn = tk.Button(btn_frame, text="▶️ Start Training", 
                                  command=self.start_training,
                                  bg=Colors.SUCCESS, fg=Colors.BG_DARK,
                                  font=("Segoe UI", 11, "bold"),
                                  relief="flat", padx=20, pady=10, cursor="hand2")
        self.start_btn.pack(fill='x', pady=2)
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#b4e0b0"))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg=Colors.SUCCESS))

        self.stop_btn = tk.Button(btn_frame, text="⏹️ Stop Training", 
                                 command=self.stop_training, state=tk.DISABLED,
                                 bg=Colors.ERROR, fg=Colors.BG_DARK,
                                 font=("Segoe UI", 11, "bold"),
                                 relief="flat", padx=20, pady=10, cursor="hand2")
        self.stop_btn.pack(fill='x', pady=2)

        # Progress card
        progress_card = CardFrame(left_panel, title="📊 Progress")
        progress_card.pack(fill='x', pady=(0, 10))

        # Epoch progress
        ttk.Label(progress_card.content, text="Epoch:").pack(anchor='w')
        self.epoch_var = tk.StringVar(value="0 / 0")
        epoch_label = ttk.Label(progress_card.content, textvariable=self.epoch_var,
                               font=("Segoe UI", 16, "bold"), foreground=Colors.ACCENT)
        epoch_label.pack(anchor='w', pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(progress_card.content, orient='horizontal',
                                       length=250, mode='determinate')
        self.progress.pack(fill='x', pady=5)

        # Metrics card
        metrics_card = CardFrame(left_panel, title="📈 Live Metrics")
        metrics_card.pack(fill='both', expand=True)

        self.metrics_text = scrolledtext.ScrolledText(metrics_card.content,
                                                     height=10,
                                                     bg=Colors.BG_DARK,
                                                     fg=Colors.FG,
                                                     font=("Consolas", 10),
                                                     relief="flat",
                                                     borderwidth=1)
        self.metrics_text.pack(fill='both', expand=True)

        # Right panel - Log
        right_panel = ttk.Frame(self.training_frame)
        right_panel.pack(side='right', fill='both', expand=True)

        log_card = CardFrame(right_panel, title="📝 Training Log")
        log_card.pack(fill='both', expand=True)

        self.log_text = scrolledtext.ScrolledText(log_card.content,
                                                 wrap='word',
                                                 bg=Colors.BG_DARK,
                                                 fg=Colors.FG,
                                                 font=("Consolas", 10),
                                                 relief="flat",
                                                 borderwidth=1)
        self.log_text.pack(fill='both', expand=True)

        # Configure log text tags for colors
        self.log_text.tag_configure("error", foreground=Colors.ERROR)
        self.log_text.tag_configure("warning", foreground=Colors.WARNING)
        self.log_text.tag_configure("success", foreground=Colors.SUCCESS)
        self.log_text.tag_configure("info", foreground=Colors.FG)

    # ============================================================
    # Visualization Tab (Improved)
    # ============================================================
    def create_visualization_tab(self):
        """Create beautiful visualization tab"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="📊 Visualization")

        # Control bar
        control_bar = ttk.Frame(self.viz_frame)
        control_bar.pack(fill='x', pady=(0, 10))

        ttk.Label(control_bar, text="📁 Load log file:").pack(side='left', padx=(0, 10))
        
        self.load_btn = tk.Button(control_bar, text="📂 Browse", 
                                 command=self.load_log_file,
                                 bg=Colors.BG_LIGHT, fg=Colors.FG,
                                 relief="flat", padx=15, pady=5, cursor="hand2")
        self.load_btn.pack(side='left')

        self.chart_notebook = ttk.Notebook(self.viz_frame)
        self.chart_notebook.pack(fill='both', expand=True)

        self.chart_labels = {}
        chart_types = [
            ("📉 Losses", ["total", "recon", "kl", "diversity"]),
            ("🔄 Drift & Consistency", ["drift", "consistency"]),
            ("📊 SNR & Latent Std", ["snr", "latent_std"]),
            ("📈 Channel Statistics", ["min_channel_std", "max_channel_std"]),
        ]

        for title, _ in chart_types:
            frame = ttk.Frame(self.chart_notebook)
            self.chart_notebook.add(frame, text=title)
            
            # Placeholder for chart
            canvas = tk.Canvas(frame, bg=Colors.BG_MEDIUM, highlightthickness=0)
            canvas.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Add text placeholder
            canvas.create_text(400, 300, 
                             text="📊 No data to display\nRun training or load a log file",
                             fill=Colors.FG_SECONDARY,
                             font=("Segoe UI", 14))
            
            self.chart_labels[title] = canvas

    # ============================================================
    # Core functionality methods
    # ============================================================
    def apply_config(self):
        """Update config with GUI values"""
        try:
            for param, var in self.config_vars.items():
                val = var.get()
                default = getattr(config, param)
                
                if isinstance(default, bool):
                    setattr(config, param, var.get())
                elif isinstance(default, int):
                    setattr(config, param, int(val))
                elif isinstance(default, float):
                    setattr(config, param, float(val))
                elif isinstance(default, str):
                    setattr(config, param, val)
                elif isinstance(default, list):
                    items = [int(x.strip()) for x in val.split(',') if x.strip()]
                    setattr(config, param, items)
            
            config.LATENT_H = config.IMG_SIZE // 8
            config.LATENT_W = config.IMG_SIZE // 8
            config.LATENT_DIM = config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W
            
            self.status_var.set("✅ Configuration applied")
            self.log_message("Configuration updated successfully", "success")
            
        except Exception as e:
            self.status_var.set(f"❌ Error applying config: {str(e)[:50]}")
            self.log_message(f"Config error: {e}", "error")

    def log_message(self, message, tag="info"):
        """Add colored message to log"""
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n", tag)
        self.log_text.see(tk.END)

    # ============================================================
    # Training methods (keep the existing training logic)
    # ============================================================
    def start_training(self):
        if self.training_running:
            return
        
        self.training_running = True
        self.stop_training_flag = False
        self.start_btn.config(state=tk.DISABLED, bg=Colors.BG_LIGHT)
        self.stop_btn.config(state=tk.NORMAL, bg=Colors.ERROR)
        self.status_var.set("🚀 Training in progress...")

        # Apply current config
        self.apply_config()

        # Clear previous data
        self.metrics.clear()
        self.current_epoch = 0
        self.log_text.delete(1.0, tk.END)
        self.metrics_text.delete(1.0, tk.END)

        # Start training thread
        self.train_thread = threading.Thread(target=self.run_training, daemon=True)
        self.train_thread.start()

        # Start progress updates
        self.update_progress()

    def stop_training(self):
        if self.training_running:
            self.stop_training_flag = True
            self.status_var.set("⏹️ Stopping after current epoch...")

    def run_training(self):
        """Training thread with device initialization"""
        # Add logging handler
        qh = QueueHandler(self.log_queue)
        qh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        config.logger.addHandler(qh)

        try:
            # Ensure device is initialized
            if config.DEVICE is None:
                self.initialize_device()

            # Load data and create trainer
            loader = dm.load_data()
            trainer = training.EnhancedLabelTrainer(loader)

            # Load latest checkpoint if exists
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
                self.log_queue.put(f"PROGRESS:{epoch+1}/{total_epochs}")

                if (epoch+1) % 5 == 0:
                    trainer.save_checkpoint()

                if (epoch+1) % 10 == 0:
                    trainer.generate_samples()

            config.logger.info("Training finished.")

        except Exception as e:
            config.logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            config.logger.removeHandler(qh)
            self.log_queue.put("TRAINING_DONE")

    def update_progress(self):
        """Update UI from queue"""
        self.process_log_queue()
        if self.training_running:
            self.root.after(500, self.update_progress)

    def process_log_queue(self):
        """Process messages from training thread"""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                
                if msg == "TRAINING_DONE":
                    self.training_running = False
                    self.start_btn.config(state=tk.NORMAL, bg=Colors.SUCCESS)
                    self.stop_btn.config(state=tk.DISABLED, bg=Colors.ERROR)
                    self.status_var.set("✅ Training completed")
                    
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
                    loss_dict = eval(parts[2])
                    self.metrics[epoch] = loss_dict
                    self.current_epoch = epoch
                    
                    # Update metrics display
                    self.update_metrics_display(loss_dict)
                    self.update_charts()
                    
                else:
                    # Color-code log messages
                    if "ERROR" in msg or "error" in msg:
                        self.log_text.insert(tk.END, msg + "\n", "error")
                    elif "WARNING" in msg:
                        self.log_text.insert(tk.END, msg + "\n", "warning")
                    else:
                        self.log_text.insert(tk.END, msg + "\n", "info")
                    self.log_text.see(tk.END)
                    
        except queue.Empty:
            pass

    def update_metrics_display(self, loss_dict):
        """Update live metrics display"""
        self.metrics_text.delete(1.0, tk.END)
        
        # Format metrics nicely
        metrics_str = f"📊 Epoch {self.current_epoch + 1}\n"
        metrics_str += "═" * 40 + "\n"
        
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                if key in ['snr']:
                    metrics_str += f"📈 {key:20s}: {value:8.2f}\n"
                elif key in ['total', 'recon', 'drift']:
                    metrics_str += f"📉 {key:20s}: {value:8.4f}\n"
                elif key in ['kl', 'diversity']:
                    metrics_str += f"📊 {key:20s}: {value:8.6f}\n"
                else:
                    metrics_str += f"   {key:20s}: {value:8.4f}\n"
        
        self.metrics_text.insert(1.0, metrics_str)

    def update_charts(self):
        """Update visualization charts"""
        if not PYGL_AVAILABLE or not self.metrics:
            return

        try:
            epochs = sorted(self.metrics.keys())
            
            # Prepare data series
            series = {}
            for ep in epochs:
                d = self.metrics[ep]
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        series.setdefault(k, []).append((ep, v))

            # Create charts for each tab
            chart_configs = [
                ("📉 Losses", ['total', 'recon', 'kl', 'diversity']),
                ("🔄 Drift & Consistency", ['drift', 'consistency']),
                ("📊 SNR & Latent Std", ['snr', 'latent_std']),
                ("📈 Channel Statistics", ['min_channel_std', 'max_channel_std']),
            ]

            for title, keys in chart_configs:
                if title not in self.chart_labels:
                    continue

                canvas = self.chart_labels[title]
                canvas.delete("all")

                if not any(k in series for k in keys):
                    canvas.create_text(400, 300,
                                     text="📊 No data available for this chart",
                                     fill=Colors.FG_SECONDARY,
                                     font=("Segoe UI", 14))
                    continue

                # Create pygal chart
                chart = pygal.Line(style=DarkStyle,
                                 show_legend=True,
                                 show_minor_x_labels=False,
                                 x_label_rotation=20,
                                 width=800,
                                 height=400)
                
                chart.title = title
                chart.x_labels = [str(ep+1) for ep in epochs]

                for key in keys:
                    if key in series:
                        vals = []
                        for ep in epochs:
                            found = next((v for e, v in series[key] if e == ep), 0)
                            vals.append(found)
                        chart.add(key.replace('_', ' ').title(), vals)

                # Render
                svg = chart.render()
                
                if CAIRO_AVAILABLE:
                    png_bytes = cairosvg.svg2png(bytestring=svg)
                    from PIL import Image, ImageTk
                    import io
                    img = Image.open(io.BytesIO(png_bytes))
                    img = img.resize((800, 400), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(img)
                    
                    canvas.create_image(10, 10, anchor='nw', image=imgtk)
                    canvas.image = imgtk
                else:
                    canvas.create_text(400, 300,
                                     text="📊 Chart generated (SVG)",
                                     fill=Colors.FG_SECONDARY,
                                     font=("Segoe UI", 14))

        except Exception as e:
            print(f"Chart update error: {e}")

    # ============================================================
    # File operations
    # ============================================================
    def load_log_file(self):
        """Load and parse a log file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Log files", "*.log"), ("All files", "*.*")],
            title="Select training log file"
        )
        
        if not filename:
            return

        self.status_var.set(f"📂 Loading {Path(filename).name}...")
        self.metrics.clear()

        # Parse log file
        epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) complete:')
        loss_pattern = re.compile(r'  (\w+ loss): ([\d\.eE+-]+)')
        snr_pattern = re.compile(r'  SNR: ([\d\.]+)dB')
        latent_std_pattern = re.compile(r'  Latent std: ([\d\.]+)')

        try:
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

            self.status_var.set(f"✅ Loaded {len(self.metrics)} epochs from {Path(filename).name}")
            self.update_charts()
            self.notebook.select(self.viz_frame)

        except Exception as e:
            self.status_var.set(f"❌ Error loading file: {str(e)[:50]}")
            messagebox.showerror("Error", f"Failed to load log file:\n{e}")

    def edit_config_file(self):
        """Open config.py in editor"""
        config_path = Path(__file__).parent / "config.py"
        
        if not config_path.exists():
            messagebox.showerror("Error", f"config.py not found at {config_path}")
            return

        # Create editor window
        editor = tk.Toplevel(self.root)
        editor.title("Edit config.py")
        editor.geometry("800x600")
        editor.configure(bg=Colors.BG_DARK)

        # Text area with line numbers (simplified)
        text_frame = ttk.Frame(editor)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_area = scrolledtext.ScrolledText(text_frame,
                                             wrap='none',
                                             font=("Consolas", 11),
                                             bg=Colors.BG_DARK,
                                             fg=Colors.FG,
                                             insertbackground=Colors.FG,
                                             relief="flat",
                                             borderwidth=1)
        text_area.pack(fill='both', expand=True)

        # Load file
        with open(config_path, 'r') as f:
            text_area.insert('1.0', f.read())

        # Button bar
        btn_frame = ttk.Frame(editor)
        btn_frame.pack(fill='x', padx=10, pady=10)

        def save_changes():
            try:
                with open(config_path, 'w') as f:
                    f.write(text_area.get('1.0', 'end-1c'))
                if messagebox.askyesno("Reload", "Config saved. Reload now?"):
                    self.load_from_config_py()
                editor.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

        save_btn = tk.Button(btn_frame, text="💾 Save", command=save_changes,
                           bg=Colors.SUCCESS, fg=Colors.BG_DARK,
                           font=("Segoe UI", 10, "bold"),
                           relief="flat", padx=20, pady=5, cursor="hand2")
        save_btn.pack(side='left', padx=5)

        cancel_btn = tk.Button(btn_frame, text="❌ Cancel", command=editor.destroy,
                              bg=Colors.BG_LIGHT, fg=Colors.FG,
                              font=("Segoe UI", 10),
                              relief="flat", padx=20, pady=5, cursor="hand2")
        cancel_btn.pack(side='left', padx=5)

    def load_from_config_py(self):
        """Reload config from file"""
        try:
            importlib.reload(config)
            for param, var in self.config_vars.items():
                if hasattr(config, param):
                    val = getattr(config, param)
                    if isinstance(val, bool):
                        var.set(val)
                    else:
                        var.set(str(val))
            self.status_var.set("✅ Config reloaded from config.py")
        except Exception as e:
            self.status_var.set(f"❌ Reload failed: {str(e)[:50]}")
            messagebox.showerror("Error", f"Failed to reload config:\n{e}")

    def save_config_json(self):
        """Save config to JSON file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save configuration as"
        )
        
        if not filename:
            return

        self.apply_config()
        
        cfg_dict = {}
        for param in dir(config):
            if param.isupper() and not param.startswith('_'):
                try:
                    val = getattr(config, param)
                    if isinstance(val, Path):
                        val = str(val)
                    cfg_dict[param] = val
                except:
                    pass

        try:
            with open(filename, 'w') as f:
                json.dump(cfg_dict, f, indent=2, default=str)
            self.status_var.set(f"✅ Config saved to {Path(filename).name}")
        except Exception as e:
            self.status_var.set(f"❌ Save failed: {str(e)[:50]}")
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def load_config_json(self):
        """Load config from JSON file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load configuration from JSON"
        )
        
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                cfg_dict = json.load(f)
            
            for param, val in cfg_dict.items():
                if param in self.config_vars:
                    self.config_vars[param].set(str(val))
                setattr(config, param, val)
            
            self.status_var.set(f"✅ Config loaded from {Path(filename).name}")
            
        except Exception as e:
            self.status_var.set(f"❌ Load failed: {str(e)[:50]}")
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    def generate_samples(self):
        """Generate samples from latest checkpoint"""
        try:
            from torch.utils.data import DataLoader, TensorDataset
            dummy = DataLoader(TensorDataset(torch.randn(1,3,64,64)), batch_size=1)
            trainer = training.EnhancedLabelTrainer(dummy)
            
            if trainer.load_for_inference():
                # Simple dialog for sample generation
                dialog = tk.Toplevel(self.root)
                dialog.title("Generate Samples")
                dialog.geometry("400x200")
                dialog.configure(bg=Colors.BG_DARK)
                
                ttk.Label(dialog, text="Labels (comma-separated):").pack(pady=5)
                labels_var = tk.StringVar(value="0,1,2,3")
                ttk.Entry(dialog, textvariable=labels_var, width=30).pack(pady=5)
                
                ttk.Label(dialog, text="Samples per label:").pack(pady=5)
                samples_var = tk.StringVar(value="2")
                ttk.Entry(dialog, textvariable=samples_var, width=30).pack(pady=5)
                
                def do_generate():
                    try:
                        labels = [int(x.strip()) for x in labels_var.get().split(',')]
                        samples = int(samples_var.get())
                        all_labels = [l for l in labels for _ in range(samples)]
                        trainer.generate_samples(labels=all_labels, num_samples=len(all_labels))
                        messagebox.showinfo("Success", f"Generated {len(all_labels)} samples")
                        dialog.destroy()
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                
                tk.Button(dialog, text="Generate", command=do_generate,
                         bg=Colors.SUCCESS, fg=Colors.BG_DARK,
                         font=("Segoe UI", 10, "bold"),
                         relief="flat", padx=20, pady=5).pack(pady=20)
                
            else:
                messagebox.showerror("Error", "No checkpoint found")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_onnx(self):
        """Export models to ONNX"""
        try:
            from torch.utils.data import DataLoader, TensorDataset
            dummy = DataLoader(TensorDataset(torch.randn(1,3,64,64)), batch_size=1)
            trainer = training.EnhancedLabelTrainer(dummy)
            
            if trainer.load_for_inference():
                trainer.export_onnx()
                messagebox.showinfo("Success", f"Models exported to {config.DIRS['onnx']}")
            else:
                messagebox.showerror("Error", "No checkpoint found")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ============================================================
# Main entry point
# ============================================================
def main():
    # Create directories
    for d in config.DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # Configure logging
    config.logger.handlers.clear()
    config.logger.addHandler(logging.StreamHandler())
    config.logger.setLevel(logging.INFO)

    # Start GUI
    root = tk.Tk()
    app = SchrödingerBridgeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()