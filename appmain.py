#!/usr/bin/env python3
"""
appmain.py – Beautiful GUI for Schrödinger Bridge Training
A modern, dark-themed interface with real-time visualization
Enhanced v2.0: Added Gallery, Latent Monitor, and Hot-Swapping.

# --- UPDATED: Added Summary of New Capabilities as requested ---
Summary of New Capabilities:
🔬 Latent Monitor: In the Training tab, you now have a real-time bar chart showing the standard deviation of each latent channel. If a bar turns red, the channel has collapsed, and you should use the Hot-Swap feature to increase the DIVERSITY_WEIGHT.
🖼️ Visual Gallery: A new tab that automatically grabs the latest generation from your samples folder. You no longer need to open your file explorer to see if the images are getting sharper.
🔥 Hot-Swap Interface: Added a new purple button. You can now change Loss Weights while the GPU is still churning. The training loop will respect the new values starting the very next epoch.
Robust Async Processing: The process_log_queue now handles specialized signals (UPDATE_GALLERY, METRICS) independently of the text logs to prevent UI lag.
# ---------------------------------------------------------------
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
from app_context import AppContext
from app_processor import TrainingProcessor

# ============================================================
# Google Material Design Colors - Light Theme
# ============================================================
class Colors:
    BG_DARK = "#f8f9fa"      # Google Gray 50
    BG_MEDIUM = "#ffffff"    # Pure White
    BG_LIGHT = "#ffffff"     # White
    FG = "#202124"           # Google Gray 900 (Text)
    FG_SECONDARY = "#5f6368" # Google Gray 700 (Secondary Text)
    ACCENT = "#1a73e8"       # Google Blue
    ACCENT2 = "#a142f4"      # Google Purple
    SUCCESS = "#1e8e3e"      # Google Green
    WARNING = "#f9ab00"      # Google Yellow
    ERROR = "#d93025"        # Google Red
    BORDER = "#dadce0"       # Google Gray 300
    CARD_SHADOW = "#e8eaed"  # Subtle shadow color

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
        
        # MCP Initialization
        self.ctx = AppContext()
        self.engine = TrainingProcessor(self.ctx)
        # Link context queue to GUI's log_queue
        self.log_queue = self.ctx.log_queue
        
        # Add QueueHandler to config.logger so we see all logs in the UI
        qh = QueueHandler(self.log_queue)
        qh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
        config.logger.addHandler(qh)

        # Set up modern styling
        self.setup_styling()
        
        # Control flags
        self.training_running = False
        self.stop_training_flag = False
        self.metrics = {}
        self.current_epoch = 0
        self.trainer_instance = None
        self.current_preview_image = None # Added for Visual Gallery
        self.training_images_cache = None  # Cache for training data
        self.training_labels_cache = None
        
        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Initialize device
        self.device_var.set(self.engine.initialize_hardware())
        
        # Start queue processing
        self.process_log_queue()
        
        # Load training data preview
        self.load_training_data_preview()

    def setup_styling(self):
        """Configure Google Material Design ttk styles"""
        style = ttk.Style()
        
        # Configure colors
        style.theme_use('clam')
        
        # Use Roboto if available, else Segoe UI
        main_font = ("Roboto", 10) if "Roboto" in tkfont.families() else ("Segoe UI", 10)
        header_font = ("Roboto", 18, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 18, "bold")
        tab_font = ("Roboto", 10, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 10, "bold")
        
        # Configure colors for all elements
        style.configure(".", 
                       background=Colors.BG_DARK,
                       foreground=Colors.FG,
                       fieldbackground=Colors.BG_LIGHT,
                       troughcolor=Colors.BG_DARK,
                       selectbackground=Colors.ACCENT,
                       selectforeground="#ffffff",
                       font=main_font)
        
        # Notebook styling (Google style tabs)
        style.configure("TNotebook", background=Colors.BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", 
                       background=Colors.BG_DARK,
                       foreground=Colors.FG_SECONDARY,
                       padding=[24, 12],
                       font=tab_font,
                       borderwidth=0)
        style.map("TNotebook.Tab",
                 background=[("selected", Colors.BG_DARK)],
                 foreground=[("selected", Colors.ACCENT)],
                 focuscolor=[("selected", Colors.BG_DARK)])
        
        # Button styling (Material Contained Button)
        style.configure("Accent.TButton",
                       background=Colors.ACCENT,
                       foreground="#ffffff",
                       borderwidth=0,
                       focuscolor="none",
                       font=tab_font)
        style.map("Accent.TButton",
                 background=[("active", "#1967d2"), ("pressed", "#185abc")])
        
        # Outlined Button style
        style.configure("TButton",
                       background=Colors.BG_MEDIUM,
                       foreground=Colors.ACCENT,
                       borderwidth=1,
                       bordercolor=Colors.BORDER,
                       focuscolor="none",
                       font=tab_font)
        style.map("TButton",
                 background=[("active", "#f1f3f4")],
                 bordercolor=[("active", Colors.ACCENT)])
        
        # Label styling
        style.configure("TLabel", 
                       background=Colors.BG_DARK,
                       foreground=Colors.FG,
                       font=main_font)
        
        style.configure("Header.TLabel",
                       font=header_font,
                       foreground=Colors.ACCENT,
                       background=Colors.BG_DARK)
        
        style.configure("CardTitle.TLabel",
                       font=("Roboto", 12, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 12, "bold"),
                       foreground=Colors.FG,
                       background=Colors.BG_MEDIUM)
        
        # Entry styling
        style.configure("TEntry",
                       fieldbackground=Colors.BG_LIGHT,
                       foreground=Colors.FG,
                       insertcolor=Colors.FG,
                       borderwidth=1,
                       relief="solid")
        
        style.map("TEntry",
                 bordercolor=[("focus", Colors.ACCENT)])
        
        # Progress bar styling
        style.configure("Horizontal.TProgressbar",
                       background=Colors.ACCENT,
                       troughcolor=Colors.BORDER,
                       borderwidth=0)
        
        # Scrollbar styling
        style.configure("Vertical.TScrollbar",
                       background=Colors.BORDER,
                       troughcolor=Colors.BG_DARK,
                       width=12,
                       borderwidth=0)
        
        # Frame styling (Card with subtle border instead of dark bg)
        style.configure("Card.TFrame",
                       background=Colors.BG_MEDIUM,
                       relief="flat",
                       borderwidth=1)
        # Use a canvas for actual cards to simulate shadows if needed, 
        # but for now, we'll use themed frames with borders.
        
        # Separator
        style.configure("TSeparator", background=Colors.BORDER)

    def create_header(self):
        """Create modern Google-style header"""
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill='x', padx=32, pady=(32, 16))
        
        # Title
        title_label = ttk.Label(header_frame, text="Schrödinger Bridge Trainer", 
                                style="Header.TLabel")
        title_label.pack(side='left')
        
        # Version and info
        info_frame = ttk.Frame(header_frame)
        info_frame.pack(side='right')
        
        self.device_var = tk.StringVar(value="Initializing...")
        device_label = ttk.Label(info_frame, textvariable=self.device_var, 
                                foreground=Colors.FG_SECONDARY, background=Colors.BG_DARK)
        device_label.pack()

    def create_main_content(self):
        """Create main notebook and content"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Create tabs
        self.create_config_tab()
        self.create_training_tab()
        self.create_training_data_tab()
        self.create_gallery_tab()
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
        """Initialize device, detect Intel Arc (XPU) or DirectML, and update config"""
        try:
            import torch
            
            # 1. Device detection logic
            if torch.cuda.is_available():
                config.DEVICE = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
                self.device_var.set(f"🎮 CUDA: {device_name[:25]}...")
                self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
                config.AMP_AVAILABLE = True # CUDA supports AMP
                
            # NEW: Intel Arc support via Intel Extension for PyTorch (IPEX)
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                config.DEVICE = torch.device("xpu")
                device_name = torch.xpu.get_device_name(0)
                self.device_var.set(f"🔵 Intel Arc: {device_name[:25]}...")
                self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
                config.AMP_AVAILABLE = True # IPEX supports AMP
                
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                config.DEVICE = torch.device("mps")
                self.device_var.set("🍎 Apple Silicon (MPS)")
                self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
                config.AMP_AVAILABLE = False
                
            else:
                # NEW: DirectML support for AMD/Intel/Arc on Windows
                try:
                    import torch_directml
                    if torch_directml.is_available():
                        config.DEVICE = torch_directml.device()
                        self.device_var.set("🎮 DirectML (AMD/Intel)")
                        self.draw_indicator(self.indicator_gpu, Colors.SUCCESS)
                        config.AMP_AVAILABLE = False
                    else:
                        raise ImportError
                except ImportError:
                    # Fallback to CPU
                    config.DEVICE = torch.device("cpu")
                    self.device_var.set("💻 CPU (No GPU Acceleration)")
                    self.draw_indicator(self.indicator_gpu, Colors.WARNING)
                    config.AMP_AVAILABLE = False
            
            # 2. Configure device-specific hyperparameters
            device_configs = {
                'cpu':  {'batch': 16, 'lr': 1e-4},
                'mps':  {'batch': 12, 'lr': 1.5e-4},
                'xpu':  {'batch': 32, 'lr': 2e-4}, # Intel Arc
                'privateuseone': {'batch': 32, 'lr': 2e-4}, # DirectML
                'cuda': {'batch': 32, 'lr': 2e-4}
            }
            
            # Fallback to default (CUDA-like) if device type not found
            settings = device_configs.get(config.DEVICE.type, device_configs['cuda'])
            config.BATCH_SIZE = settings['batch']
            config.LR = settings['lr']
            
            config.DTYPE = torch.float32
            config.logger.info(f"Hardware Initialized: {config.DEVICE} | AMP: {config.AMP_AVAILABLE}")
            
        except Exception as e:
            self.device_var.set("⚠️ Device init failed")
            self.draw_indicator(self.indicator_gpu, Colors.ERROR)
            config.logger.error(f"Hardware initialization failed: {e}")

    # ============================================================
    # Training Data Tab - View Actual Training Images
    # ============================================================
    def create_training_data_tab(self):
        """Create tab to view actual training images from the dataset"""
        self.training_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_data_frame, text="📚 Training Data")
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.training_data_frame, bg=Colors.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.training_data_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar.pack(side="right", fill="y")
        
        # Control panel
        control_frame = CardFrame(scrollable_frame, title="Training Data Viewer")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Controls
        btn_frame = ttk.Frame(control_frame.content)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Label(btn_frame, text="Images per row:").pack(side='left', padx=5)
        self.nrow_var = tk.IntVar(value=8)
        nrow_spin = ttk.Spinbox(btn_frame, from_=2, to=16, textvariable=self.nrow_var, width=5)
        nrow_spin.pack(side='left', padx=5)
        
        ttk.Label(btn_frame, text="Max images:").pack(side='left', padx=5)
        self.max_images_var = tk.IntVar(value=64)
        max_spin = ttk.Spinbox(btn_frame, from_=8, to=200, textvariable=self.max_images_var, width=8)
        max_spin.pack(side='left', padx=5)
        
        refresh_btn = tk.Button(btn_frame, text="🔄 Refresh", 
                               command=self.load_training_data_preview,
                               bg=Colors.ACCENT, fg=Colors.BG_DARK,
                               font=("Segoe UI", 10, "bold"),
                               relief="flat", padx=15, pady=5, cursor="hand2")
        refresh_btn.pack(side='left', padx=10)
        
        compare_btn = tk.Button(btn_frame, text="🖼️ Compare with Latest Generation", 
                               command=self.compare_with_generation,
                               bg=Colors.ACCENT2, fg=Colors.BG_DARK,
                               font=("Segoe UI", 10, "bold"),
                               relief="flat", padx=15, pady=5, cursor="hand2")
        compare_btn.pack(side='left', padx=10)
        
        # Info label
        self.data_info_var = tk.StringVar(value="Loading training data...")
        info_label = ttk.Label(btn_frame, textvariable=self.data_info_var, foreground=Colors.FG_SECONDARY)
        info_label.pack(side='right', padx=10)
        
        # Image display area
        self.training_display_frame = ttk.Frame(scrollable_frame)
        self.training_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Label legend
        legend_frame = CardFrame(scrollable_frame, title="Class Legend")
        legend_frame.pack(fill='x', padx=10, pady=5)
        
        class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                      "dog", "frog", "horse", "ship", "truck"]
        # Professional categorical palette
        colors = ['#E11D48', '#D97706', '#059669', '#2563EB', '#7C3AED',
                  '#DB2777', '#0891B2', '#4B5563', '#92400E', '#1E40AF']
        
        legend_grid = ttk.Frame(legend_frame.content)
        legend_grid.pack()
        
        for i, (name, color) in enumerate(zip(class_names, colors)):
            label_frame = ttk.Frame(legend_grid)
            label_frame.grid(row=i//5, column=i%5, padx=10, pady=2, sticky='w')
            
            color_box = tk.Canvas(label_frame, width=12, height=12, bg=color, highlightthickness=0)
            color_box.pack(side='left')
            
            ttk.Label(label_frame, text=f"{i}: {name}").pack(side='left', padx=5)

    def load_training_data_preview(self):
        """Load and display actual training images from the dataset"""
        try:
            import torch
            import torchvision.utils as vutils
            
            # Load data loader
            loader = dm.load_data()
            
            # Get a batch of images
            max_images = self.max_images_var.get()
            nrow = self.nrow_var.get()
            
            all_images = []
            all_labels = []
            
            # Collect images from multiple batches if needed
            for batch_idx, batch in enumerate(loader):
                if len(all_images) >= max_images:
                    break
                
                if isinstance(batch, dict):
                    images = batch['image']
                    labels = batch['label']
                else:
                    images, labels = batch
                
                # Convert to display range [0, 1]
                images_display = (images + 1) / 2
                images_display = torch.clamp(images_display, 0, 1)
                
                remaining = max_images - len(all_images)
                all_images.append(images_display[:remaining])
                all_labels.extend(labels[:remaining].tolist())
            
            if not all_images:
                self.data_info_var.set("No images loaded!")
                return
            
            # Concatenate images
            all_images_tensor = torch.cat(all_images, dim=0)[:max_images]
            all_labels_list = all_labels[:max_images]
            
            # Cache for later use
            self.training_images_cache = all_images_tensor
            self.training_labels_cache = all_labels_list
            
            # Create grid
            grid = vutils.make_grid(all_images_tensor, nrow=nrow, padding=2, normalize=False)
            
            # Convert to PIL Image
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = (grid_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(grid_np)
            
            # Resize for display (max width 1200)
            max_width = 1200
            if pil_image.width > max_width:
                ratio = max_width / pil_image.width
                new_size = (max_width, int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.training_photo = ImageTk.PhotoImage(pil_image)
            
            # Clear previous display
            for widget in self.training_display_frame.winfo_children():
                widget.destroy()
            
            # Create scrollable canvas for image
            image_canvas = tk.Canvas(self.training_display_frame, bg=Colors.BG_DARK, highlightthickness=0)
            image_canvas.pack(fill='both', expand=True)
            
            image_canvas.create_image(0, 0, anchor='nw', image=self.training_photo)
            image_canvas.configure(scrollregion=image_canvas.bbox("all"))
            
            # Add scrollbar to image canvas
            v_scrollbar = ttk.Scrollbar(self.training_display_frame, orient="vertical", command=image_canvas.yview)
            h_scrollbar = ttk.Scrollbar(self.training_display_frame, orient="horizontal", command=image_canvas.xview)
            image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            v_scrollbar.pack(side='right', fill='y')
            h_scrollbar.pack(side='bottom', fill='x')
            
            # Add click-to-view functionality
            def on_click(event):
                x, y = event.x, event.y
                # Calculate which image was clicked
                img_width = pil_image.width
                img_height = pil_image.height
                img_per_row = nrow
                img_size = img_width // img_per_row
                
                col = x // img_size
                row = y // img_size
                idx = int(row * img_per_row + col)
                
                if 0 <= idx < len(all_labels_list):
                    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                                  "dog", "frog", "horse", "ship", "truck"]
                    label = all_labels_list[idx]
                    
                    # Create popup window to show individual image
                    popup = tk.Toplevel(self.root)
                    popup.title(f"Training Image #{idx} - Class {label}: {class_names[label]}")
                    popup.geometry("300x300")
                    popup.configure(bg=Colors.BG_DARK)
                    
                    # Get individual image
                    single_img = all_images_tensor[idx]
                    single_img_np = single_img.permute(1, 2, 0).cpu().numpy()
                    single_img_np = (single_img_np * 255).astype(np.uint8)
                    single_pil = Image.fromarray(single_img_np)
                    single_pil.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    
                    photo = ImageTk.PhotoImage(single_pil)
                    img_label = tk.Label(popup, image=photo, bg=Colors.BG_DARK)
                    img_label.image = photo
                    img_label.pack(pady=20)
                    
                    ttk.Label(popup, text=f"Class {label}: {class_names[label]}", 
                             font=("Segoe UI", 12, "bold")).pack()
                    ttk.Label(popup, text=f"Index: {idx}").pack()
                    
                    tk.Button(popup, text="Close", command=popup.destroy,
                             bg=Colors.ACCENT, fg=Colors.BG_DARK,
                             relief="flat", padx=20, pady=5).pack(pady=10)
            
            image_canvas.bind("<Button-1>", on_click)
            
            self.data_info_var.set(f"Loaded {len(all_labels_list)} training images")
            config.logger.info(f"Training data preview loaded: {len(all_labels_list)} images")
            
        except Exception as e:
            self.data_info_var.set(f"Error loading data: {str(e)[:50]}")
            config.logger.error(f"Failed to load training data preview: {e}")
            import traceback
            traceback.print_exc()

    def compare_with_generation(self):
        """Compare training images with the latest generated samples"""
        # Find the latest generated sample
        samples_dir = config.DIRS["samples"]
        png_files = list(samples_dir.glob("gen_*.png"))
        
        if not png_files:
            messagebox.showinfo("No Samples", "No generated samples found. Train the model first or generate samples.")
            return
        
        latest_sample = max(png_files, key=lambda p: p.stat().st_mtime)
        
        if self.training_images_cache is None or len(self.training_images_cache) == 0:
            messagebox.showinfo("No Training Data", "Please refresh the Training Data tab first.")
            return
        
        # Create comparison window
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Comparison: Training Data vs Generated Samples")
        compare_window.geometry("1400x800")
        compare_window.configure(bg=Colors.BG_DARK)
        
        # Create notebook for comparison views
        compare_notebook = ttk.Notebook(compare_window)
        compare_notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Side by Side
        side_by_side_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(side_by_side_frame, text="Side by Side")
        
        # Left panel - Training images
        left_card = CardFrame(side_by_side_frame, title="Training Data (First 16 Images)")
        left_card.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Right panel - Generated images
        right_card = CardFrame(side_by_side_frame, title=f"Generated Samples - {latest_sample.name}")
        right_card.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Display training images (first 16)
        train_display = self.training_images_cache[:16]
        nrow = 4
        import torchvision.utils as vutils
        train_grid = vutils.make_grid(train_display, nrow=nrow, padding=2, normalize=False)
        train_grid_np = train_grid.permute(1, 2, 0).cpu().numpy()
        train_grid_np = (train_grid_np * 255).astype(np.uint8)
        train_pil = Image.fromarray(train_grid_np)
        train_pil.thumbnail((500, 500), Image.Resampling.LANCZOS)
        train_photo = ImageTk.PhotoImage(train_pil)
        
        train_label = ttk.Label(left_card.content, image=train_photo)
        train_label.image = train_photo
        train_label.pack(pady=10)
        
        # Display generated images
        try:
            gen_img = Image.open(latest_sample)
            gen_img.thumbnail((500, 500), Image.Resampling.LANCZOS)
            gen_photo = ImageTk.PhotoImage(gen_img)
            gen_label = ttk.Label(right_card.content, image=gen_photo)
            gen_label.image = gen_photo
            gen_label.pack(pady=10)
        except Exception as e:
            ttk.Label(right_card.content, text=f"Error loading image: {e}").pack()
        
        # Tab 2: Training Data Labels
        labels_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(labels_frame, text="Training Labels")
        
        labels_card = CardFrame(labels_frame, title="Training Images with Labels")
        labels_card.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create grid with labels
        class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                      "dog", "frog", "horse", "ship", "truck"]
        
        # Show first 32 images with labels
        nrow_labels = 8
        for idx in range(min(32, len(self.training_images_cache))):
            row = idx // nrow_labels
            col = idx % nrow_labels
            
            frame = ttk.Frame(labels_card.content)
            frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Get individual image
            img = self.training_images_cache[idx]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img.thumbnail((80, 80), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            
            img_label = ttk.Label(frame, image=photo)
            img_label.image = photo
            img_label.pack()
            
            label = self.training_labels_cache[idx]
            ttk.Label(frame, text=f"{label}: {class_names[label]}", 
                     font=("Segoe UI", 8)).pack()
        
        # Tab 3: Class Distribution
        dist_frame = ttk.Frame(compare_notebook)
        compare_notebook.add(dist_frame, text="Class Distribution")
        
        from collections import Counter
        label_counts = Counter(self.training_labels_cache)
        
        dist_card = CardFrame(dist_frame, title="Label Distribution in Training Data")
        dist_card.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create text display
        dist_text = scrolledtext.ScrolledText(dist_card.content, height=15, width=40,
                                              bg=Colors.BG_DARK, fg=Colors.FG,
                                              font=("Consolas", 10))
        dist_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        dist_text.insert(tk.END, "Class Distribution:\n\n")
        for label in range(10):
            count = label_counts.get(label, 0)
            percentage = (count / len(self.training_labels_cache)) * 100 if self.training_labels_cache else 0
            dist_text.insert(tk.END, f"{label:2d} - {class_names[label]:12s}: {count:4d} images ({percentage:5.1f}%)\n")
            dist_text.insert(tk.END, "█" * int(percentage / 2) + "\n\n")
        
        dist_text.configure(state='disabled')

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
            ("⚡ Training Hyperparameters", ["LR", "EPOCHS", "WEIGHT_DECAY", "GRAD_CLIP", "BATCH_SIZE", "CST_COEF_GAUSSIAN_PRIO"]),
            ("⚖️ Loss Weights", ["KL_WEIGHT", "RECON_WEIGHT", "DRIFT_WEIGHT", "DIVERSITY_WEIGHT", "CONSISTENCY_WEIGHT", "PERCEPTUAL_WEIGHT", "SSIM_WEIGHT"]),
            ("🎨 VAE Specific", ["LATENT_SCALE", "FREE_BITS", "DIVERSITY_TARGET_STD", "DIVERSITY_BALANCE_WEIGHT",
                                "KL_ANNEALING_EPOCHS", "LOGVAR_CLAMP_MIN", "LOGVAR_CLAMP_MAX", "MU_NOISE_SCALE"]),
            ("📅 Training Schedule", ["PHASE1_EPOCHS", "PHASE2_EPOCHS"]),
            ("🧠 Neural Tokenizer", ["USE_NEURAL_TOKENIZER", "TEXT_EMBEDDING_DIM", "CONTRASTIVE_WEIGHT"]),
            ("🔮 Inference", ["DEFAULT_STEPS", "DEFAULT_SEED", "INFERENCE_TEMPERATURE",
                             "LANGEVIN_STEP_SIZE", "LANGEVIN_SCORE_SCALE"]),
            ("✨ Enhanced Features", ["USE_SUBPIXEL_CONV", "USE_PERCENTILE", "USE_SNAPSHOTS", "USE_AMP"]),
        ]

        self.config_vars = {}

        for group_name, param_list in param_groups:
            # Create card for each group
            card = CardFrame(scrollable_frame, title=group_name)
            card.pack(fill='x', padx=10, pady=5)

            # Create parameter grid
            for i, param in enumerate(param_list):
                self.create_param_row(card.content, param, i)

    def create_param_row(self, parent, param, row):
        """Create a parameter row with Google-style inputs"""
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill='x', pady=4)

        # Get current value with fallback
        default = getattr(config, param, None)
        if default is None:
            # Check if it's in a dict (like TRAINING_SCHEDULE)
            if param == "TRAINING_SCHEDULE":
                default = config.TRAINING_SCHEDULE.get('mode', 'auto')
            else:
                default = ""
        
        # Label
        label = ttk.Label(frame, text=param, width=25, anchor='w', 
                         foreground=Colors.FG_SECONDARY, background=Colors.BG_MEDIUM)
        label.pack(side='left', padx=(0, 10))
        
        # Add tooltip
        tooltip_text = self.get_param_description(param)
        ToolTip(label, tooltip_text)

        # Input field
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

    def create_training_tab(self):
        """Create Google-style training control tab"""
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🎮 Training")

        # Left panel
        left_panel = ttk.Frame(self.training_frame, width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 16))
        left_panel.pack_propagate(False)

        # Control card
        control_card = CardFrame(left_panel, title="Controls")
        control_card.pack(fill='x', pady=(0, 16))

        self.start_btn = tk.Button(control_card.content, text="Start Training", 
                                  command=self.start_training,
                                  bg=Colors.ACCENT, fg="#ffffff",
                                  font=("Roboto", 10, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 10, "bold"),
                                  relief="flat", padx=24, pady=10, cursor="hand2")
        self.start_btn.pack(fill='x', pady=4)
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#1967d2"))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg=Colors.ACCENT))

        self.swap_btn = tk.Button(control_card.content, text="Hot-Swap Weights", 
                                 command=self.hot_swap_weights,
                                 bg=Colors.BG_MEDIUM, fg=Colors.ACCENT2,
                                 font=("Roboto", 10, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 10, "bold"),
                                 relief="flat", pady=8, cursor="hand2",
                                 highlightthickness=1, highlightbackground=Colors.BORDER)
        self.swap_btn.pack(fill='x', pady=4)

        self.stop_btn = tk.Button(control_card.content, text="Stop Training", 
                                 command=self.stop_training, state=tk.DISABLED,
                                 bg="#ffffff", fg=Colors.ERROR,
                                 font=("Roboto", 10, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 10, "bold"),
                                 relief="flat", pady=10, cursor="hand2",
                                 highlightthickness=1, highlightbackground=Colors.BORDER)
        self.stop_btn.pack(fill='x', pady=4)

        # Progress card
        progress_card = CardFrame(left_panel, title="Progress")
        progress_card.pack(fill='x', pady=(0, 16))

        self.epoch_var = tk.StringVar(value="0 / 0")
        ttk.Label(progress_card.content, text="Current Epoch", foreground=Colors.FG_SECONDARY, background=Colors.BG_MEDIUM).pack(anchor='w')
        epoch_label = ttk.Label(progress_card.content, textvariable=self.epoch_var,
                               font=("Roboto", 24, "bold") if "Roboto" in tkfont.families() else ("Segoe UI", 24, "bold"), 
                               foreground=Colors.ACCENT, background=Colors.BG_MEDIUM)
        epoch_label.pack(anchor='w', pady=(0, 8))

        self.progress = ttk.Progressbar(progress_card.content, orient='horizontal', mode='determinate')
        self.progress.pack(fill='x', pady=8)

        # Metrics card
        metrics_card = CardFrame(left_panel, title="Live Metrics")
        metrics_card.pack(fill='both', expand=True)

        self.metrics_text = scrolledtext.ScrolledText(metrics_card.content,
                                                     height=8,
                                                     bg="#f1f3f4",
                                                     fg=Colors.FG,
                                                     font=("Consolas", 10),
                                                     relief="flat",
                                                     padx=8, pady=8)
        self.metrics_text.pack(fill='x', expand=False, pady=(0, 12))

        # Latent Monitor
        ttk.Label(metrics_card.content, text="Latent Channel Stdev", foreground=Colors.FG_SECONDARY, background=Colors.BG_MEDIUM).pack(anchor='w')
        self.latent_canvas = tk.Canvas(metrics_card.content, bg="#ffffff", height=120, highlightthickness=1, highlightbackground=Colors.BORDER)
        self.latent_canvas.pack(fill='x', expand=True, pady=8)

        # Right panel - Log
        right_panel = ttk.Frame(self.training_frame)
        right_panel.pack(side='right', fill='both', expand=True)

        log_card = CardFrame(right_panel, title="Training Log")
        log_card.pack(fill='both', expand=True)

        self.log_text = scrolledtext.ScrolledText(log_card.content,
                                                 wrap='word',
                                                 bg="#ffffff",
                                                 fg=Colors.FG,
                                                 font=("Consolas", 10),
                                                 relief="flat")
        self.log_text.pack(fill='both', expand=True)
        self.log_text.tag_configure("error", foreground=Colors.ERROR)
        self.log_text.tag_configure("warning", foreground=Colors.WARNING)
        self.log_text.tag_configure("success", foreground=Colors.SUCCESS)
        self.log_text.tag_configure("info", foreground=Colors.FG)


    def create_gallery_tab(self):
        """Enhancement: Gallery tab for real-time visual results"""
        self.gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gallery_frame, text="🖼️ Gallery")
        
        self.gal_scroll_canvas = tk.Canvas(self.gallery_frame, bg=Colors.BG_DARK, highlightthickness=0)
        gal_scroll = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.gal_scroll_canvas.yview)
        self.gal_container = ttk.Frame(self.gal_scroll_canvas)
        
        self.gal_scroll_canvas.create_window((0,0), window=self.gal_container, anchor="nw")
        self.gal_scroll_canvas.configure(yscrollcommand=gal_scroll.set)
        self.gal_scroll_canvas.pack(side="left", fill="both", expand=True)
        gal_scroll.pack(side="right", fill="y")
        
        self.preview_card = CardFrame(self.gal_container, title="✨ Latest Sample")
        self.preview_card.pack(fill='x', padx=10, pady=10)
        
        self.preview_label = ttk.Label(self.preview_card.content, text="Waiting for samples...", font=("Segoe UI", 12))
        self.preview_label.pack(pady=10)
        
        refresh_btn = tk.Button(self.preview_card.content, text="🔄 Refresh Gallery", 
                               command=self.refresh_gallery,
                               bg=Colors.ACCENT, fg=Colors.BG_DARK,
                               font=("Segoe UI", 10, "bold"),
                               relief="flat", padx=15, pady=5, cursor="hand2")
        refresh_btn.pack(pady=5)
    
    def refresh_gallery(self):
        """Load the latest generated sample from the samples folder"""
        samples_dir = config.DIRS["samples"]
        png_files = list(samples_dir.glob("gen_*.png"))
        if png_files:
            latest = max(png_files, key=lambda p: p.stat().st_mtime)
            self.update_gallery(str(latest))

    def hot_swap_weights(self):
        """Open a dialog to hot-swap training loss weights in real-time"""
        dialog = tk.Toplevel(self.root)
        dialog.title("🔥 Hot-Swap Loss Weights")
        dialog.geometry("350x250")
        dialog.configure(bg=Colors.BG_DARK)
        dialog.transient(self.root)
        
        weights_to_swap = [
            ("KL_WEIGHT", config.KL_WEIGHT),
            ("RECON_WEIGHT", config.RECON_WEIGHT),
            ("DIVERSITY_WEIGHT", config.DIVERSITY_WEIGHT),
            ("SIM_LOST_FACTOR", config.SIM_LOST_FACTOR),
            ("PERSP_LOST_FACTOR", config.PERSP_LOST_FACTOR)
        ]
        
        vars_dict = {}
        for param, default in weights_to_swap:
            frame = ttk.Frame(dialog)
            frame.pack(fill='x', padx=20, pady=10)
            ttk.Label(frame, text=param, width=18).pack(side='left')
            var = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=var, width=15).pack(side='right')
            vars_dict[param] = var
            
        def apply_swap():
            try:
                for param, var in vars_dict.items():
                    val = float(var.get())
                    setattr(config, param, val)
                self.log_message("🔥 Hot-Swap Applied: New weights will be used next epoch.", "success")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers.")

        tk.Button(dialog, text="Apply Changes", command=apply_swap,
                 bg=Colors.ACCENT2, fg=Colors.BG_DARK,
                 font=("Segoe UI", 10, "bold"),
                 relief="flat", padx=20, pady=8).pack(pady=15)


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
        self.load_btn.pack(side='left', padx=5)

        self.latest_btn = tk.Button(control_bar, text="🔄 Load Latest", 
                                   command=self.load_latest_log,
                                   bg=Colors.ACCENT, fg="#ffffff",
                                   relief="flat", padx=15, pady=5, cursor="hand2")
        self.latest_btn.pack(side='left', padx=5)

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

        # Start engine
        self.engine.start_training(on_epoch_done=self._on_epoch)

        # Start progress updates
        self.update_progress()

    def _on_epoch(self, epoch, losses):
        """Callback from engine when an epoch is finished"""
        self.log_queue.put(f"EPOCH_DONE:{epoch}:{losses}")
        self.log_queue.put(f"PROGRESS:{epoch+1}/{config.EPOCHS}")

    def stop_training(self):
        self.engine.stop_training()
        self.status_var.set("⏹️ Stopping after current epoch...")

    def update_progress(self):
        """Update UI from queue"""
        self.process_log_queue()
        if self.ctx.is_training:
            self.root.after(500, self.update_progress)
        else:
            self.training_running = False
            self.start_btn.config(state=tk.NORMAL, bg=Colors.ACCENT)
            self.stop_btn.config(state=tk.DISABLED, bg=Colors.BG_MEDIUM)
            self.status_var.set("✅ Training stopped")

    def process_log_queue(self):
        """Process messages from training thread with Async capability"""
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

                elif msg == "UPDATE_GALLERY":
                    self.refresh_gallery()

                elif msg.startswith("LATENT_MONITOR:"):
                    stds = eval(msg.split(":", 1)[1])
                    self.draw_latent_monitor(stds)
                    
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

    def draw_latent_monitor(self, stds: List[float]):
        """Draw a live bar chart of latent channel standard deviations"""
        self.latent_canvas.delete("all")
        width = self.latent_canvas.winfo_width()
        height = self.latent_canvas.winfo_height()
        if width <= 1: return # Canvas not yet drawn

        num_channels = len(stds)
        bar_w = width / max(1, num_channels)
        max_val = max(stds) if stds else 1.0
        
        for i, val in enumerate(stds):
            # If standard deviation is too low, the channel collapsed (turn red)
            color = Colors.ERROR if val < 0.1 else Colors.ACCENT
            
            bar_h = (val / max_val) * (height - 20)
            x0 = i * bar_w + 2
            y0 = height - bar_h
            x1 = (i + 1) * bar_w - 2
            y1 = height
            
            self.latent_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

    def update_gallery(self, image_path: str):
        """Load and display the latest generated sample on the Gallery Tab"""
        try:
            if not os.path.exists(image_path):
                return
                
            img = Image.open(image_path)
            
            # Keep original aspect ratio, resize to fit nicely
            img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            self.current_preview_image = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=self.current_preview_image, text="")
        except Exception as e:
            config.logger.error(f"Gallery update failed: {e}")


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
    def load_latest_log(self):
        """Automatically find and load the latest log file"""
        log_dir = config.DIRS["logs"]
        log_files = list(log_dir.glob("train_*.log"))
        if not log_files:
            messagebox.showinfo("No Logs", "No log files found in enhanced_label_sb/logs")
            return
        
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        self.perform_load_log(str(latest_log))

    def load_log_file(self):
        """Open file dialog to load a log file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Log files", "*.log"), ("All files", "*.*")],
            title="Select training log file"
        )
        if filename:
            self.perform_load_log(filename)

    def perform_load_log(self, filename):
        """Parse and load the specified log file"""
        self.status_var.set(f"📂 Loading {Path(filename).name}...")
        self.metrics.clear()

        # Improved Regex for multi-word loss names (e.g., 'Diversity loss')
        epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) complete:')
        loss_pattern = re.compile(r'  ([\w\s]+ loss): ([\d\.eE+-]+)')
        snr_pattern = re.compile(r'  SNR: ([\d\.]+)dB')
        latent_std_pattern = re.compile(r'  Latent std: ([\d\.]+)')

        try:
            with open(filename, 'r') as f:
                lines = f.readlines()

            current_epoch = None
            for line in lines:
                m = epoch_pattern.search(line)
                if m:
                    current_epoch = int(m.group(1)) - 1 # Zero-indexed for consistency
                    self.metrics[current_epoch] = {}
                    continue

                if current_epoch is not None:
                    m = loss_pattern.search(line)
                    if m:
                        # Clean key name: 'Total loss' -> 'total'
                        key = m.group(1).lower().replace(' loss', '').strip()
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
            dummy = DataLoader(TensorDataset(torch.randn(1,3,config.IMG_SIZE,config.IMG_SIZE)), batch_size=1)
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
            from tkinter import filedialog
            export_dir = filedialog.askdirectory(title="Select ONNX Export Directory", initialdir=config.DIRS["onnx"])
            if not export_dir:
                return

            from torch.utils.data import DataLoader, TensorDataset
            dummy = DataLoader(TensorDataset(torch.randn(1,3,config.IMG_SIZE,config.IMG_SIZE)), batch_size=1)
            trainer = training.EnhancedLabelTrainer(dummy)
            
            if trainer.load_for_inference():
                # Temporarily override the config directory for this export
                original_onnx_dir = config.DIRS["onnx"]
                config.DIRS["onnx"] = Path(export_dir)
                config.DIRS["onnx"].mkdir(parents=True, exist_ok=True)
                
                try:
                    trainer.export_onnx()
                    messagebox.showinfo("Success", f"Models exported to {export_dir}")
                finally:
                    # Restore original config
                    config.DIRS["onnx"] = original_onnx_dir
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
    # config.logger is already configured in config.py on import
    # Just ensure we have a StreamHandler for the terminal if needed
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in config.logger.handlers)
    if not has_stream_handler:
        config.logger.addHandler(logging.StreamHandler())
    config.logger.setLevel(logging.INFO)

    # Start GUI
    root = tk.Tk()
    app = SchrödingerBridgeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()