# ============================================================================
# CONTEXT LAYER - Application State and Configuration Management
# ============================================================================

import os
import sys
import json
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional

# Local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

class AppContext:
    """The 'Source of Truth' for the application state."""
    
    def __init__(self):
        self.config = config
        self.metrics: Dict[int, Dict[str, Any]] = {}
        self.log_queue = queue.Queue()
        self.is_training = False
        self.stop_signal = False
        self.current_epoch = 0
        self.device_info = "Initializing..."
        self.latest_sample_path: Optional[str] = None
        
        # UI Shared Colors (Google Material)
        self.colors = {
            "bg_dark": "#f8f9fa",
            "bg_medium": "#ffffff",
            "accent": "#1a73e8",
            "accent2": "#a142f4",
            "success": "#1e8e3e",
            "error": "#d93025",
            "text": "#202124",
            "text_secondary": "#5f6368",
            "border": "#dadce0"
        }

    def update_metric(self, epoch: int, loss_dict: Dict[str, Any]):
        self.metrics[epoch] = loss_dict
        self.current_epoch = epoch

    def get_param(self, name: str) -> Any:
        return getattr(self.config, name, None)

    def set_param(self, name: str, value: Any):
        if hasattr(self.config, name):
            setattr(self.config, name, value)
            
    def reload_config(self):
        import importlib
        importlib.reload(self.config)

    def save_config_json(self, path: str):
        cfg_dict = {k: v for k, v in vars(self.config).items() if k.isupper() and not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(cfg_dict, f, indent=2, default=str)
