#!/usr/bin/env python3
"""
gpu_utils.py - Standalone utility to clean and reset GPU memory.
Can be imported as a module or run directly from the terminal.
"""

import os
import gc
import sys

def clean_gpu(verbose=True):
    """
    Aggressively reclaim GPU memory. 
    Can be called inside training loops or at startup.
    """
    cleaned = False
    try:
        import torch
        if torch.cuda.is_available():
            # 1. Clear Python references
            gc.collect()
            
            # 2. Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # 3. Clear Inter-Process Communication (IPC)
            torch.cuda.ipc_collect()
            
            # 4. Reset stats for every available GPU
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                # Final empty cache per device
                torch.cuda.empty_cache()
            
            if verbose:
                free_mem, total_mem = torch.cuda.mem_get_info()
                print(f"🧹 GPU Memory Cleaned. Currently: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
            cleaned = True
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            # Intel Arc Support
            gc.collect()
            torch.xpu.empty_cache()
            if verbose:
                print("🧹 XPU Memory Cleaned.")
            cleaned = True
        elif verbose:
            print("ℹ️ No GPU available to clean.")
            
    except ImportError:
        if verbose:
            print("❌ PyTorch not found. Cannot clean GPU.")
    except Exception as e:
        if verbose:
            print(f"⚠️ GPU clean failed: {e}")
            
    return cleaned

if __name__ == "__main__":
    # If run as a script: python gpu_utils.py
    print("🚀 Initiating standalone GPU memory reset...")
    success = clean_gpu(verbose=True)
    if success:
        print("✨ GPU is now as clean as possible.")
    else:
        sys.exit(1)
