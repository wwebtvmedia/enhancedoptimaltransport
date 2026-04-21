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
        
        # 1. Clear Python references first (does not require CUDA context)
        gc.collect()
        
        if not torch.cuda.is_available():
            if verbose: print("ℹ️ No CUDA GPU available.")
            return False

        try:
            # 2. Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # 3. Clear Inter-Process Communication (IPC)
            torch.cuda.ipc_collect()
            
            # 4. Reset stats for every available GPU
            # We wrap this in another try-except because device_count() can trigger OOM
            num_devices = torch.cuda.device_count()
            for i in range(num_devices):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.empty_cache()
            
            if verbose:
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    print(f"🧹 GPU Memory Cleaned. Currently: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
                except:
                    print("🧹 GPU Cache cleared (could not read memory stats - GPU likely full).")
            cleaned = True
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                if verbose: 
                    print("❌ GPU is so full that PyTorch cannot even initialize the cleaning context.")
                    print("👉 Please manually kill processes like 'ollama' or 'firefox' using: fuser -v /dev/nvidia*")
            else:
                if verbose: print(f"⚠️ GPU clean failed: {e}")
            
    except ImportError:
        if verbose: print("❌ PyTorch not found.")
    except Exception as e:
        if verbose: print(f"⚠️ Unexpected error: {e}")
            
    return cleaned

if __name__ == "__main__":
    print("🚀 Initiating standalone GPU memory reset...")
    clean_gpu(verbose=True)
