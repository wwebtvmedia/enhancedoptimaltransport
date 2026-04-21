#!/bin/bash
# kill_gpu_procs.sh - Forcefully kill all processes using NVIDIA GPUs
# Useful when the GPU is so full that PyTorch cannot initialize.

echo "🔍 Scanning for processes using NVIDIA devices..."

# Extract PIDs from all nvidia device files
# We use sort -u to get a unique list of PIDs
PIDS=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+' | sort -u)

if [ -z "$PIDS" ]; then
    echo "✅ No processes found using the GPU."
    exit 0
fi

echo "⚠️ Found the following processes holding VRAM:"
echo "----------------------------------------------------"

for pid in $PIDS; do
    if [ -d "/proc/$pid" ]; then
        # Get the command name
        CMD=$(cat /proc/$pid/cmdline | tr '\0' ' ' | cut -c1-100)
        USER=$(ps -o user= -p $pid)
        echo "💀 Killing PID $pid ($USER): $CMD"
        
        # Kill the process
        kill -9 $pid 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "   -> Successfully terminated."
        else
            echo "   -> ERROR: Could not kill $pid (permission denied?)"
        fi
    fi
done

echo "----------------------------------------------------"
echo "✨ GPU cleanup complete. You can now try running 'python gpu_utils.py' or 'python main.py'."
