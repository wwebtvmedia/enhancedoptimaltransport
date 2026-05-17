#!/bin/bash
# setup_pi5.sh - Install and configure OpenCL/Vulkan for Raspberry Pi 5

echo "----------------------------------------------------"
echo "Raspberry Pi 5 OpenCL/Vulkan Setup Script"
echo "----------------------------------------------------"

# 1. Update System
echo "[1/4] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Vulkan & GLFW (Native RPi5 Support)
echo "[2/4] Installing Vulkan drivers and window management..."
sudo apt install -y mesa-vulkan-drivers libvulkan-dev vulkan-tools libglfw3-dev

# 3. Install OpenCL (PoCL for CPU, Mesa for potential GPU)
echo "[3/4] Installing OpenCL implementations..."
# PoCL is highly recommended for RPi5 as it uses ARM NEON SIMD optimizations
sudo apt install -y pocl-opencl-icd ocl-icd-opencl-dev clinfo

# 4. Build Tools
echo "[4/4] Installing build essentials..."
sudo apt install -y build-essential cmake pkg-config

echo "----------------------------------------------------"
echo "Setup Complete!"
echo "Checking environment..."
echo "----------------------------------------------------"

# Display Summary
vulkaninfo --summary | grep "deviceName"
clinfo | grep "Platform Name"

echo "----------------------------------------------------"
echo "To run the app on Pi 5, use: ./build_and_run_pi5.sh"
echo "----------------------------------------------------"
