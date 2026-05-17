#!/bin/bash
# setup_env.sh - Install dependencies for OpenCL/Vulkan development

echo "Installing system dependencies..."

# Update package lists
sudo apt-get update

# Install C++ build tools and CMake
sudo apt-get install -y build-essential cmake pkg-config

# Install OpenCL headers and loaders
sudo apt-get install -y opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1

# Install Vulkan SDK (headers and loader)
sudo apt-get install -y libvulkan-dev vulkan-tools

# Install GLFW3 for window management
sudo apt-get install -y libglfw3-dev

# Install useful diagnostic tools
sudo apt-get install -y clinfo

echo "----------------------------------------------------"
echo "Setup complete. You can verify your drivers with:"
echo "  clinfo"
echo "  vulkan-info"
echo "----------------------------------------------------"
