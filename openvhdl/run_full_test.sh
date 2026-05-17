#!/bin/bash
# run_full_test.sh - Master script to install dependencies and run the hardware test suite

# Exit on any error
set -e

echo "===================================================="
echo "   OpenVHDL Accelerator: Full Toolchain Setup"
echo "===================================================="

# 1. Install System Dependencies
echo "[1/3] Checking dependencies..."

MISSING_TOOLS=()
command -v ghdl >/dev/null 2>&1 || MISSING_TOOLS+=("ghdl")
command -v gtkwave >/dev/null 2>&1 || MISSING_TOOLS+=("gtkwave")

# Check python packages
python3 -c "import numpy; import PIL" >/dev/null 2>&1 || MISSING_TOOLS+=("python3-numpy" "python3-pil")

if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "Installing missing tools: ${MISSING_TOOLS[*]}"
    sudo apt-get update -y
    sudo apt-get install -y ghdl gtkwave python3-pip python3-numpy python3-pil
else
    echo "All tools (GHDL, GTKWave, NumPy, PIL) are already installed."
fi

# 2. Set Permissions
echo "[2/3] Setting script permissions..."
chmod +x install_tools.sh sim_hdl.sh

# 3. Launch Simulation and Image Generation
echo "[3/3] Launching Simulation and Image Generation..."
./sim_hdl.sh

echo ""
echo "===================================================="
echo "   Verification Complete!"
echo "===================================================="
echo "Results generated:"
echo " - Waveforms: openvhdl/dsp_accel.vcd"
echo " - Raw Dump:  openvhdl/output_image.txt"
echo " - PNG Image: openvhdl/output_image.png"
echo ""
echo "To view the waveforms, run: gtkwave dsp_accel.vcd"
echo "To view the image, open output_image.png"
echo "===================================================="
