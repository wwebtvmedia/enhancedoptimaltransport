#!/bin/bash
# install_tools.sh - Install VHDL simulation and visualization tools

MISSING_TOOLS=()
command -v ghdl >/dev/null 2>&1 || MISSING_TOOLS+=("ghdl")
command -v gtkwave >/dev/null 2>&1 || MISSING_TOOLS+=("gtkwave")

if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    echo "GHDL and GTKWave are already installed."
    exit 0
fi

echo "--- Updating Package List ---"
sudo apt-get update -y

echo "--- Installing Missing Tools: ${MISSING_TOOLS[*]} ---"
sudo apt-get install -y "${MISSING_TOOLS[@]}"

echo "--- Installation Complete ---"
echo "You can now run the simulation with: ./sim_hdl.sh"
