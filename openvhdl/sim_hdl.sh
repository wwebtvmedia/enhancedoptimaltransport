#!/bin/bash
# sim_hdl.sh - VHDL Simulation Script for DSP Accelerator

# Check for GHDL
if ! command -v ghdl &> /dev/null
then
    echo "Error: GHDL not found. Please install it (e.g., sudo apt install ghdl)"
    echo "Alternatively, import these files into Vivado or ModelSim:"
    echo "  - opencl/dsp_accelerator.vhd"
    echo "  - opencl/dsp_accelerator_tb.vhd"
    exit 1
fi

echo "--- Analyzing VHDL Files ---"
ghdl -a --std=08 dsp_accelerator.vhd
ghdl -a --std=08 dsp_accelerator_tb.vhd

echo "--- Elaborating Testbench ---"
ghdl -e --std=08 dsp_accelerator_tb

echo "--- Running Simulation ---"
ghdl -r --std=08 dsp_accelerator_tb --stop-time=2ms --vcd=dsp_accel.vcd

if [ $? -eq 0 ]; then
    echo "--- SUCCESS ---"
    echo "VCD file generated: dsp_accel.vcd (View with gtkwave)"
    
    if [ -f "output_image.txt" ]; then
        echo "--- Converting Output to PNG ---"
        python3 raw_to_png.py
    fi
else
    echo "--- FAILURE ---"
fi
