# OpenVHDL Hardware Accelerator

This directory contains the VHDL implementation of the DSP hardware accelerator, ported from OpenCL primitives.

## Files
- `dsp_accelerator.vhd`: The core RTL implementation with AXI4-Lite control and AXI4-Master DMA.
- `dsp_accelerator_tb.vhd`: Testbench for functional verification.
- `sim_hdl.sh`: Script to run the simulation using GHDL.
- `install_tools.sh`: Script to install required tools (GHDL, GTKWave).

## Getting Started

### 1. Install Tools
Run the installation script to set up GHDL and GTKWave:
```bash
chmod +x install_tools.sh
./install_tools.sh
```

### 2. Run Simulation
Execute the simulation script:
```bash
chmod +x sim_hdl.sh
./sim_hdl.sh
```
This script will:
1. Compile the VHDL code.
2. Run the functional testbench (processing a 128x128 gradient).
3. Generate `dsp_accel.vcd` for waveforms.
4. Dump the resulting memory image to `output_image.txt`.
5. Automatically run `raw_to_png.py` to generate `output_image.png`.

### 3. View Waveforms
Open the generated VCD file in GTKWave:
```bash
gtkwave dsp_accel.vcd
```

## Architecture
The accelerator uses a **Shadow Register** architecture to allow the host processor to configure the next task while the current one is still executing, maximizing hardware utilization.
