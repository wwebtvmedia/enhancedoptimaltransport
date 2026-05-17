# OpenVHDL Hardware Accelerator

This directory contains the VHDL implementation of the DSP hardware accelerator, ported from OpenCL primitives and fully aligned with the Drift/Generator network architectures.

## Key Features
- **RAM-Mapped Parameters**: All weights, biases, and feature maps are stored in emulated RAM and fetched via DMA.
- **Neural Ops Support**: Native hardware support for Conv2D (MAC-optimized), PixelShuffle, and SiLU activations.
- **AXI4-Master DMA**: High-bandwidth data movement for multi-dimensional tensors.
- **AXI4-Lite Control**: Standard interface for host-side configuration.

## Files
- `dsp_accelerator.vhd`: Core RTL implementation.
- `dsp_accelerator_tb.vhd`: Testbench verifying memory-to-memory Conv2D flow.
- `sim_hdl.sh`: Automation script for GHDL simulation.
- `raw_to_png.py`: Utility to visualize hardware memory dumps.

## Register Map (Extended)
| Offset | Register | Description |
| :--- | :--- | :--- |
| 0x00 | Control | bit 0: Start |
| 0x04 | Status | bit 0: Busy, bit 1: Done |
| 0x08 | Opcode | 8: Conv2D, 5: PixelShuffle, 6: SiLU, 0: Add, 2: Mul |
| 0x18 | Addr A | Base Address for Input A (DMA) |
| 0x20 | Addr Out | Base Address for Output (DMA) |
| 0x44 | In Channels | Number of input channels |
| 0x48 | Out Channels | Number of output channels |
| 0x4C | In Height | Input spatial height |
| 0x50 | In Width | Input spatial width |
| 0x54 | K Height | Convolution kernel height |
| 0x58 | K Width | Convolution kernel width |
| 0x5C | Upscale | PixelShuffle factor (r) |
| 0x60 | Addr W | Base Address for Weights (RAM) |
| 0x64 | Addr Bias | Base Address for Bias (RAM) |

## Getting Started

### 1. Install Tools
```bash
chmod +x install_tools.sh
./install_tools.sh
```

### 2. Run Aligned Simulation
```bash
chmod +x sim_hdl.sh
./sim_hdl.sh
```
The simulation verifies the memory-to-memory Conv2D flow, mirroring the software pipeline logic.

### 3. View Waveforms
```bash
gtkwave dsp_accel.vcd
```
