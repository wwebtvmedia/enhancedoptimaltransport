#!/bin/bash
# ============================================================================
# STM32N6 DEPLOYMENT PIPELINE — FULL AUTOMATION
# ============================================================================
# This script orchestrates the entire flow:
# 1. Calibration (FP32 ONNX -> Data)
# 2. Quantization (FP32 -> INT8 QDQ)
# 3. Code Generation (ST Edge AI 4.0)
# 4. Binary Compilation (ARM GCC)
# 5. Hardware Flashing (STM32CubeProgrammer)
# ============================================================================

set -e # Exit on error

# --- 1. CONFIGURATION ---
# Paths (Adjust these if they differ on your system)
STEDGEAI="/home/pc/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai"
PROG="/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin/STM32_Programmer_CLI"
STM32CUBE_DIR="/home/pc/sby/STM32CubeN6"
TOOLCHAIN_DIR="/home/pc/sby/demoicair_fixed/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi"

# Project paths
PROJECT_ROOT=$(pwd)
ONNX_DIR="$PROJECT_ROOT/enhanced_label_sb/onnx"
STAI_OUTPUT="$PROJECT_ROOT/st_ai_output"
APP_DIR="$PROJECT_ROOT/stm32n6_app"

# Memory Addresses
ADDR_FSBL="0x70000000"
ADDR_APP="0x70100400"
ADDR_WEIGHTS="0x71000000"

# --- 2. HELPERS ---
echo_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
echo_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
echo_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; exit 1; }

# --- 3. PREREQUISITES CHECK ---
echo_info "Checking prerequisites..."
[ -f "$STEDGEAI" ] || echo_error "stedgeai not found at $STEDGEAI"
[ -f "$PROG" ] || echo_error "STM32_Programmer_CLI not found at $PROG"
[ -d "$STM32CUBE_DIR" ] || echo_error "STM32CubeN6 not found at $STM32CUBE_DIR"
[ -d ".venv" ] || echo_error "Virtual environment (.venv) not found. Run 'python3 -m venv .venv' first."

source .venv/bin/activate

# --- 4. STEP 1: CALIBRATION ---
echo_info "Step 1: Generating realistic calibration data from FP32 models..."
python3 generate_calibration_data.py
echo_success "Calibration data generated."

# --- 5. STEP 2: QUANTIZATION ---
echo_info "Step 2: Quantizing FP32 ONNX models to optimized INT8 QDQ..."
python3 quantize_static.py
echo_success "Quantization complete. INT8 models saved in $ONNX_DIR"

# --- 6. STEP 3: ST EDGE AI CODE GENERATION ---
echo_info "Step 3: Generating C code and NPU weights binary using ST Edge AI..."
mkdir -p "$STAI_OUTPUT"
"$STEDGEAI" generate \
    --model "$ONNX_DIR/generator_static_int8.onnx" \
    --target stm32n6 \
    "--st-neural-art=-E" \
    --binary \
    --workspace "$PROJECT_ROOT/workspace_gen" \
    --output "$STAI_OUTPUT"
echo_success "ST Edge AI generation complete."

# --- 7. STEP 4: COMPILATION ---
echo_info "Step 4: Compiling STM32N6 application..."
cd "$APP_DIR"
make clean
make \
    STM32CUBE_DIR="$STM32CUBE_DIR" \
    TOOLCHAIN_DIR="$TOOLCHAIN_DIR" \
    STAI_DIR="$STAI_OUTPUT" \
    -j$(nproc)
echo_success "Application binary compiled: $APP_DIR/build/sb_display.bin"

# --- 8. STEP 5: FLASHING ---
read -p "Connect STM32N6570-DK and press Enter to start flashing (or Ctrl+C to skip)..."

# Find FSBL (First Stage Boot Loader)
# This is typically pre-built or found in the CubeN6 templates
FSBL_PATH="$STM32CUBE_DIR/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE/Boot/Debug/Template_XIP_FSBL.bin"

echo_info "Flashing FSBL to $ADDR_FSBL..."
"$PROG" --connect port=SWD freq=8000 reset=HWrst \
    --download "$FSBL_PATH" "$ADDR_FSBL" --verify

echo_info "Flashing optimized INT8 weights to $ADDR_WEIGHTS..."
"$PROG" --connect port=SWD freq=8000 reset=HWrst \
    --download "$STAI_OUTPUT/network_data.bin" "$ADDR_WEIGHTS" --verify

echo_info "Flashing application to $ADDR_APP..."
"$PROG" --connect port=SWD freq=8000 reset=HWrst \
    --download "$APP_DIR/build/sb_display.bin" "$ADDR_APP" --verify --go

echo_success "Deployment complete! The board should now be running the AI display demo."
