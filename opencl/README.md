# OpenCL DSP Primitives for Quantized ONNX Inference

This directory contains the OpenCL implementation of mathematical primitives required to run inference for the quantized models used in this project.

## Origin of Source
The primitives implemented here are derived from the operator requirements of the following quantized ONNX models located in `enhanced_label_sb/onnx/`:
- `drift_static_int8.onnx`
- `generator_static_int8.onnx`

These models use `int8` quantization to optimize for edge devices and DSPs. The kernels in `dsp_imp.cl` provide the necessary mathematical backbone to execute these models on hardware supporting OpenCL or Vulkan Compute (SPIR-V).

## Mathematical Primitives (`dsp_imp.cl`)

The following primitives are implemented and optimized for Multiply-Accumulate (MAC) operations, which are common in DSP architectures.

### 1. Quantization & Dequantization
Used to convert between `float32` and `int8` domains.
- **`quantize_linear`**: Maps floating-point values to 8-bit integers using a scale and zero-point.
- **`dequantize_linear`**: Restores 8-bit integers to floating-point values.

### 2. MAC-Optimized Operations
High-performance kernels that form the bulk of neural network computation.
- **`matmul_mac`**: Matrix multiplication optimized for accumulation loops.
- **`conv2d_mac`**: 2D Convolution using a sum-of-products approach.

### 3. Element-wise Arithmetic
Basic vector operations for tensor manipulation.
- **`elementwise_add`**: $C = A + B$
- **`elementwise_sub`**: $C = A - B$
- **`elementwise_mul`**: $C = A \times B$
- **`elementwise_div`**: $C = A / B$
- **`elementwise_pow`**: $C = A^{exponent}$
- **`elementwise_sqrt`**: $C = \sqrt{A}$
- **`elementwise_max` / `elementwise_min`**: Element-wise comparison.

### 4. Activation Functions
Non-linear mappings used between layers.
- **`sigmoid_activation`**: $1 / (1 + e^{-x})$
- **`tanh_activation`**: Standard hyperbolic tangent.
- **`clip_op`**: Clamps values to a specified $[min, max]$ range.

### 5. Normalization & Reduction
Used for statistical operations and feature scaling.
- **`instance_norm`**: Performs Instance Normalization (per-channel scaling and shifting).
- **`reduce_mean`**: Computes the average across specified dimensions.
- **`softmax_op`**: Computes the Softmax function for probability distribution or attention.

### 6. Trigonometric Operations
Used in coordinate transforms and positional encodings.
- **`sincos_op`**: Simultaneously computes sine and cosine.

## Usage in Project
The `vulkan_opencl_app.cpp` file serves as a host-side driver that:
1. Initializes the OpenCL context.
2. Compiles `dsp_imp.cl` into device-specific machine code.
3. Orchestrates the execution of these primitives to simulate the `Drift -> Generator` pipeline.
4. (Optionally) Bridges the output to a Vulkan-based display frontend.

## Implementation Strategy
For porting to a specific DSP or NPU:
1. Replace the OpenCL kernel loops with hardware-specific intrinsic MAC instructions.
2. Utilize local memory (SRAM/TCM) to cache weights and activations.
3. Use DMA controllers to overlap data movement with computation.

---
*See `dsp_porting.md` for a detailed mapping of ONNX operators to these primitives.*
