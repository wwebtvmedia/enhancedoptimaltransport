# DSP Porting Guide: Mathematical Primitives for Quantized ONNX

This document lists the mathematical primitives and operators extracted from the quantized ONNX models (`drift_static_int8.onnx` and `generator_static_int8.onnx`). These operators are essential for implementing the inference engine on a DSP-like architecture with Multiply-Accumulate (MAC) support.

## 1. MAC-Optimized Operations (High Priority)
These operations are the most computationally intensive and should be implemented using hardware-accelerated MAC primitives.

| Operator | Description | Implementation Strategy |
| :--- | :--- | :--- |
| **Conv** | 2D Convolution | sliding window sum-of-products. Use MAC for inner loops. |
| **Gemm** | General Matrix Multiplication | `alpha * A * B + beta * C`. Core matrix-vector or matrix-matrix multiplication. |
| **MatMul** | Matrix Multiplication | Standard matrix product. Can be implemented as a subset of Gemm. |

## 2. Quantization Primitives
Required for handling the `int8` data flow and mapping between integer and floating-point domains.

| Operator | Formula | Implementation Strategy |
| :--- | :--- | :--- |
| **QuantizeLinear** | `y = clamp(round(x / scale) + zero_point)` | Scale (multiplication by 1/scale), offset (addition), and saturation. |
| **DequantizeLinear** | `y = (x - zero_point) * scale` | Subtraction followed by scaling (multiplication). |

## 3. Element-wise Arithmetic
Basic vector operations.

| Operator | Description |
| :--- | :--- |
| **Add** | Vector-Vector or Vector-Scalar addition. |
| **Sub** | Vector-Vector or Vector-Scalar subtraction. |
| **Mul** | Vector-Vector or Vector-Scalar multiplication. |
| **Div** | Vector-Vector or Vector-Scalar division. |
| **Pow** | Element-wise power (e.g., `x^2`). |
| **Sqrt** | Element-wise square root. |

## 4. Normalization and Reduction
These involve computing statistics over specific dimensions.

| Operator | Description | Implementation Strategy |
| :--- | :--- | :--- |
| **InstanceNormalization** | `y = scale * (x - mean) / sqrt(variance + epsilon) + bias` | Requires calculating mean (sum) and variance (sum of squares) using MAC. |
| **ReduceMean** | Mean of elements across dimensions. | Summation using MAC, followed by division by count. |

## 5. Activation Functions
Non-linear functions. For DSP, these are often implemented via Look-Up Tables (LUT) or piecewise polynomial approximations.

| Operator | Formula |
| :--- | :--- |
| **Sigmoid** | `1 / (1 + exp(-x))` |
| **Tanh** | `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` |
| **Softmax** | `exp(x_i) / sum(exp(x_j))` |

## 6. Transcendental and Trigonometric Functions
Often used in coordinate transforms or attention mechanisms.

| Operator | Description |
| :--- | :--- |
| **Sin** | Sine function. |
| **Cos** | Cosine function. |

## 7. Logic and Comparison

| Operator | Description |
| :--- | :--- |
| **Clip** | Limit values to a range `[min, max]`. |
| **Max** | Element-wise maximum. |
| **Min** | Element-wise minimum. |

## 8. Tensor Manipulation (Non-Mathematical)
While not "math primitives", these are required for data movement and layout management.

- **Concat**: Joining tensors along a dimension.
- **Reshape**: Changing tensor shape without altering data.
- **Slice**: Extracting a sub-tensor.
- **Transpose**: Permuting tensor dimensions (often requires optimized memory strides).

---
*Generated based on operators found in `enhanced_label_sb/onnx/*.onnx`*
