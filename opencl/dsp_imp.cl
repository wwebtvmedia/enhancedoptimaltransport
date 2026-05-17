/**
 * dsp_imp.cl - OpenCL Kernels for ONNX Primitives
 * 
 * This file provides reference implementations for mathematical primitives
 * optimized for DSP-like architectures with Multiply-Accumulate (MAC) support.
 * 
 * Note: While this uses OpenCL C syntax, the logic maps directly to 
 * Vulkan Compute Shaders (GLSL/SPIR-V) and specialized DSP ISAs.
 */

// --- 1. Quantization Primitives ---

kernel void quantize_linear(
    global const float* input,
    global char* output,
    float scale,
    int zero_point,
    int size) 
{
    int i = get_global_id(0);
    if (i < size) {
        float val = input[i] / scale + zero_point;
        // round and clamp to int8 [-128, 127]
        output[i] = (char)clamp((int)round(val), -128, 127);
    }
}

kernel void dequantize_linear(
    global const char* input,
    global float* output,
    float scale,
    int zero_point,
    int size)
{
    int i = get_global_id(0);
    if (i < size) {
        output[i] = (float)(input[i] - zero_point) * scale;
    }
}

// --- 2. Element-wise Arithmetic ---

kernel void elementwise_add(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}

kernel void elementwise_mul(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] * b[i];
}

// --- 3. MAC-Optimized: GEMM / MatMul ---
// Basic version optimized for MAC (Multiply and Accumulate)
kernel void matmul_mac(
    global const float* A, 
    global const float* B, 
    global float* C,
    int M, int N, int K) 
{
    int row = get_global_id(1); // M
    int col = get_global_id(0); // N

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // Core MAC primitive: sum = sum + (A * B)
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --- 4. MAC-Optimized: Conv2D ---
// Simple 2D Convolution (NCWH layout assumption)
kernel void conv2d_mac(
    global const float* input,
    global const float* weight,
    global const float* bias,
    global float* output,
    int in_channels, int out_channels,
    int in_h, int in_w,
    int kernel_h, int kernel_w,
    int out_h, int out_w)
{
    int oc = get_global_id(2); // Output Channel
    int oh = get_global_id(1); // Output Height
    int ow = get_global_id(0); // Output Width

    if (oc < out_channels && oh < out_h && ow < out_w) {
        float sum = bias[oc];
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int ih = oh + kh; // simple stride=1, padding=0
                    int iw = ow + kw;
                    
                    float val = input[(ic * in_h + ih) * in_w + iw];
                    float w = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                    
                    // Core MAC primitive
                    sum += val * w;
                }
            }
        }
        output[(oc * out_h + oh) * out_w + ow] = sum;
    }
}

// --- 5. Activations ---

inline float sigmoid_func(float x) {
    return 1.0f / (1.0f + exp(-x));
}

kernel void sigmoid_activation(global const float* in, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = sigmoid_func(in[i]);
}

kernel void tanh_activation(global const float* in, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = tanh(in[i]);
}

kernel void clip_op(global const float* in, global float* out, float min_val, float max_val, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = clamp(in[i], min_val, max_val);
}

// --- 6. Normalization & Reduction ---

kernel void reduce_mean(global const float* in, global float* out, int inner_dim, int outer_dim) {
    int o = get_global_id(0);
    if (o < outer_dim) {
        float sum = 0.0f;
        for (int i = 0; i < inner_dim; i++) {
            // MAC used for accumulation
            sum += in[o * inner_dim + i];
        }
        out[o] = sum / (float)inner_dim;
    }
}

// --- 7. Trigonometric ---

kernel void sincos_op(global const float* in, global float* out_sin, global float* out_cos, int n) {
    int i = get_global_id(0);
    if (i < n) {
        out_sin[i] = sin(in[i]);
        out_cos[i] = cos(in[i]);
    }
}
