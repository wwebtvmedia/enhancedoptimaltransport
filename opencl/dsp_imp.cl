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

kernel void elementwise_sub(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] - b[i];
}

kernel void elementwise_mul(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] * b[i];
}

kernel void elementwise_div(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = a[i] / b[i];
}

kernel void elementwise_pow(global const float* a, float exponent, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = pow(a[i], exponent);
}

kernel void elementwise_sqrt(global const float* in, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = sqrt(in[i]);
}

kernel void elementwise_max(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = max(a[i], b[i]);
}

kernel void elementwise_min(global const float* a, global const float* b, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = min(a[i], b[i]);
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
                    
                    if (ih < in_h && iw < in_w) {
                        float val = input[(ic * in_h + ih) * in_w + iw];
                        float w = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                        
                        // Core MAC primitive
                        sum += val * w;
                    }
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

kernel void silu_activation(global const float* in, global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];
        out[i] = x / (1.0f + exp(-x));
    }
}

kernel void clip_op(global const float* in, global float* out, float min_val, float max_val, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = clamp(in[i], min_val, max_val);
}

// --- 6. Normalization & Reduction ---

kernel void instance_norm(
    global const float* input,
    global float* output,
    global const float* scale,
    global const float* bias,
    global const float* mean,
    global const float* variance,
    float epsilon,
    int channels,
    int spatial_dim)
{
    int c = get_global_id(1); // Channel
    int s = get_global_id(0); // Spatial (H*W)
    
    if (c < channels && s < spatial_dim) {
        int idx = c * spatial_dim + s;
        float val = input[idx];
        float m = mean[c];
        float v = variance[c];
        float s_val = scale[c];
        float b_val = bias[c];
        
        output[idx] = s_val * (val - m) / sqrt(v + epsilon) + b_val;
    }
}

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

// --- 8. Softmax ---

kernel void softmax_op(global const float* in, global float* out, int inner_dim, int outer_dim) {
    int o = get_global_id(0);
    if (o < outer_dim) {
        // Find max for numerical stability
        float max_val = in[o * inner_dim];
        for (int i = 1; i < inner_dim; i++) {
            max_val = max(max_val, in[o * inner_dim + i]);
        }
        
        // Compute sum of exponentials
        float sum = 0.0f;
        for (int i = 0; i < inner_dim; i++) {
            float exp_val = exp(in[o * inner_dim + i] - max_val);
            out[o * inner_dim + i] = exp_val; // Temporarily store exp
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < inner_dim; i++) {
            out[o * inner_dim + i] /= sum;
        }
    }
}

// --- 9. Tensor Manipulation: Pixel Shuffle ---

kernel void pixel_shuffle(
    global const float* input,
    global float* output,
    int upscale_factor,
    int in_c, int in_h, int in_w)
{
    int oc = get_global_id(2); // Output Channel
    int oh = get_global_id(1); // Output Height
    int ow = get_global_id(0); // Output Width

    int out_h = in_h * upscale_factor;
    int out_w = in_w * upscale_factor;
    int out_c = in_c / (upscale_factor * upscale_factor);

    if (oc < out_c && oh < out_h && ow < out_w) {
        int ih = oh / upscale_factor;
        int iw = ow / upscale_factor;
        int r = upscale_factor;
        
        // PixelShuffle formula: 
        // ic = oc * r^2 + (oh % r) * r + (ow % r)
        int ic = oc * (r * r) + (oh % r) * r + (ow % r);
        
        output[(oc * out_h + oh) * out_w + ow] = input[(ic * in_h + ih) * in_w + iw];
    }
}

// --- 10. Label Conditioning: Broadcast Addition ---
kernel void broadcast_add_vector(
    global float* feature_map,
    global const float* vector,
    int channels, int height, int width)
{
    int c = get_global_id(2);
    int y = get_global_id(1);
    int x = get_global_id(0);

    if (c < channels && y < height && x < width) {
        // Add the channel-specific component of the vector to every pixel
        feature_map[(c * height + y) * width + x] += vector[c];
    }
}
