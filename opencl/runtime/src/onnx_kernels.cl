/* onnx_kernels.cl - OpenCL kernels matching ONNX opset 14 semantics.
 *
 * All tensors are float32 unless suffixed (e.g. _i8 for int8 inputs).
 * Broadcasting kernels use precomputed per-axis stride arrays so the host
 * can handle arbitrary broadcast patterns without recompiling.
 */

// =========================================================================
// Quantization (QDQ format used by drift/generator)
// =========================================================================

// QuantizeLinear (int8): y = clip(round(x/scale) + zp, -128, 127)
// Uses banker's rounding (round-half-to-even) to match ONNX.
kernel void quantize_linear_i8(
    global const float* x,
    global char*        y,
    const float         scale,
    const int           zp,
    const int           n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float q = x[i] / scale;
    // round-half-to-even
    float f = floor(q);
    float r = q - f;
    int   v;
    if (r > 0.5f)      v = (int)(f + 1.0f);
    else if (r < 0.5f) v = (int)(f);
    else               v = (((int)f) & 1) ? (int)(f + 1.0f) : (int)f;
    v += zp;
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    y[i] = (char)v;
}

// DequantizeLinear (from int8 weights): y = (x - zp) * scale
kernel void dequantize_linear_i8(
    global const char*  x,
    global float*       y,
    const float         scale,
    const int           zp,
    const int           n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    y[i] = (float)((int)x[i] - zp) * scale;
}

// Per-axis dequant (scale/zp are vectors broadcast along axis `axis_dim`,
// with inner volume `inner` after the axis).
kernel void dequantize_linear_i8_per_axis(
    global const char*  x,
    global const float* scales,    // [axis_dim]
    global const char*  zps,       // [axis_dim]
    global float*       y,
    const int           axis_dim,
    const int           inner,
    const int           n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    int a = (i / inner) % axis_dim;
    y[i] = (float)((int)x[i] - (int)zps[a]) * scales[a];
}

// =========================================================================
// Elementwise (no broadcast — operands have identical shape)
// =========================================================================
#define EWISE_BINARY(NAME, EXPR)                                     \
kernel void ewise_##NAME##_eq(                                       \
    global const float* a, global const float* b,                    \
    global float* y, const int n)                                    \
{                                                                    \
    int i = get_global_id(0);                                        \
    if (i < n) y[i] = (EXPR);                                        \
}
EWISE_BINARY(add, a[i] + b[i])
EWISE_BINARY(sub, a[i] - b[i])
EWISE_BINARY(mul, a[i] * b[i])
EWISE_BINARY(div, a[i] / b[i])
EWISE_BINARY(max, fmax(a[i], b[i]))
EWISE_BINARY(min, fmin(a[i], b[i]))
#undef EWISE_BINARY

// Generic broadcasted binary op.
// strides_* are in *elements*; rank ≤ 8. For each output index we decompose
// into per-axis coords and recompute a[ ] / b[ ] flat index using per-tensor
// strides (which are 0 for broadcast axes).
#define BCAST_BINARY(NAME, EXPR)                                                 \
kernel void bcast_##NAME(                                                        \
    global const float* a, global const float* b, global float* y,               \
    constant const int* out_shape,                                               \
    constant const int* a_strides,                                               \
    constant const int* b_strides,                                               \
    const int rank,                                                              \
    const int n_out)                                                             \
{                                                                                \
    int idx = get_global_id(0);                                                  \
    if (idx >= n_out) return;                                                    \
    int ai = 0, bi = 0;                                                          \
    int rem = idx;                                                               \
    /* compute output coord per axis from slow to fast */                        \
    int dim_size[8];                                                             \
    for (int d = 0; d < rank; ++d) dim_size[d] = out_shape[d];                   \
    int suffix = 1;                                                              \
    int coord[8];                                                                \
    for (int d = rank - 1; d >= 0; --d) {                                        \
        coord[d] = (rem / suffix) % dim_size[d];                                 \
        suffix *= dim_size[d];                                                   \
    }                                                                            \
    for (int d = 0; d < rank; ++d) {                                             \
        ai += coord[d] * a_strides[d];                                           \
        bi += coord[d] * b_strides[d];                                           \
    }                                                                            \
    float av = a[ai];                                                            \
    float bv = b[bi];                                                            \
    y[idx] = (EXPR);                                                             \
}
BCAST_BINARY(add, av + bv)
BCAST_BINARY(sub, av - bv)
BCAST_BINARY(mul, av * bv)
BCAST_BINARY(div, av / bv)
BCAST_BINARY(max, fmax(av, bv))
BCAST_BINARY(min, fmin(av, bv))
#undef BCAST_BINARY

// =========================================================================
// Activations / Unary
// =========================================================================
kernel void act_sigmoid(global const float* x, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = 1.0f / (1.0f + exp(-x[i]));
}
kernel void act_tanh(global const float* x, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = tanh(x[i]);
}
kernel void act_sin(global const float* x, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = sin(x[i]);
}
kernel void act_cos(global const float* x, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = cos(x[i]);
}
kernel void act_sqrt(global const float* x, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = sqrt(x[i]);
}
// Pow with scalar exponent.
kernel void act_pow_scalar(global const float* x, const float p, global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = pow(x[i], p);
}
// Clip with scalar min/max.
kernel void act_clip_scalar(global const float* x, const float lo, const float hi,
                             global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = fmax(lo, fmin(hi, x[i]));
}

// =========================================================================
// Conv2D (NCHW), supports 1x1, 3x3, 4x4 with pad/stride/dilation/groups=1.
// Generic form so a single kernel covers all our cases.
// =========================================================================
kernel void conv2d_nchw(
    global const float* x,           // [N, IC, IH, IW]
    global const float* w,           // [OC, IC, KH, KW]
    global const float* bias,        // [OC]  (may be null sentinel offset)
    global float*       y,           // [N, OC, OH, OW]
    const int N, const int IC, const int IH, const int IW,
    const int OC,
    const int KH, const int KW,
    const int PAD_T, const int PAD_L,
    const int STR_H, const int STR_W,
    const int DIL_H, const int DIL_W,
    const int OH, const int OW,
    const int HAS_BIAS)
{
    int ow = get_global_id(0);
    int oh = get_global_id(1);
    int oc_b = get_global_id(2);
    int n  = oc_b / OC;
    int oc = oc_b % OC;
    if (n >= N || oc >= OC || oh >= OH || ow >= OW) return;

    float acc = HAS_BIAS ? bias[oc] : 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * STR_H + kh * DIL_H - PAD_T;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * STR_W + kw * DIL_W - PAD_L;
                if (iw < 0 || iw >= IW) continue;
                int xi = ((n * IC + ic) * IH + ih) * IW + iw;
                int wi = ((oc * IC + ic) * KH + kh) * KW + kw;
                acc += x[xi] * w[wi];
            }
        }
    }
    y[((n * OC + oc) * OH + oh) * OW + ow] = acc;
}

// =========================================================================
// Gemm (always transB=1, alpha=beta=1 in our models):
//   y[m,n] = sum_k x[m,k] * W[n,k] + b[n]
// =========================================================================
kernel void gemm_transB(
    global const float* x,           // [M, K]
    global const float* w,           // [N, K]   (since transB=1)
    global const float* bias,        // [N]
    global float*       y,           // [M, N]
    const int M, const int N, const int K,
    const int HAS_BIAS)
{
    int j = get_global_id(0);  // N
    int i = get_global_id(1);  // M
    if (i >= M || j >= N) return;
    float acc = HAS_BIAS ? bias[j] : 0.0f;
    for (int k = 0; k < K; ++k) acc += x[i * K + k] * w[j * K + k];
    y[i * N + j] = acc;
}

// =========================================================================
// MatMul (batched, ranks 2-4):
//   y[..., m, n] = sum_k a[..., m, k] * b[..., k, n]
// We pre-flatten the leading "batch" dims on the host so the kernel only sees
// (B, M, K) x (B, K, N) → (B, M, N).
// =========================================================================
kernel void matmul_batched(
    global const float* a,           // [B, M, K]
    global const float* b,           // [B, K, N]
    global float*       y,           // [B, M, N]
    const int B, const int M, const int N, const int K)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    int bb = get_global_id(2);
    if (bb >= B || i >= M || j >= N) return;
    float acc = 0.0f;
    int a_base = bb * M * K;
    int b_base = bb * K * N;
    for (int k = 0; k < K; ++k) acc += a[a_base + i * K + k] * b[b_base + k * N + j];
    y[bb * M * N + i * N + j] = acc;
}

// =========================================================================
// InstanceNormalization (per-channel mean/var over spatial dims)
// One workgroup per (n, c); each thread accumulates a strided slice.
// =========================================================================
kernel void instance_norm(
    global const float* x,           // [N, C, S]    (S = H*W flattened)
    global const float* scale,       // [C]
    global const float* bias,        // [C]
    global float*       y,           // [N, C, S]
    const int N, const int C, const int S,
    const float epsilon)
{
    int c = get_global_id(0);
    int n = get_global_id(1);
    if (n >= N || c >= C) return;
    int base = (n * C + c) * S;
    float sum = 0.0f;
    for (int i = 0; i < S; ++i) sum += x[base + i];
    float mean = sum / (float)S;
    float sq = 0.0f;
    for (int i = 0; i < S; ++i) { float d = x[base + i] - mean; sq += d * d; }
    float var = sq / (float)S;
    float inv = rsqrt(var + epsilon);
    float sc = scale[c];
    float bs = bias[c];
    for (int i = 0; i < S; ++i) y[base + i] = sc * (x[base + i] - mean) * inv + bs;
}

// =========================================================================
// ReduceMean over a contiguous trailing slice.
// Host re-organises tensor so reduced axes are trailing: input [outer, inner].
// y[outer] = mean over inner.
// =========================================================================
kernel void reduce_mean_trailing(
    global const float* x,
    global float*       y,
    const int outer, const int inner)
{
    int o = get_global_id(0);
    if (o >= outer) return;
    float sum = 0.0f;
    for (int i = 0; i < inner; ++i) sum += x[o * inner + i];
    y[o] = sum / (float)inner;
}

// =========================================================================
// Softmax along last axis (host transposes so the softmax axis is last).
//   m = max(x, -1); e = exp(x-m); y = e / sum(e, -1)
// =========================================================================
kernel void softmax_last_axis(
    global const float* x,
    global float*       y,
    const int outer, const int inner)
{
    int o = get_global_id(0);
    if (o >= outer) return;
    int base = o * inner;
    float m = x[base];
    for (int i = 1; i < inner; ++i) m = fmax(m, x[base + i]);
    float sum = 0.0f;
    for (int i = 0; i < inner; ++i) {
        float e = exp(x[base + i] - m);
        y[base + i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < inner; ++i) y[base + i] *= inv;
}

// =========================================================================
// Generic transpose, ranks 2-6.
// out_strides_in_input[d] = stride of input axis perm[d].
// =========================================================================
kernel void transpose_generic(
    global const float* x,
    global float*       y,
    constant const int* out_shape,                // [rank]
    constant const int* out_strides_in_input,     // [rank]
    const int rank,
    const int n_out)
{
    int idx = get_global_id(0);
    if (idx >= n_out) return;
    // Decompose output index into coordinates.
    int rem = idx;
    int dim_size[8];
    for (int d = 0; d < rank; ++d) dim_size[d] = out_shape[d];
    int suffix = 1;
    int coord[8];
    for (int d = rank - 1; d >= 0; --d) {
        coord[d] = (rem / suffix) % dim_size[d];
        suffix *= dim_size[d];
    }
    int in_idx = 0;
    for (int d = 0; d < rank; ++d) in_idx += coord[d] * out_strides_in_input[d];
    y[idx] = x[in_idx];
}

// =========================================================================
// Generic gather for Slice: pre-computed source-element-index list on host.
// (Simpler than parameterising starts/ends/steps in the kernel.)
// =========================================================================
kernel void gather_index(
    global const float* x,
    global const int*   src_idx,
    global float*       y,
    const int n_out)
{
    int i = get_global_id(0);
    if (i >= n_out) y;  // no-op
    if (i < n_out) y[i] = x[src_idx[i]];
}

// =========================================================================
// Concat: pre-flattened on host into pairs (src_buffer, offset, count).
// For simplicity we expose a single-piece copy kernel; host calls it once
// per concat input and tracks destination offsets.
// =========================================================================
kernel void copy_chunk(
    global const float* src,
    global float*       dst,
    const int src_offset,
    const int dst_offset,
    const int n)
{
    int i = get_global_id(0);
    if (i < n) dst[dst_offset + i] = src[src_offset + i];
}
