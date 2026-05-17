# ONNX → OpenCL Math Specification

Reference document for the C++/OpenCL re-implementation of:
- `enhanced_label_sb/onnx/drift_static_int8.onnx`
- `enhanced_label_sb/onnx/generator_static_int8.onnx`

Source of truth: the ONNX graphs themselves (loaded via the manifest under
`opencl/runtime/assets/{drift,generator}/manifest.json`).

---

## 1. Pipeline overview

```
text  ──► text_encoder (offline, NOT in ONNX) ──► text_embedding [1,512]
                                                       │
                                                       ▼
z₀ ~ 𝒩(0,σ²) [1,8,12,12] ──┐
                            │ for i = 0..steps-1:
                            │     t = [i/steps]
                            │     d = drift(z, t, text_embedding, cfg_scale)
                            │     z = z + d * (1/steps)
                            ▼
z_final [1,8,12,12] ──► generator(z, text_embedding) ──► img [1,3,96,96]
                                                              │
                                                              ▼
                                       pixel = clip((img+1)*127.5, 0, 255)
```

The text encoder (`NeuralTokenizer`) is *not* in the ONNX. Embeddings are
precomputed (one fp32 file per class in `opencl/runtime/assets/embeddings/`
or single `horse_embedding.bin`).

---

## 2. ONNX op inventory (drift + generator)

| Op                    | drift count | gen count | Notes                                                       |
| --------------------- | ----------: | --------: | ----------------------------------------------------------- |
| Add                   |          85 |        39 | broadcasting                                                |
| Clip                  |           1 |         0 | `clip(x, min, max)` (min/max as tensors)                    |
| Concat                |           3 |         1 | along `axis`                                                |
| Conv                  |          33 |        11 | only configs used: 1×1/p0/s1, 3×3/p1/s1, 4×4/p1/s2 (g=1)    |
| Cos                   |           1 |         0 | for sinusoidal time embedding                               |
| DequantizeLinear      |          83 |        32 | `y = (x − zp) · scale` — fp32 output                        |
| Div                   |          16 |         7 |                                                             |
| Gemm                  |          32 |         8 | always `alpha=1, beta=1, transB=1` → `y = x · Wᵀ + b`       |
| InstanceNormalization |          25 |        10 | per-channel mean/var, attr `epsilon`                        |
| MatMul                |          16 |        16 | batched (N-d), no attrs                                     |
| Max / Min             |       2 / 2 |       0/0 | element-wise                                                |
| Mul                   |         100 |        38 | broadcasting                                                |
| Pow                   |           4 |         4 | exponent is scalar tensor (usually 2)                       |
| QuantizeLinear        |          65 |        21 | `y = clip(round(x/scale) + zp, qmin, qmax)`                 |
| ReduceMean            |           8 |         8 | with `axes`, keepdims=1 by default                          |
| Reshape               |         104 |        56 | shape from second input; attr `allowzero`                   |
| Sigmoid               |          42 |        13 | `σ(x) = 1/(1+e⁻ˣ)`                                          |
| Sin                   |           1 |         0 | sinusoidal time embedding                                   |
| Slice                 |          36 |        18 | inputs: data, starts, ends, axes, steps                     |
| Softmax               |           4 |         4 | along `axis` with numerical-stability max-subtract           |
| Sqrt                  |           4 |         4 |                                                             |
| Sub                   |          17 |         7 |                                                             |
| Tanh                  |           0 |         1 | final activation in generator                               |
| Transpose             |          26 |        27 | attr `perm`                                                 |

Total ops to implement: **23 distinct op types**.

---

## 3. Per-op math (exact spec, matching ONNX opset 14)

### 3.1 Quantization

**QuantizeLinear** (per-tensor or per-axis; scale/zp are tensors)
```
y = clip(round(x / scale) + zero_point, qmin, qmax)
qmin, qmax depend on output dtype: int8 → [-128,127], uint8 → [0,255]
```
`round` is round-half-to-even (banker's rounding) per ONNX spec.

**DequantizeLinear**
```
y = (x_int − zero_point) · scale     // y is fp32
```

> **Pragmatic note**: In these models QDQ pairs are mostly used around weights
> and activations as "fake quantization". A pure-fp32 OpenCL device gets no
> speed win from QDQ — we still implement them exactly to match ORT bit-for-bit.

### 3.2 Conv (2D)
NCHW layout. With pads `[pt,pl,pb,pr]`, strides `[sh,sw]`, dilations `[dh,dw]`,
groups `g`:
```
out[n,oc,oh,ow] = bias[oc] + Σ_ic Σ_kh Σ_kw
                    input[n, ic_in, oh·sh + kh·dh − pt, ow·sw + kw·dw − pl]
                  · weight[oc, ic, kh, kw]
```
where `ic_in = (oc/(out_C/g))·(in_C/g) + ic` for grouped conv (g=1 in our models,
so the simpler `ic_in = ic` applies).

Output spatial:
```
oh_dim = floor((in_H + pt+pb − dh·(kh−1) − 1)/sh + 1)
```

### 3.3 Gemm (always transB=1, alpha=beta=1 in our models)
```
y[i,j] = Σ_k x[i,k] · W[j,k] + b[j]
```

### 3.4 MatMul (general broadcasting)
For 2-D: `y = x @ w`. For higher-rank: ONNX broadcasts leading dims (`np.matmul`
semantics). In these models we see 4-D batched matmul for attention:
`q [B,H,S,D] @ kT [B,H,D,S] → [B,H,S,S]` and then `attn @ v [B,H,S,D]`.

### 3.5 InstanceNormalization (epsilon ≈ 1e-5)
Per (n, c) over spatial dims:
```
μ      = mean_{h,w}(x[n,c,:,:])
σ²     = var_{h,w}(x[n,c,:,:])      // population variance
y[n,c] = scale[c] · (x[n,c] − μ) / sqrt(σ² + epsilon) + bias[c]
```

> **Caveat**: in this model the InstanceNorm `scale` and `bias` inputs are
> constants of value 1 and 0 respectively (the channel-wise affine is applied
> later by a Mul+Add with a label-projected vector). Verify before assuming.

### 3.6 Sigmoid / Tanh / Sin / Cos / Pow / Sqrt
Standard scalar math, element-wise.

### 3.7 Add / Sub / Mul / Div / Max / Min (binary)
NumPy-style broadcasting required (one of the inputs is often a [C] or [1,C,1,1]
tensor broadcast against [N,C,H,W]).

### 3.8 Clip
`y = min(max(x, min_val), max_val)` where min/max may be tensor inputs (opset 11+).

### 3.9 ReduceMean
`y = mean over axes`. Attr `keepdims` (default 1). Used in InstanceNorm-style
manual normalisation chains and in attention softmax temperature.

### 3.10 Softmax (opset 13+)
Normalises along `axis` only (not flattened like opset ≤12):
```
m = max(x, axis=axis, keepdims=true)
e = exp(x − m)
y = e / sum(e, axis=axis, keepdims=true)
```

### 3.11 Reshape / Transpose / Slice / Concat
Pure data movement:
- **Reshape**: shape from second input (int64 tensor); `0` keeps that dim
  unless `allowzero=1`; one `-1` is inferred.
- **Transpose**: permute axes per `perm`.
- **Slice**: `y = x[starts:ends:steps]` per `axes`; defaults: steps=1.
- **Concat**: stack along `axis`.

Implemented as host-side index transforms feeding into `clEnqueueCopyBuffer*` or
small index-permute kernels — no MACs.

---

## 4. Quantization layout in these specific models

The producer is `onnx.quantize 0.1.0` (QDQ format):
- **Weights**: stored as `int8` initializers + per-tensor `fp32 scale` and
  `int8 zero_point`. A DequantizeLinear runs *once at start* to produce fp32
  weights that the Conv/Gemm/MatMul consumes.
- **Activations**: a `QuantizeLinear → DequantizeLinear` pair is inserted
  around tensors at quant-noise injection points. On a fp32 device this is a
  round-trip that loses precision but is a no-op semantically (still matches
  ORT exactly).

So at runtime:
1. Run all the static `DequantizeLinear` ops on weights once at startup → cache
   the fp32 weights in cl_mem.
2. For each forward pass, run the per-step QDQ pairs on activations (or skip
   them in fp32-fast mode — but **the spec requires running them** to match ORT).

---

## 5. Numerical tolerance for verification

When comparing OpenCL/VHDL output to the ORT golden tensor:

| Stage           | Tolerance (max abs err)       | Notes                              |
| --------------- | ------------------------------- | ---------------------------------- |
| Per-op (fp32)   | 1e-4 absolute, 1e-3 relative   | Float reordering causes small drift |
| Drift output    | 1e-3 absolute (sum of 20 steps) | Accumulated rounding                |
| Final image     | ≤ 2 pixel levels (out of 255)  | After clip+round                    |

For VHDL fixed-point: tolerances need to be derived from the chosen Q format.
Start with at most ±3 LSBs of the final output buffer.

---

## 6. Verification harness contract

The C++ runtime (`onnx_opencl_runner`) reads:
- `assets/<model>/manifest.json` — graph
- `assets/<model>/init/*.bin` — initializer weights
- `assets/<model>/golden/<input_set>/*.bin` — ORT reference per node

For each node it computes the output via the OpenCL kernel, then diffs against
the golden tensor. On first divergence beyond tolerance it logs:
```
[FAIL] node #42 InstanceNormalization name=/dec_blocks.0/norm/...
       golden shape=[1,256,12,12] dtype=float32
       max_abs_err=0.0073 (>1e-4)  argmax=(0,17,3,4)
       golden val=0.4291  got=0.4218  diff=+0.0073
       dumping inputs → logs/run_<ts>/node42_input0.bin ...
```
and stops (configurable: `--continue-on-error`).
