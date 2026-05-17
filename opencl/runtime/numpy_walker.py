"""
numpy_walker.py - Reference walker that mirrors the C++ ONNX→OpenCL runtime
                  but computes each op in pure numpy.

Purpose: bisect bugs. If numpy_walker matches the ORT golden but the C++
runtime produces garbage, the bug is in the OpenCL kernels. If numpy_walker
ALSO diverges, the bug is in the graph dispatch / op semantics shared
between Python and C++.

Usage:
    python3 numpy_walker.py drift   --steps 1
    python3 numpy_walker.py generator
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ----- Manifest / init loading -----
DTYPE_MAP = {
    "float32": np.float32, "uint8": np.uint8, "int8": np.int8,
    "uint16": np.uint16, "int16": np.int16, "int32": np.int32,
    "int64": np.int64, "bool": np.bool_, "float16": np.float16,
}


def load_manifest(path):
    return json.load(open(path))


def load_init(asset_dir, init_meta):
    p = Path(asset_dir) / init_meta["file"]
    dt = DTYPE_MAP[init_meta["dtype"]]
    arr = np.fromfile(str(p), dtype=dt).reshape(init_meta["shape"])
    return arr


# ----- Op handlers (mirror runtime.cpp) -----

def op_dequantize_linear(inputs, attrs):
    x, scale, zp = inputs
    if scale.size == 1 and zp.size == 1:
        return ((x.astype(np.int32) - int(zp.item())) * float(scale.item())).astype(np.float32)
    # per-axis: assume axis 0
    axis_dim = x.shape[0]
    inner = int(np.prod(x.shape[1:])) if x.ndim > 1 else 1
    sc = scale.reshape(-1).astype(np.float32)
    zz = zp.reshape(-1).astype(np.int32)
    s = sc.reshape((axis_dim,) + (1,) * (x.ndim - 1))
    z = zz.reshape((axis_dim,) + (1,) * (x.ndim - 1))
    return ((x.astype(np.int32) - z) * s).astype(np.float32)


def op_quantize_linear(inputs, attrs):
    x, scale, zp = inputs
    sv = float(scale.item()) if scale.size == 1 else scale
    zpv = int(zp.item()) if zp.size == 1 else zp
    # banker's rounding
    q = x / sv
    r = np.rint(q).astype(np.int32) + zpv
    if zp.dtype == np.int8:
        return np.clip(r, -128, 127).astype(np.int8)
    if zp.dtype == np.uint8:
        return np.clip(r, 0, 255).astype(np.uint8)
    return r.astype(zp.dtype)


def op_binary(inputs, attrs, fn):
    a, b = inputs
    return fn(a, b).astype(np.float32 if a.dtype == np.float32 else a.dtype)


def op_conv(inputs, attrs):
    x = inputs[0]              # [N, IC, IH, IW]
    w = inputs[1]              # [OC, IC, KH, KW]
    b = inputs[2] if len(inputs) > 2 else None
    kshape = attrs.get("kernel_shape", {"value": list(w.shape[2:])})["value"]
    pads   = attrs.get("pads",         {"value": [0, 0, 0, 0]})["value"]
    strides= attrs.get("strides",      {"value": [1, 1]})["value"]
    dils   = attrs.get("dilations",    {"value": [1, 1]})["value"]
    group  = attrs.get("group",        {"value": 1})["value"]

    N, IC, IH, IW = x.shape
    OC = w.shape[0]
    KH, KW = kshape
    PT, PL, PB, PR = pads
    SH, SW = strides
    DH, DW = dils
    OH = (IH + PT + PB - DH*(KH-1) - 1) // SH + 1
    OW = (IW + PL + PR - DW*(KW-1) - 1) // SW + 1

    # naive nested-loop, slow but obviously correct
    x_pad = np.pad(x, ((0,0),(0,0),(PT,PB),(PL,PR)))
    y = np.zeros((N, OC, OH, OW), dtype=np.float32)
    for n in range(N):
      for oc in range(OC):
        acc = np.zeros((OH, OW), dtype=np.float32)
        for ic in range(IC):
          for kh in range(KH):
            for kw in range(KW):
              acc += w[oc, ic, kh, kw] * x_pad[n, ic,
                                              kh*DH:kh*DH+OH*SH:SH,
                                              kw*DW:kw*DW+OW*SW:SW]
        y[n, oc] = acc
    if b is not None:
        y += b.reshape(1, OC, 1, 1)
    return y


def op_gemm(inputs, attrs):
    x, w = inputs[0], inputs[1]
    b = inputs[2] if len(inputs) > 2 else None
    alpha = attrs.get("alpha", {"value": 1.0})["value"]
    beta  = attrs.get("beta",  {"value": 1.0})["value"]
    transA= attrs.get("transA",{"value": 0})["value"]
    transB= attrs.get("transB",{"value": 0})["value"]
    A = x.T if transA else x
    B = w.T if transB else w
    y = alpha * (A @ B)
    if b is not None: y = y + beta * b
    return y.astype(np.float32)


def op_matmul(inputs, attrs):
    return (inputs[0] @ inputs[1]).astype(np.float32)


def op_instance_norm(inputs, attrs):
    x, scale, bias = inputs
    eps = attrs.get("epsilon", {"value": 1e-5})["value"]
    N, C = x.shape[0], x.shape[1]
    flat = x.reshape(N, C, -1)
    m = flat.mean(axis=2, keepdims=True)
    v = flat.var(axis=2, keepdims=True)  # population variance (numpy default ddof=0)
    nrm = (flat - m) / np.sqrt(v + eps)
    sc = scale.reshape(1, C, 1)
    bs = bias.reshape(1, C, 1)
    return (nrm * sc + bs).reshape(x.shape).astype(np.float32)


def op_reduce_mean(inputs, attrs):
    x = inputs[0]
    axes = tuple(attrs.get("axes", {"value": list(range(x.ndim))})["value"])
    keepdims = attrs.get("keepdims", {"value": 1})["value"]
    return x.mean(axis=axes, keepdims=bool(keepdims)).astype(np.float32)


def op_softmax(inputs, attrs):
    x = inputs[0]
    axis = attrs.get("axis", {"value": -1})["value"]
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def op_reshape(inputs, attrs):
    x, shp = inputs
    s = shp.astype(np.int64).tolist()
    # 0 means keep dim; -1 means infer
    out = []
    for i, d in enumerate(s):
        if d == 0:
            out.append(x.shape[i])
        else:
            out.append(d)
    return x.reshape(out)


def op_transpose(inputs, attrs):
    perm = attrs.get("perm", {"value": list(range(inputs[0].ndim))[::-1]})["value"]
    return inputs[0].transpose(perm).copy()


def op_slice(inputs, attrs):
    x = inputs[0]
    starts = inputs[1].astype(np.int64).tolist()
    ends = inputs[2].astype(np.int64).tolist()
    axes = inputs[3].astype(np.int64).tolist() if len(inputs) > 3 else list(range(len(starts)))
    steps = inputs[4].astype(np.int64).tolist() if len(inputs) > 4 else [1] * len(starts)
    sl = [slice(None)] * x.ndim
    for k, ax in enumerate(axes):
        if ax < 0: ax += x.ndim
        sl[ax] = slice(starts[k], ends[k], steps[k])
    return x[tuple(sl)].copy()


def op_concat(inputs, attrs):
    axis = attrs.get("axis", {"value": 0})["value"]
    return np.concatenate(inputs, axis=axis)


OPS = {
    "DequantizeLinear":      op_dequantize_linear,
    "QuantizeLinear":        op_quantize_linear,
    "Add":  lambda i, a: op_binary(i, a, np.add),
    "Sub":  lambda i, a: op_binary(i, a, np.subtract),
    "Mul":  lambda i, a: op_binary(i, a, np.multiply),
    "Div":  lambda i, a: op_binary(i, a, np.divide),
    "Max":  lambda i, a: op_binary(i, a, np.maximum),
    "Min":  lambda i, a: op_binary(i, a, np.minimum),
    "Pow":  lambda i, a: op_binary(i, a, np.power),
    "Sqrt": lambda i, a: np.sqrt(i[0]).astype(np.float32),
    "Sigmoid": lambda i, a: (1.0 / (1.0 + np.exp(-i[0]))).astype(np.float32),
    "Tanh":    lambda i, a: np.tanh(i[0]).astype(np.float32),
    "Sin":     lambda i, a: np.sin(i[0]).astype(np.float32),
    "Cos":     lambda i, a: np.cos(i[0]).astype(np.float32),
    "Clip": lambda i, a: np.clip(i[0],
                                 i[1].item() if len(i) > 1 and i[1] is not None else -np.inf,
                                 i[2].item() if len(i) > 2 and i[2] is not None else  np.inf
                                ).astype(np.float32),
    "Conv":     op_conv,
    "Gemm":     op_gemm,
    "MatMul":   op_matmul,
    "InstanceNormalization": op_instance_norm,
    "ReduceMean": op_reduce_mean,
    "Softmax":  op_softmax,
    "Reshape":  op_reshape,
    "Transpose": op_transpose,
    "Slice":    op_slice,
    "Concat":   op_concat,
}


def sanitize(name):
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def walk(asset_dir, inputs, verify_dir=None, abs_tol=1e-4, stop_on_fail=True,
         max_nodes=None):
    """Walk the manifest, run each op via numpy. Optionally diff outputs."""
    m = load_manifest(os.path.join(asset_dir, "manifest.json"))
    tensors = {}

    # initializers
    for ii in m["initializers"]:
        tensors[ii["name"]] = load_init(asset_dir, ii)

    # inputs
    for name, arr in inputs.items():
        tensors[name] = arr

    fails = 0
    n_nodes = len(m["nodes"])
    limit = n_nodes if max_nodes is None else min(max_nodes, n_nodes)
    for idx, node in enumerate(m["nodes"][:limit]):
        op = node["op_type"]
        in_arrs = []
        for in_name in node["inputs"]:
            if in_name == "":
                in_arrs.append(None)
                continue
            if in_name not in tensors:
                print(f"[#{idx} {op}] MISSING input '{in_name}' — fatal")
                return fails + 1
            in_arrs.append(tensors[in_name])
        if op not in OPS:
            print(f"[#{idx} {op}] no handler — abort")
            return fails + 1
        try:
            res = OPS[op](in_arrs, node["attrs"])
        except Exception as e:
            print(f"[#{idx} {op}] '{node['name']}' EXCEPTION: {e}")
            fails += 1
            if stop_on_fail:
                return fails
            res = np.zeros((1,), dtype=np.float32)
        outname = node["outputs"][0]
        if isinstance(res, tuple):  # multi-output ops would go here
            for o, r in zip(node["outputs"], res):
                tensors[o] = r
        else:
            tensors[outname] = res

        if verify_dir is not None:
            gp = Path(verify_dir) / (sanitize(outname) + ".bin")
            if gp.exists() and isinstance(res, np.ndarray) and res.dtype == np.float32:
                gold = np.fromfile(str(gp), dtype=np.float32).reshape(res.shape)
                diff = np.abs(res.astype(np.float32) - gold)
                mx = float(diff.max())
                mn = float(diff.mean())
                tag = "FAIL" if mx > abs_tol else "ok  "
                inputs_str = ", ".join(node["inputs"][:3])
                print(f"  #{idx:4d} {op:22s} out={outname[:50]:50s} max_abs={mx:.3e} mean={mn:.3e} {tag}")
                if mx > abs_tol:
                    fails += 1
                    if stop_on_fail:
                        print(f"    inputs: {inputs_str}")
                        print(f"    golden head: {gold.ravel()[:6]}")
                        print(f"    got    head: {res.ravel()[:6]}")
                        return fails
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["drift", "generator"])
    ap.add_argument("--assets", default="opencl/runtime/assets")
    ap.add_argument("--embedding", default="horse_embedding.bin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--save-image", type=str, default=None,
                    help="run end-to-end and save final image (PNG)")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    np.random.seed(args.seed)
    z = np.random.randn(1, 8, 12, 12).astype(np.float32)
    emb = np.fromfile(args.embedding, dtype=np.float32).reshape(1, 512)
    cfg = np.array([args.cfg], dtype=np.float32)

    if args.model == "drift":
        asset_dir = os.path.join(args.assets, "drift")
        inputs = dict(z=z,
                      t=np.array([[0.0]], dtype=np.float32),
                      text_embedding=emb,
                      cfg_scale=cfg)
        verify = None if args.no_verify else os.path.join(asset_dir, "golden", "step0")
    else:
        asset_dir = os.path.join(args.assets, "generator")
        # Need final z (same as the export script's "horse" set used).
        # Reload by running ORT through drift quickly.
        import onnxruntime as ort
        d = ort.InferenceSession("enhanced_label_sb/onnx/drift_static_int8.onnx",
                                 providers=["CPUExecutionProvider"])
        zi = z.copy()
        steps = 20; dt = 1.0 / steps
        for i in range(steps):
            t = np.array([[i/steps]], dtype=np.float32)
            zi = zi + d.run(None, {"z": zi, "t": t,
                                    "text_embedding": emb,
                                    "cfg_scale": cfg})[0] * dt
        inputs = dict(z=zi, text_embedding=emb)
        verify = None if args.no_verify else os.path.join(asset_dir, "golden", "horse")

    if args.save_image:
        # Run drift `steps` times via the WALKER, then run generator via walker,
        # save final RGB image. This validates the walker end-to-end vs ORT.
        from PIL import Image
        drift_dir = os.path.join(args.assets, "drift")
        gen_dir   = os.path.join(args.assets, "generator")
        zi = z.copy()
        dt = 1.0 / args.steps
        for i in range(args.steps):
            ti = np.array([[i / args.steps]], dtype=np.float32)
            tensors = walk_to_output(drift_dir,
                                     dict(z=zi, t=ti, text_embedding=emb, cfg_scale=cfg),
                                     "drift")
            zi = zi + tensors * dt
            print(f"  drift step {i+1}/{args.steps}  z stats mean={zi.mean():.3f} std={zi.std():.3f}")
        img = walk_to_output(gen_dir, dict(z=zi, text_embedding=emb), "reconstruction")
        rgb = ((img[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(args.save_image)
        print(f"saved {args.save_image}  shape={rgb.shape} std={rgb.std():.1f}")
        sys.exit(0)

    print(f"Walking {args.model} graph (verify={verify is not None})...")
    fails = walk(asset_dir, inputs, verify_dir=verify,
                 stop_on_fail=not args.cont, max_nodes=args.max_nodes)
    print(f"\nTotal failures: {fails}")
    sys.exit(0 if fails == 0 else 1)


def walk_to_output(asset_dir, inputs, out_name):
    """Walk graph and return the named output tensor (no verify)."""
    m = load_manifest(os.path.join(asset_dir, "manifest.json"))
    tensors = {}
    for ii in m["initializers"]:
        tensors[ii["name"]] = load_init(asset_dir, ii)
    for name, arr in inputs.items():
        tensors[name] = arr
    for node in m["nodes"]:
        in_arrs = [tensors[n] if n else None for n in node["inputs"]]
        if node["op_type"] not in OPS:
            print(f"WARN: missing handler {node['op_type']}")
            return np.zeros((1,), dtype=np.float32)
        try:
            res = OPS[node["op_type"]](in_arrs, node["attrs"])
        except Exception as e:
            print(f"ERR node {node['index']} {node['op_type']}: {e}")
            res = np.zeros((1,), dtype=np.float32)
        tensors[node["outputs"][0]] = res
    return tensors[out_name]


if __name__ == "__main__":
    main()
