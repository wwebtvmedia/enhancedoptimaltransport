"""
Export ONNX model assets for the C++ OpenCL runtime.

For each ONNX model produces:
  <out>/<model>/manifest.json     - graph topology (nodes, IOs, attrs, init shapes/dtypes)
  <out>/<model>/init/<name>.bin   - one binary file per initializer (raw little-endian)

Plus a per-op golden tensor dump using a modified ONNX where every intermediate
value is exposed as an output. The golden dump goes to:
  <out>/<model>/golden/<input_set>/<tensor_name>.bin
  <out>/<model>/golden/<input_set>/manifest.json   - shapes/dtypes per tensor

The C++ runtime uses init/ to populate weights and golden/ to verify each op output.
"""

import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


# Map ONNX dtype enum -> human + element size (bytes)
DTYPE_INFO = {
    TensorProto.FLOAT:   ("float32",  4),
    TensorProto.UINT8:   ("uint8",    1),
    TensorProto.INT8:    ("int8",     1),
    TensorProto.UINT16:  ("uint16",   2),
    TensorProto.INT16:   ("int16",    2),
    TensorProto.INT32:   ("int32",    4),
    TensorProto.INT64:   ("int64",    8),
    TensorProto.BOOL:    ("bool",     1),
    TensorProto.FLOAT16: ("float16",  2),
    TensorProto.DOUBLE:  ("float64",  8),
}


def safe_filename(name: str) -> str:
    """Make a tensor/node name safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def attr_to_jsonable(attr):
    """Turn an AttributeProto into a JSON-serializable dict."""
    t = attr.type
    if t == onnx.AttributeProto.INT:
        return {"type": "int", "value": attr.i}
    if t == onnx.AttributeProto.FLOAT:
        return {"type": "float", "value": float(attr.f)}
    if t == onnx.AttributeProto.STRING:
        return {"type": "string", "value": attr.s.decode("utf-8", errors="replace")}
    if t == onnx.AttributeProto.INTS:
        return {"type": "ints", "value": list(attr.ints)}
    if t == onnx.AttributeProto.FLOATS:
        return {"type": "floats", "value": [float(x) for x in attr.floats]}
    if t == onnx.AttributeProto.STRINGS:
        return {"type": "strings", "value": [s.decode("utf-8", errors="replace") for s in attr.strings]}
    if t == onnx.AttributeProto.TENSOR:
        arr = numpy_helper.to_array(attr.t)
        return {"type": "tensor", "shape": list(arr.shape), "dtype": str(arr.dtype),
                "value": arr.flatten().tolist()}
    return {"type": "unknown_attr_type", "value": None}


def dim_to_int(d):
    """Convert an onnx Dim to int or 0 if symbolic."""
    if d.HasField("dim_value"):
        return int(d.dim_value)
    return 0  # symbolic / unknown


def value_info_to_json(vi):
    tt = vi.type.tensor_type
    return {
        "name": vi.name,
        "dtype_enum": int(tt.elem_type),
        "dtype": DTYPE_INFO.get(tt.elem_type, (f"enum_{tt.elem_type}", 0))[0],
        "shape": [dim_to_int(d) for d in tt.shape.dim],
    }


def export_initializers(model: onnx.ModelProto, out_dir: Path):
    """Dump every initializer as a raw .bin (little-endian, native dtype)."""
    init_dir = out_dir / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for tp in model.graph.initializer:
        arr = numpy_helper.to_array(tp)
        # Ensure little-endian and contiguous
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        arr = np.ascontiguousarray(arr)
        safe = safe_filename(tp.name) + ".bin"
        (init_dir / safe).write_bytes(arr.tobytes())
        dt_name, elem_size = DTYPE_INFO.get(tp.data_type, (str(arr.dtype), arr.dtype.itemsize))
        entries.append({
            "name": tp.name,
            "file": f"init/{safe}",
            "dtype": dt_name,
            "dtype_enum": int(tp.data_type),
            "shape": list(arr.shape),
            "byte_size": int(arr.nbytes),
            "elem_size": int(elem_size),
        })
    return entries


def export_manifest(model: onnx.ModelProto, init_entries, out_dir: Path):
    graph = model.graph
    nodes = []
    for i, n in enumerate(graph.node):
        nodes.append({
            "index": i,
            "name": n.name or f"{n.op_type}_{i}",
            "op_type": n.op_type,
            "domain": n.domain or "",
            "inputs": list(n.input),
            "outputs": list(n.output),
            "attrs": {a.name: attr_to_jsonable(a) for a in n.attribute},
        })
    manifest = {
        "model_path": "(see source)",
        "ir_version": int(model.ir_version),
        "producer": f"{model.producer_name} {model.producer_version}",
        "opset": [{"domain": o.domain, "version": int(o.version)} for o in model.opset_import],
        "inputs": [value_info_to_json(v) for v in graph.input],
        "outputs": [value_info_to_json(v) for v in graph.output],
        "value_info": [value_info_to_json(v) for v in graph.value_info],
        "initializers": init_entries,
        "nodes": nodes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def dump_golden(model_path: str, out_dir: Path, input_name: str, inputs: dict,
                tensor_name_filter=None):
    """
    Expose every intermediate tensor as a graph output, run ORT once, dump each
    tensor as raw .bin alongside a small JSON listing shape/dtype.

    inputs: { name: np.ndarray }
    input_name: subfolder under golden/ (e.g. 'horse_step0').
    """
    import onnxruntime as ort
    from onnx import shape_inference

    base = onnx.load(model_path)
    # Run shape inference so every intermediate has a known dtype/shape.
    inferred = shape_inference.infer_shapes(base, check_type=False, strict_mode=False)
    base = inferred

    # Build a name -> (dtype_enum, shape) lookup from value_info + initializers + inputs.
    type_map = {}
    for vi in list(base.graph.value_info) + list(base.graph.input) + list(base.graph.output):
        tt = vi.type.tensor_type
        type_map[vi.name] = (int(tt.elem_type), [dim_to_int(d) for d in tt.shape.dim])
    for tp in base.graph.initializer:
        type_map[tp.name] = (int(tp.data_type), list(tp.dims))

    # Build set of names that need exposing: every node output that isn't already an output.
    existing_outs = {o.name for o in base.graph.output}
    extra_outs = []
    for n in base.graph.node:
        for o in n.output:
            if o and o not in existing_outs and (tensor_name_filter is None or tensor_name_filter(o)):
                extra_outs.append(o)
                existing_outs.add(o)

    # Add each as a ValueInfo so ORT will return it. Fall back to FLOAT if shape-inference missed it.
    for name in extra_outs:
        dtype_enum, shape = type_map.get(name, (int(onnx.TensorProto.FLOAT), None))
        if dtype_enum == 0:  # UNDEFINED
            dtype_enum = int(onnx.TensorProto.FLOAT)
        vi = onnx.helper.make_tensor_value_info(name, dtype_enum, shape)
        base.graph.output.append(vi)

    # Save to a temp file so ORT can load it (some versions of ORT don't accept proto in-memory cleanly).
    tmp_path = out_dir / "_with_all_outputs.onnx"
    onnx.save(base, str(tmp_path), save_as_external_data=False)

    sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(output_names, inputs)

    dump_dir = out_dir / "golden" / input_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, arr in zip(output_names, outs):
        safe = safe_filename(name) + ".bin"
        arr = np.ascontiguousarray(arr)
        (dump_dir / safe).write_bytes(arr.tobytes())
        entries.append({
            "name": name,
            "file": safe,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "byte_size": int(arr.nbytes),
            "stats": {
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std":  float(arr.std()) if arr.size else 0.0,
            } if arr.dtype.kind in "fi" else None,
        })
    (dump_dir / "manifest.json").write_text(json.dumps({
        "model": str(model_path),
        "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
        "tensors": entries,
    }, indent=2))
    tmp_path.unlink()
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="opencl/runtime/assets",
                    help="Output dir (relative to repo root)")
    ap.add_argument("--drift", type=str, default="enhanced_label_sb/onnx/drift_static_int8.onnx")
    ap.add_argument("--generator", type=str, default="enhanced_label_sb/onnx/generator_static_int8.onnx")
    ap.add_argument("--embedding", type=str, default="horse_embedding.bin",
                    help="text embedding to use for golden dump (512-dim fp32)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--no-golden", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for model_path, name in [(args.drift, "drift"), (args.generator, "generator")]:
        print(f"\n=== {name} ===")
        model_out = out_root / name
        model = onnx.load(model_path)
        inits = export_initializers(model, model_out)
        manifest = export_manifest(model, inits, model_out)
        print(f"  initializers: {len(inits)}  nodes: {len(manifest['nodes'])}")
        print(f"  wrote {model_out}/manifest.json")

    if args.no_golden:
        return

    np.random.seed(args.seed)
    z = np.random.randn(1, 8, 12, 12).astype(np.float32)
    emb = np.fromfile(args.embedding, dtype=np.float32).reshape(1, -1)
    assert emb.shape[1] == 512, f"embedding must be 512-dim, got {emb.shape}"
    cfg = np.array([args.cfg], dtype=np.float32)

    print("\n=== golden: drift step 0 ===")
    dump_golden(args.drift, out_root / "drift", input_name="step0",
                inputs={"z": z, "t": np.array([[0.0]], dtype=np.float32),
                        "text_embedding": emb, "cfg_scale": cfg})

    print("=== golden: generator (after 20 drift steps) ===")
    import onnxruntime as ort
    drift_sess = ort.InferenceSession(args.drift, providers=["CPUExecutionProvider"])
    steps = 20
    dt = 1.0 / steps
    zi = z.copy()
    for i in range(steps):
        ti = np.array([[i / steps]], dtype=np.float32)
        d = drift_sess.run(None, {"z": zi, "t": ti, "text_embedding": emb, "cfg_scale": cfg})[0]
        zi = zi + d * dt
    # Save final latent for reproducibility
    (out_root / "drift" / "golden").mkdir(parents=True, exist_ok=True)
    (out_root / "drift" / "golden" / "final_latent.bin").write_bytes(zi.astype(np.float32).tobytes())
    print(f"  z final stats: mean={zi.mean():.4f} std={zi.std():.4f}")

    dump_golden(args.generator, out_root / "generator", input_name="horse",
                inputs={"z": zi, "text_embedding": emb})

    # Also dump the final pixel image for sanity check
    img = ort.InferenceSession(args.generator, providers=["CPUExecutionProvider"]).run(
        None, {"z": zi, "text_embedding": emb})[0]
    arr = ((img[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(arr).save(out_root / "reference_horse.png")
    print(f"  saved {out_root}/reference_horse.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
