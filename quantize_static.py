import onnx
import onnx.numpy_helper
import os
import numpy as np
import torch
import shutil
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, CalibrationMethod, quant_pre_process
import config
import data_management as dm

class SBCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, input_names, num_samples=32, batch_size=1):
        super().__init__()
        self.model_path = model_path
        self.input_names = input_names
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.current_sample = 0
        
        # Pre-generate or load calibration data
        self.data = []
        
        # Try to get some real labels and text bytes if possible
        try:
            # We don't need the full loader, just some samples
            labels = np.random.randint(0, 10, size=(num_samples,))
            text_bytes = []
            for l in labels:
                desc = dm.CLASS_DESCRIPTIONS[l] if l < 10 else f"class_{l}"
                text_bytes.append(dm.text_to_bytes(desc))
            text_bytes = np.array(text_bytes, dtype=np.int64)
            labels = labels.astype(np.int64)
        except:
            labels = np.random.randint(0, 10, size=(num_samples,)).astype(np.int64)
            text_bytes = np.zeros((num_samples, config.MAX_TEXT_BYTES), dtype=np.int64)

        for i in range(0, num_samples, batch_size):
            batch_labels = labels[i:i+batch_size]
            batch_text = text_bytes[i:i+batch_size]
            batch_source = np.zeros((batch_size,), dtype=np.int64)

            # z depends on the model
            z = np.random.randn(batch_size, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W).astype(np.float32)

            # Pre-compute normalized text embeddings for headless models
            text_emb = np.random.randn(batch_size, config.TEXT_EMBEDDING_DIM).astype(np.float32)
            norms = np.linalg.norm(text_emb, axis=-1, keepdims=True)
            text_emb = text_emb / (norms + 1e-8)

            sample = {}
            if 'z' in input_names: sample['z'] = z
            if 'label' in input_names: sample['label'] = batch_labels
            if 'text_bytes' in input_names: sample['text_bytes'] = batch_text
            if 'text_embedding' in input_names: sample['text_embedding'] = text_emb
            if 'source_id' in input_names: sample['source_id'] = batch_source
            
            # Apply the same scaling as in training/inference if it's the drift model
            if "drift" in model_path:
                if 'z' in sample:
                    sample['z'] = sample['z'] * config.CST_COEF_GAUSSIAN_PRIO
                if 't' in input_names:
                    sample['t'] = np.random.rand(batch_size, 1).astype(np.float32)
                if 'cfg_scale' in input_names:
                    sample['cfg_scale'] = np.array([config.CFG_SCALE], dtype=np.float32)
            
            self.data.append(sample)

    def get_next(self):
        if self.current_sample >= len(self.data):
            return None
        
        batch = self.data[self.current_sample]
        self.current_sample += 1
        return batch

def fix_batch_size(model_path, output_path, batch_size=1):
    """
    Sets dynamic batch_size to a fixed value to avoid shape inference errors.
    """
    print(f"Fixing batch_size for {model_path}...")
    model = onnx.load(model_path)
    
    # Clear value_info to allow re-inference with new shapes
    while(len(model.graph.value_info) > 0):
        model.graph.value_info.pop()
        
    for input in model.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param == 'batch_size':
                dim.dim_value = batch_size
    onnx.save(model, output_path)
    return output_path

def strip_nonstandard_opsets(model):
    """Remove opset imports not supported by onnxruntime-web WASM (e.g. com.microsoft, org.pytorch.aten)."""
    web_safe_domains = {'', 'ai.onnx'}
    kept = [o for o in model.opset_import if o.domain in web_safe_domains]
    del model.opset_import[:]
    model.opset_import.extend(kept)
    return model


def split_shared_qdq_outputs(model):
    """
    Enforce strict 1-consumer-per-output for QuantizeLinear and DequantizeLinear.

    onnxruntime's RemoveNode / CanRemoveNodeAndMergeEdges optimizer logic
    assumes each Q/DQ output has exactly one consumer.  In the CFG drift model,
    shared weights and activations fan-out through the same Q or DQ node into
    both conditioned and unconditioned branches, causing ORT 1.20 to raise:
      "Should be unreachable if CanRemoveNodeAndMergeEdges is in sync…"
    Fix: for each extra consumer beyond the first, clone the Q/DQ node with a
    fresh output name so every consumer has its own dedicated Q/DQ node.
    """
    from collections import defaultdict

    def build_consumers(graph):
        c = defaultdict(list)
        for node in graph.node:
            for idx, inp in enumerate(node.input):
                if inp:
                    c[inp].append((node, idx))
        return c

    split_count = 0

    for op_type in ('QuantizeLinear', 'DequantizeLinear'):
        consumers = build_consumers(model.graph)
        new_nodes = []
        for node in list(model.graph.node):
            if node.op_type != op_type:
                continue
            out = node.output[0]
            consuming = consumers.get(out, [])
            if len(consuming) <= 1:
                continue
            for consumer_node, input_idx in consuming[1:]:
                split_count += 1
                new_out = f"{out}_split_{split_count}"
                new_node = onnx.helper.make_node(
                    op_type,
                    inputs=list(node.input),
                    outputs=[new_out],
                    name=f"{node.name}_split_{split_count}" if node.name else f"{op_type}_split_{split_count}",
                )
                for attr in node.attribute:
                    new_node.attribute.append(attr)
                new_nodes.append(new_node)
                consumer_node.input[input_idx] = new_out
        model.graph.node.extend(new_nodes)
        if new_nodes:
            print(f"  Split {len(new_nodes)} shared {op_type} consumer edges.")

    print(f"  Total splits: {split_count} (+{split_count} new Q/DQ nodes).")
    return model


def fold_int32_dequantize_nodes(model):
    """
    Replace constant INT32 DequantizeLinear nodes with plain float32 initializers.

    onnxruntime-web WASM has no runtime kernel for DequantizeLinear with INT32 input.
    Desktop ORT constant-folds these at session init; WASM does not.  Bias nodes for
    Conv/Gemm and scale/bias nodes for InstanceNorm/LayerNorm all land here.
    """
    init_map = {i.name: i for i in model.graph.initializer}
    nodes_to_remove = []
    new_initializers = []

    for node in model.graph.node:
        if node.op_type != 'DequantizeLinear' or len(node.input) < 3:
            continue
        zp_name = node.input[2]
        x_name  = node.input[0]
        scale_name = node.input[1]
        if zp_name not in init_map or init_map[zp_name].data_type != onnx.TensorProto.INT32:
            continue
        if x_name not in init_map or scale_name not in init_map:
            continue  # dynamic tensor — leave as-is

        x_arr     = onnx.numpy_helper.to_array(init_map[x_name]).astype(np.int32)
        scale_arr = onnx.numpy_helper.to_array(init_map[scale_name]).astype(np.float32)
        zp_arr    = onnx.numpy_helper.to_array(init_map[zp_name]).astype(np.int32)

        dequant = (x_arr.astype(np.float64) - zp_arr.astype(np.float64)) * scale_arr.astype(np.float64)
        dequant_f32 = dequant.astype(np.float32)

        out_name = node.output[0]
        new_init = onnx.numpy_helper.from_array(dequant_f32, name=out_name)
        new_initializers.append(new_init)
        nodes_to_remove.append(node)

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    model.graph.initializer.extend(new_initializers)

    # Remove stale initializers that were only used as DQ inputs
    used_names = set()
    for node in model.graph.node:
        used_names.update(node.input)
    for graph_input in model.graph.input:
        used_names.add(graph_input.name)

    to_remove = [i for i in model.graph.initializer if i.name not in used_names]
    for i in to_remove:
        model.graph.initializer.remove(i)

    print(f"  Folded {len(nodes_to_remove)} INT32 DequantizeLinear nodes; "
          f"removed {len(to_remove)} stale initializers.")
    return model


def topological_sort(model):
    """
    Ensures that the nodes in the ONNX model are in topological order.
    Required after splitting QDQ nodes because new nodes are appended to the end.
    """
    from collections import deque, defaultdict
    
    graph = model.graph
    # Map from tensor name to the node that produces it
    producers = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node
            
    # Initializers and inputs are also "producers" (of themselves)
    known_tensors = set(i.name for i in graph.initializer) | set(i.name for i in graph.input)
    
    sorted_nodes = []
    nodes_to_sort = list(graph.node)
    
    # Simple greedy topological sort
    while nodes_to_sort:
        ready_node_idx = -1
        for i, node in enumerate(nodes_to_sort):
            # A node is ready if all its inputs are already known
            if all(not inp or inp in known_tensors for inp in node.input):
                ready_node_idx = i
                break
        
        if ready_node_idx == -1:
            # Cycle or missing input? In our case, likely just need to handle it.
            # If we can't find a ready node, just take the first one and hope for the best,
            # but this shouldn't happen in a DAG.
            print(f"  Warning: Topological sort stalled at node {nodes_to_sort[0].name}. Forcing...")
            ready_node_idx = 0
            
        node = nodes_to_sort.pop(ready_node_idx)
        sorted_nodes.append(node)
        for output in node.output:
            known_tensors.add(output)
            
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return model


def fix_dcr_mode(model):
    """
    ST Edge AI / atonn only supports DCR mode for DepthToSpace nodes.

    PyTorch's F.pixel_shuffle traces to DepthToSpace(mode=CRD).  CRD and DCR
    differ in channel ordering:
      CRD: out[n,c,h*r+a,w*r+b] = in[n, c*r*r + a*r + b, h, w]
      DCR: out[n,c,h*r+a,w*r+b] = in[n, a*r*C + b*C + c,   h, w]

    To convert without changing model semantics we permute the OUTPUT CHANNELS
    of the Conv immediately upstream of DepthToSpace from CRD order to DCR
    order, then change the DepthToSpace mode attribute to DCR.  The permutation
    is: dcr_perm[a*r*C + b*C + c] = c*r*r + a*r + b
    """
    init_map = {i.name: i for i in model.graph.initializer}
    # Map output tensor → producing node (for walking upstream)
    output_to_node = {}
    for n in model.graph.node:
        for out in n.output:
            output_to_node[out] = n

    count = 0
    for node in model.graph.node:
        if node.op_type != 'DepthToSpace':
            continue
        mode = 'CRD'
        blocksize = 2
        for attr in node.attribute:
            if attr.name == 'mode':
                mode = attr.s.decode('utf-8')
            if attr.name == 'blocksize':
                blocksize = attr.i
        if mode != 'CRD':
            continue

        r = blocksize
        print(f"  Converting {node.name} (blocksize={r}) CRD→DCR with weight permutation…")

        # Walk upstream through DequantizeLinear nodes to find the Conv
        upstream_name = node.input[0]
        upstream = output_to_node.get(upstream_name)
        while upstream is not None and upstream.op_type in ('DequantizeLinear', 'QuantizeLinear'):
            upstream_name = upstream.input[0]
            upstream = output_to_node.get(upstream_name)

        if upstream is None or upstream.op_type != 'Conv':
            print(f"    WARNING: could not find upstream Conv for {node.name}, skipping permutation.")
            for attr in node.attribute:
                if attr.name == 'mode':
                    attr.s = b'DCR'
            continue

        # Determine C (output channels / r^2).
        # In QDQ models the Conv weight goes: INT8_init → DequantizeLinear → Conv,
        # so upstream.input[1] is a DQ output, not a direct initializer.
        weight_ref = upstream.input[1]
        if weight_ref not in init_map:
            # Walk through DequantizeLinear
            dq = output_to_node.get(weight_ref)
            if dq is not None and dq.op_type == 'DequantizeLinear':
                weight_ref = dq.input[0]
        if weight_ref not in init_map:
            print(f"    WARNING: Conv weight not in initializers for {node.name}, skipping.")
            for attr in node.attribute:
                if attr.name == 'mode':
                    attr.s = b'DCR'
            continue

        weight_init = init_map[weight_ref]
        w_arr = onnx.numpy_helper.to_array(weight_init)
        N = w_arr.shape[0]          # total output channels = C * r * r
        C = N // (r * r)

        # Build CRD→DCR permutation
        # dcr_channel for pixel (c,a,b): a*r*C + b*C + c
        # crd_channel for pixel (c,a,b): c*r*r + a*r + b
        # We want: new_weight_row[dcr_idx] = old_weight_row[crd_idx]
        perm = np.zeros(N, dtype=np.int64)
        for a in range(r):
            for b in range(r):
                for c in range(C):
                    crd_idx = c * r * r + a * r + b
                    dcr_idx = a * r * C + b * C + c
                    perm[dcr_idx] = crd_idx

        # Apply permutation to Conv weight rows (INT8 or float32)
        new_w = w_arr[perm]
        new_init = onnx.numpy_helper.from_array(new_w, name=weight_ref)
        weight_init.CopyFrom(new_init)

        # Apply permutation to bias (if present; may be direct or via DQ)
        if len(upstream.input) > 2 and upstream.input[2]:
            bias_ref = upstream.input[2]
            if bias_ref not in init_map:
                dq = output_to_node.get(bias_ref)
                if dq is not None and dq.op_type == 'DequantizeLinear':
                    bias_ref = dq.input[0]
            if bias_ref in init_map:
                b_arr = onnx.numpy_helper.to_array(init_map[bias_ref])
                if b_arr.ndim == 1 and b_arr.shape[0] == N:
                    new_b = b_arr[perm]
                    new_bias = onnx.numpy_helper.from_array(new_b, name=bias_ref)
                    init_map[bias_ref].CopyFrom(new_bias)

        # Permute per-channel Q/DQ scale and zero_point for the Conv output
        # and for the DequantizeLinear that feeds the weight.
        conv_out = upstream.output[0]
        for qnode in model.graph.node:
            if qnode.op_type not in ('QuantizeLinear', 'DequantizeLinear'):
                continue
            # Q/DQ attached to Conv output (output quantization)
            if qnode.input[0] == conv_out:
                for slot in (1, 2):
                    if len(qnode.input) > slot and qnode.input[slot] in init_map:
                        arr = onnx.numpy_helper.to_array(init_map[qnode.input[slot]])
                        if arr.ndim == 1 and arr.shape[0] == N:
                            new_arr = arr[perm]
                            new_i = onnx.numpy_helper.from_array(new_arr, name=qnode.input[slot])
                            init_map[qnode.input[slot]].CopyFrom(new_i)
            # DQ attached to the weight initializer (weight dequantization)
            if (qnode.op_type == 'DequantizeLinear' and
                    len(qnode.input) > 0 and qnode.input[0] == weight_ref):
                for slot in (1, 2):
                    if len(qnode.input) > slot and qnode.input[slot] in init_map:
                        arr = onnx.numpy_helper.to_array(init_map[qnode.input[slot]])
                        if arr.ndim == 1 and arr.shape[0] == N:
                            new_arr = arr[perm]
                            new_i = onnx.numpy_helper.from_array(new_arr, name=qnode.input[slot])
                            init_map[qnode.input[slot]].CopyFrom(new_i)

        # Finally flip the mode attribute
        for attr in node.attribute:
            if attr.name == 'mode':
                attr.s = b'DCR'
        count += 1

    print(f"  Converted {count} DepthToSpace node(s) CRD→DCR with weight permutation.")
    return model

def replace_depthtospace_with_reshape(model):
    """
    Replace DepthToSpace nodes with a 6D Reshape+Transpose+Reshape decomposition.

    stedgeai 4.0 / atonn does not support the DepthToSpace op despite listing it in
    supported-ops.  The 6D Reshape+Transpose+Reshape sequence is semantically equivalent
    and IS accepted by atonn (falls back to a SW epoch).

    DCR mode:  input [N, r²C, H, W] → Reshape[N,r,r,C,H,W] → Transpose[0,3,4,1,5,2]
                                     → Reshape[N,C,Hr,Wr]
    CRD mode:  input [N, Cr², H, W] → Reshape[N,C,r,r,H,W] → Transpose[0,1,4,2,5,3]
                                     → Reshape[N,C,Hr,Wr]
    """
    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.tensor_type.HasField('shape'):
            shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    nodes_to_remove, new_nodes, new_inits = [], [], []
    count = 0

    for node in model.graph.node:
        if node.op_type != 'DepthToSpace':
            continue
        r, mode = 2, 'DCR'
        for a in node.attribute:
            if a.name == 'blocksize': r = a.i
            if a.name == 'mode': mode = a.s.decode()

        inp, out = node.input[0], node.output[0]
        in_shape = shape_map.get(inp)
        if in_shape is None:
            print(f"  WARNING: no shape for {inp}, skipping {node.name}")
            continue

        N, rrC, H, W = in_shape
        C = rrC // (r * r)
        prefix = (node.name or f'dts_{count}').replace('/', '_')

        sh1_name, sh2_name = f'{prefix}_sh1', f'{prefix}_sh2'
        r1_name, t1_name  = f'{prefix}_r1', f'{prefix}_t1'

        if mode == 'DCR':
            shape1, perm = [N, r, r, C, H, W], [0, 3, 4, 1, 5, 2]
        else:
            shape1, perm = [N, C, r, r, H, W], [0, 1, 4, 2, 5, 3]
        shape2 = [N, C, H * r, W * r]

        new_inits += [
            onnx.numpy_helper.from_array(np.array(shape1, dtype=np.int64), name=sh1_name),
            onnx.numpy_helper.from_array(np.array(shape2, dtype=np.int64), name=sh2_name),
        ]
        new_nodes += [
            onnx.helper.make_node('Reshape',   [inp, sh1_name], [r1_name], name=f'{prefix}_reshape1'),
            onnx.helper.make_node('Transpose', [r1_name],        [t1_name], perm=perm, name=f'{prefix}_transpose'),
            onnx.helper.make_node('Reshape',   [t1_name, sh2_name], [out], name=f'{prefix}_reshape2'),
        ]
        nodes_to_remove.append(node)
        count += 1
        print(f'  Replaced {node.name} ({mode}, r={r}): {in_shape} -> {shape2}')

    for n in nodes_to_remove:
        model.graph.node.remove(n)
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    print(f'  Replaced {count} DepthToSpace node(s) with Reshape+Transpose+Reshape.')
    return model


def simplify_onnx(model_path, output_path):
    """
    Simplifies the ONNX model using onnx-simplifier (constant folding, shape inference, etc.)
    """
    import onnx
    from onnxsim import simplify
    print(f"Simplifying {model_path}...")
    model = onnx.load(model_path)
    model_simp, check = simplify(model)
    if not check:
        print("Warning: Simplified model check failed!")
    onnx.save(model_simp, output_path)
    return output_path

def convert_int64_to_int32(model):
    """
    Converts all INT64 inputs and initializers to INT32.
    """
    import onnx
    import numpy as np
    print("Converting INT64 to INT32 for hardware compatibility...")
    
    # 1. Convert INT64 inputs to INT32
    for input in model.graph.input:
        if input.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            input.type.tensor_type.elem_type = onnx.TensorProto.INT32
            
    # 2. Convert INT64 initializers to INT32
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr = onnx.numpy_helper.to_array(init)
            new_init = onnx.numpy_helper.from_array(arr.astype(np.int32), name=init.name)
            init.CopyFrom(new_init)
            
    # 3. Convert INT64 value_info to INT32
    for vi in model.graph.value_info:
        if vi.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            vi.type.tensor_type.elem_type = onnx.TensorProto.INT32
            
    return model

def quantize_model_static(model_path, output_path):
    """
    Quantizes an ONNX model to INT8 (static).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Simplify and fix batch size before quantization
    temp_simp = model_path.replace(".onnx", "_temp_simp.onnx")
    simplify_onnx(model_path, temp_simp)
    
    fixed_path = temp_simp.replace("_temp_simp.onnx", "_fixed.onnx")
    fix_batch_size(temp_simp, fixed_path, batch_size=1)
    
    # Simplify AGAIN after fixing batch size to constant-fold shape-dependent nodes
    simplify_onnx(fixed_path, fixed_path)

    print(f"Preparing calibration data for {fixed_path}...")
    
    # Identify input names
    model = onnx.load(fixed_path)
    input_names = [input.name for input in model.graph.input]
    
    dr = SBCalibrationDataReader(model_path, input_names, num_samples=64, batch_size=1)

    print(f"Quantizing {fixed_path} to INT8 (static)...")
    quantize_static(
        model_input=fixed_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=1,  # QDQ: QuantizeLinear/DequantizeLinear — supported by onnxruntime-web WASM
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, # Signed Int8 is often better for weights
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'EnableShapeInference': True,
            'ActivationSymmetric': True,
            'WeightSymmetric': True,
        }
    )
    print(f"Saved static INT8 model to {output_path}")

    # Post-process for onnxruntime-web WASM and STM32 compatibility:
    print("Post-processing for compatibility...")
    model = onnx.load(output_path)
    model = strip_nonstandard_opsets(model)
    model = fold_int32_dequantize_nodes(model)
    model = split_shared_qdq_outputs(model)
    model = topological_sort(model)
    # Note: INT64→INT32 conversion is intentionally skipped. All data inputs are now
    # float32 (text_embedding replaces text_bytes; source_id is folded to constant).
    # The only remaining INT64 values are ONNX-required shape tensors for Reshape nodes,
    # which must remain INT64 per the ONNX spec — converting them breaks shape inference.

    onnx.save(model, output_path)

    # Final simplification pass to clean up any messy Q/DQ splits
    simplify_onnx(output_path, output_path)

    # Populate value_info with explicit type/shape annotations for all intermediate
    # tensors (including unquantized float32 LayerNorm ops).  stedgeai's shape
    # inference engine fails when these are absent.
    from onnx import shape_inference as _si
    model = onnx.load(output_path)
    model = _si.infer_shapes(model)
    onnx.save(model, output_path)

    # fix_dcr_mode MUST run last — onnxsim reverts DepthToSpace attributes.
    # It also permutes the preceding Conv output channels so CRD→DCR is lossless.
    model = onnx.load(output_path)
    model = fix_dcr_mode(model)
    onnx.save(model, output_path)

    # Replace DepthToSpace with 6D Reshape+Transpose+Reshape.
    # stedgeai 4.0 / atonn does not support the DepthToSpace op (despite listing it in
    # supported-ops). The 6D Reshape+Transpose+Reshape decomposition is semantically
    # equivalent and IS accepted by atonn as a SW epoch. fix_dcr_mode must run first
    # so the correct DCR decomposition is used.
    model = onnx.load(output_path)
    model = replace_depthtospace_with_reshape(model)
    model = topological_sort(model)
    from onnx import shape_inference as _si2
    model = _si2.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Compatibility-optimized model saved to {output_path}")

    # Cleanup
    for p in [temp_simp, fixed_path]:
        if os.path.exists(p):
            os.remove(p)



if __name__ == "__main__":
    base_dir = "enhanced_label_sb/onnx"
    models = ["drift.onnx", "generator.onnx"]

    for m in models:
        src = os.path.join(base_dir, m)
        dst_int8_static = os.path.join(base_dir, m.replace(".onnx", "_static_int8.onnx"))
        try:
            quantize_model_static(src, dst_int8_static)
        except Exception as e:
            print(f"Failed to quantize {m} statically: {e}")

    print("\nStatic Quantization complete.")
