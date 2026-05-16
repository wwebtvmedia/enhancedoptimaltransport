import onnx
import onnx.numpy_helper
import os
import numpy as np
import torch
import shutil
import json
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, CalibrationMethod, quant_pre_process
import config
import data_management as dm

class SBCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, input_names, num_samples=256, batch_size=1):
        super().__init__()
        self.model_path = model_path
        self.input_names = input_names
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.current_sample = 0
        
        # Load real embeddings if available
        self.label_embeddings = None
        embed_path = os.path.join(os.path.dirname(model_path), "label_embeddings.json")
        if os.path.exists(embed_path):
            try:
                with open(embed_path, 'r') as f:
                    self.label_embeddings = json.load(f)
                print(f"  [Calibration] Loaded {len(self.label_embeddings)} real embeddings.")
            except Exception as e:
                print(f"  [Calibration] Warning: Could not load label_embeddings.json: {e}")

        # Try to load pre-generated real calibration data
        self.data = []
        calib_dir = "calibration_data"
        is_drift = "drift" in model_path.lower()
        data_file = "drift_inputs.npy" if is_drift else "gen_inputs.npy"
        data_path = os.path.join(calib_dir, data_file)

        if os.path.exists(data_path):
            try:
                raw_data = np.load(data_path, allow_pickle=True)
                print(f"  [Calibration] Loaded {len(raw_data)} real samples from {data_file}")
                # Filter data to only include inputs present in the model
                for item in raw_data:
                    sample = {k: v for k, v in item.items() if k in input_names}
                    self.data.append(sample)
                # Shuffle to get a good mix
                np.random.shuffle(self.data)
                self.data = self.data[:num_samples]
            except Exception as e:
                print(f"  [Calibration] Error loading {data_file}: {e}. Falling back to random.")

        if not self.data:
            print(f"  [Calibration] Generating {num_samples} random samples...")
            for i in range(0, num_samples, batch_size):
                sample = {}
                # 1. Latent 'z': use standard normal scaled by prior coef
                z = np.random.randn(batch_size, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W).astype(np.float32)
                scale = 0.8 + 0.4 * (i / num_samples) 
                z = z * scale
                if 'z' in input_names: sample['z'] = z

                # 2. Labels and Text Embeddings
                batch_labels = np.random.randint(0, 10, size=(batch_size,))
                if 'label' in input_names: 
                    sample['label'] = batch_labels.astype(np.int64)
                
                if 'text_embedding' in input_names:
                    text_emb = np.zeros((batch_size, config.TEXT_EMBEDDING_DIM), dtype=np.float32)
                    for j in range(batch_size):
                        l_str = str(batch_labels[j])
                        if self.label_embeddings and l_str in self.label_embeddings:
                            text_emb[j] = np.array(self.label_embeddings[l_str], dtype=np.float32)
                        else:
                            temp = np.random.randn(config.TEXT_EMBEDDING_DIM).astype(np.float32)
                            text_emb[j] = temp / (np.linalg.norm(temp) + 1e-8)
                    sample['text_embedding'] = text_emb

                # 3. Time 't' and CFG
                if 't' in input_names:
                    t_val = (i / num_samples)
                    sample['t'] = np.full((batch_size, 1), t_val, dtype=np.float32)
                
                if 'cfg_scale' in input_names:
                    sample['cfg_scale'] = np.array([config.CFG_SCALE], dtype=np.float32)

                if 'source_id' in input_names:
                    sample['source_id'] = np.zeros((batch_size,), dtype=np.int64)

                self.data.append(sample)

    def get_next(self):
        if self.current_sample >= len(self.data):
            return None
        batch = self.data[self.current_sample]
        self.current_sample += 1
        return batch

def fix_batch_size(model_path, output_path, batch_size=1):
    """Sets dynamic batch_size to a fixed value to avoid shape inference errors."""
    print(f"Fixing batch_size for {model_path}...")
    model = onnx.load(model_path)
    while(len(model.graph.value_info) > 0):
        model.graph.value_info.pop()
    for input in model.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param == 'batch_size':
                dim.dim_value = batch_size
    onnx.save(model, output_path)
    return output_path

def strip_nonstandard_opsets(model):
    """Remove opset imports not supported by onnxruntime-web WASM."""
    web_safe_domains = {'', 'ai.onnx'}
    kept = [o for o in model.opset_import if o.domain in web_safe_domains]
    del model.opset_import[:]
    model.opset_import.extend(kept)
    return model

def split_shared_qdq_outputs(model):
    """Enforce strict 1-consumer-per-output for Q/DQ nodes to avoid ORT optimizer crashes."""
    from collections import defaultdict
    def build_consumers(graph):
        c = defaultdict(list)
        for node in graph.node:
            for idx, inp in enumerate(node.input):
                if inp: c[inp].append((node, idx))
        return c
    split_count = 0
    for op_type in ('QuantizeLinear', 'DequantizeLinear'):
        consumers = build_consumers(model.graph)
        new_nodes = []
        for node in list(model.graph.node):
            if node.op_type != op_type: continue
            out = node.output[0]
            consuming = consumers.get(out, [])
            if len(consuming) <= 1: continue
            for consumer_node, input_idx in consuming[1:]:
                split_count += 1
                new_out = f"{out}_split_{split_count}"
                new_node = onnx.helper.make_node(
                    op_type, inputs=list(node.input), outputs=[new_out],
                    name=f"{node.name}_split_{split_count}" if node.name else f"{op_type}_split_{split_count}",
                )
                for attr in node.attribute: new_node.attribute.append(attr)
                new_nodes.append(new_node)
                consumer_node.input[input_idx] = new_out
        model.graph.node.extend(new_nodes)
        if new_nodes: print(f"  Split {len(new_nodes)} shared {op_type} consumer edges.")
    return model

def fold_int32_dequantize_nodes(model):
    """Fold constant INT32 DQ nodes (bias/scales) into float32 initializers for WASM."""
    init_map = {i.name: i for i in model.graph.initializer}
    nodes_to_remove, new_initializers = [], []
    for node in model.graph.node:
        if node.op_type != 'DequantizeLinear' or len(node.input) < 3: continue
        zp_name, x_name, scale_name = node.input[2], node.input[0], node.input[1]
        if zp_name not in init_map or init_map[zp_name].data_type != onnx.TensorProto.INT32: continue
        if x_name not in init_map or scale_name not in init_map: continue
        x_arr = onnx.numpy_helper.to_array(init_map[x_name]).astype(np.int32)
        scale_arr = onnx.numpy_helper.to_array(init_map[scale_name]).astype(np.float32)
        zp_arr = onnx.numpy_helper.to_array(init_map[zp_name]).astype(np.int32)
        dequant = (x_arr.astype(np.float64) - zp_arr.astype(np.float64)) * scale_arr.astype(np.float64)
        out_name = node.output[0]
        new_initializers.append(onnx.numpy_helper.from_array(dequant.astype(np.float32), name=out_name))
        nodes_to_remove.append(node)
    for node in nodes_to_remove: model.graph.node.remove(node)
    model.graph.initializer.extend(new_initializers)
    used_names = set()
    for node in model.graph.node: used_names.update(node.input)
    for gi in model.graph.input: used_names.add(gi.name)
    to_remove = [i for i in model.graph.initializer if i.name not in used_names]
    for i in to_remove: model.graph.initializer.remove(i)
    print(f"  Folded {len(nodes_to_remove)} INT32 DQ nodes; removed {len(to_remove)} initializers.")
    return model

def topological_sort(model):
    """Ensures nodes are in topological order."""
    graph = model.graph
    known_tensors = set(i.name for i in graph.initializer) | set(i.name for i in graph.input)
    sorted_nodes, nodes_to_sort = [], list(graph.node)
    while nodes_to_sort:
        ready_idx = -1
        for i, node in enumerate(nodes_to_sort):
            if all(not inp or inp in known_tensors for inp in node.input):
                ready_idx = i; break
        if ready_idx == -1: ready_idx = 0
        node = nodes_to_sort.pop(ready_idx)
        sorted_nodes.append(node)
        for out in node.output: known_tensors.add(out)
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return model

def fix_dcr_mode(model):
    """Convert DepthToSpace from CRD to DCR losslessy by permuting upstream Conv weights."""
    init_map = {i.name: i for i in model.graph.initializer}
    output_to_node = {}
    for n in model.graph.node:
        for out in n.output: output_to_node[out] = n
    count = 0
    for node in model.graph.node:
        if node.op_type != 'DepthToSpace': continue
        mode, r = 'CRD', 2
        for a in node.attribute:
            if a.name == 'mode': mode = a.s.decode()
            if a.name == 'blocksize': r = a.i
        if mode != 'CRD': continue
        print(f"  Converting {node.name} CRD→DCR...")
        upstream_name = node.input[0]
        upstream = output_to_node.get(upstream_name)
        while upstream is not None and upstream.op_type in ('DequantizeLinear', 'QuantizeLinear'):
            upstream_name = upstream.input[0]; upstream = output_to_node.get(upstream_name)
        if upstream is None or upstream.op_type != 'Conv':
            for a in node.attribute:
                if a.name == 'mode': a.s = b'DCR'
            continue
        weight_ref = upstream.input[1]
        if weight_ref not in init_map:
            dq = output_to_node.get(weight_ref)
            if dq and dq.op_type == 'DequantizeLinear': weight_ref = dq.input[0]
        if weight_ref not in init_map:
            for a in node.attribute:
                if a.name == 'mode': a.s = b'DCR'
            continue
        w_arr = onnx.numpy_helper.to_array(init_map[weight_ref])
        N, C = w_arr.shape[0], w_arr.shape[0] // (r*r)
        perm = np.zeros(N, dtype=np.int64)
        for a in range(r):
            for b in range(r):
                for c in range(C): perm[a*r*C + b*C + c] = c*r*r + a*r + b
        init_map[weight_ref].CopyFrom(onnx.numpy_helper.from_array(w_arr[perm], name=weight_ref))
        if len(upstream.input) > 2 and upstream.input[2]:
            b_ref = upstream.input[2]
            if b_ref not in init_map:
                dq = output_to_node.get(b_ref)
                if dq and dq.op_type == 'DequantizeLinear': b_ref = dq.input[0]
            if b_ref in init_map:
                b_arr = onnx.numpy_helper.to_array(init_map[b_ref])
                if b_arr.ndim == 1 and b_arr.shape[0] == N:
                    init_map[b_ref].CopyFrom(onnx.numpy_helper.from_array(b_arr[perm], name=b_ref))
        conv_out = upstream.output[0]
        for qnode in model.graph.node:
            if qnode.op_type not in ('QuantizeLinear', 'DequantizeLinear'): continue
            if qnode.input[0] == conv_out or (qnode.op_type == 'DequantizeLinear' and qnode.input[0] == weight_ref):
                for slot in (1, 2):
                    if len(qnode.input) > slot and qnode.input[slot] in init_map:
                        arr = onnx.numpy_helper.to_array(init_map[qnode.input[slot]])
                        if arr.ndim == 1 and arr.shape[0] == N:
                            init_map[qnode.input[slot]].CopyFrom(onnx.numpy_helper.from_array(arr[perm], name=qnode.input[slot]))
        for a in node.attribute:
            if a.name == 'mode': a.s = b'DCR'
        count += 1
    return model

def replace_depthtospace_with_reshape(model):
    """Replace DepthToSpace nodes with Reshape+Transpose+Reshape decomposition."""
    shape_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.tensor_type.HasField('shape'): shape_map[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    nodes_to_remove, new_nodes, new_inits = [], [], []
    count = 0
    for node in model.graph.node:
        if node.op_type != 'DepthToSpace': continue
        r, mode = 2, 'DCR'
        for a in node.attribute:
            if a.name == 'blocksize': r = a.i
            if a.name == 'mode': mode = a.s.decode()
        inp, out = node.input[0], node.output[0]
        in_shape = shape_map.get(inp)
        if not in_shape: continue
        N, rrC, H, W = in_shape; C = rrC // (r*r)
        prefix = (node.name or f'dts_{count}').replace('/', '_')
        sh1, sh2 = f'{prefix}_sh1', f'{prefix}_sh2'; r1, t1 = f'{prefix}_r1', f'{prefix}_t1'
        if mode == 'DCR': s1, p = [N, r, r, C, H, W], [0, 3, 4, 1, 5, 2]
        else: s1, p = [N, C, r, r, H, W], [0, 1, 4, 2, 5, 3]
        s2 = [N, C, H*r, W*r]
        new_inits += [onnx.numpy_helper.from_array(np.array(s1, dtype=np.int64), name=sh1), onnx.numpy_helper.from_array(np.array(s2, dtype=np.int64), name=sh2)]
        new_nodes += [onnx.helper.make_node('Reshape', [inp, sh1], [r1], name=f'{prefix}_reshape1'), onnx.helper.make_node('Transpose', [r1], [t1], perm=p, name=f'{prefix}_transpose'), onnx.helper.make_node('Reshape', [t1, sh2], [out], name=f'{prefix}_reshape2')]
        nodes_to_remove.append(node); count += 1
    for n in nodes_to_remove: model.graph.node.remove(n)
    model.graph.node.extend(new_nodes); model.graph.initializer.extend(new_inits)
    return model

def simplify_onnx(model_path, output_path):
    from onnxsim import simplify
    print(f"Simplifying {model_path}...")
    model = onnx.load(model_path); ms, check = simplify(model)
    onnx.save(ms, output_path); return output_path

def quantize_model_static(model_path, output_path):
    if not os.path.exists(model_path): return
    temp_simp = model_path.replace(".onnx", "_temp_simp.onnx")
    simplify_onnx(model_path, temp_simp)
    fixed_path = temp_simp.replace("_temp_simp.onnx", "_fixed.onnx")
    fix_batch_size(temp_simp, fixed_path, batch_size=1)
    simplify_onnx(fixed_path, fixed_path)
    model = onnx.load(fixed_path); input_names = [i.name for i in model.graph.input]
    dr = SBCalibrationDataReader(fixed_path, input_names, num_samples=256, batch_size=1)
    
    is_drift = "drift" in model_path.lower()
    
    # Identify CFG nodes to exclude in drift model
    nodes_to_exclude = []
    if is_drift:
        # We look for the last few nodes which are Sub, Mul, Add for CFG
        m = onnx.load(fixed_path)
        # Typically the last nodes are the CFG ones
        # We'll search for them by type and proximity to output
        output_name = m.graph.output[0].name
        current_node = None
        for node in m.graph.node:
            if output_name in node.output:
                current_node = node
                break
        
        if current_node:
            nodes_to_exclude.append(current_node.name) # The final Add
            # Trace back to Sub and Mul
            for inp in current_node.input:
                for node in m.graph.node:
                    if inp in node.output:
                        nodes_to_exclude.append(node.name)
                        # Go one more level for the Sub
                        for inp2 in node.input:
                            for node2 in m.graph.node:
                                if inp2 in node2.output and node2.op_type == 'Sub':
                                    nodes_to_exclude.append(node2.name)

    print(f"Quantizing {fixed_path} to INT8 (static)...")
    if nodes_to_exclude:
        print(f"  Excluding {len(nodes_to_exclude)} nodes from quantization: {nodes_to_exclude}")
    
    # MinMax is safer for preserving the full dynamic range of generative models
    cal_method = CalibrationMethod.MinMax
    
    # STM32N6 (stedgeai 4.0) prefers symmetric signed INT8 for activations
    extra_options = {'EnableShapeInference': True, 'ActivationSymmetric': True, 'WeightSymmetric': True}

    quantize_static(
        model_input=fixed_path, model_output=output_path,
        calibration_data_reader=dr, quant_format=1, # 1 = QDQ
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=cal_method,
        extra_options=extra_options,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm']
    )
    print("Post-processing...")
    model = onnx.load(output_path); model = strip_nonstandard_opsets(model)
    model = fold_int32_dequantize_nodes(model); model = split_shared_qdq_outputs(model)
    model = topological_sort(model); onnx.save(model, output_path)
    simplify_onnx(output_path, output_path)
    from onnx import shape_inference as _si
    model = onnx.load(output_path); model = _si.infer_shapes(model); onnx.save(model, output_path)
    model = onnx.load(output_path); model = fix_dcr_mode(model); onnx.save(model, output_path)
    model = onnx.load(output_path); model = replace_depthtospace_with_reshape(model); model = topological_sort(model)
    model = _si.infer_shapes(model); onnx.save(model, output_path)
    for p in [temp_simp, fixed_path]:
        if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    base_dir = "enhanced_label_sb/onnx"; models = ["drift.onnx", "generator.onnx"]
    for m in models:
        src = os.path.join(base_dir, m); dst = os.path.join(base_dir, m.replace(".onnx", "_static_int8.onnx"))
        try: quantize_model_static(src, dst)
        except Exception as e: print(f"Failed to quantize {m}: {e}")
    print("\nStatic Quantization complete with Entropy calibration.")
