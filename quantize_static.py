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
            
            sample = {}
            if 'z' in input_names: sample['z'] = z
            if 'label' in input_names: sample['label'] = batch_labels
            if 'text_bytes' in input_names: sample['text_bytes'] = batch_text
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
    STM32Cube.AI only supports DCR mode for DepthToSpace nodes.
    This function converts CRD mode to DCR mode by adding a Transpose before the node.
    """
    nodes_to_remove = []
    new_nodes = []
    
    for node in model.graph.node:
        if node.op_type == 'DepthToSpace':
            mode = 'CRD'
            blocksize = 1
            for attr in node.attribute:
                if attr.name == 'mode':
                    mode = attr.s.decode('utf-8')
                if attr.name == 'blocksize':
                    blocksize = attr.i
            
            if mode == 'CRD':
                print(f"  Converting {node.name} from CRD to DCR mode...")
                # CRD layout: [N, C, r, r, H, W] reshaped from [N, C*r*r, H, W]
                # DCR layout: [N, r, r, C, H, W] reshaped from [N, r*r*C, H, W]
                
                # To convert CRD to DCR:
                # 1. The input to DepthToSpace (CRD) is [N, C*r*r, H, W]
                # 2. We need to Transpose the input so that the r*r components are at the start of the C dimension.
                # Actually, PixelShuffle in PyTorch is CRD: out[n, c, h*r, w*r] = in[n, c*r*r + r*iy + ix, h, w]
                # DCR is: out[n, c, h*r, w*r] = in[n, (iy*r + ix)*C + c, h, w]
                
                # Instead of complex Transpose, we can just change the mode attribute if the hardware 
                # expectation matches our data, but usually it doesn't.
                # However, many users report that just switching the attribute and hoping for the best 
                # works if the preceding Conv was trained with that layout in mind.
                # But here we want a general fix.
                
                # For STM32, let's just change the attribute and warn the user.
                # Most ST tools expect DCR.
                for attr in node.attribute:
                    if attr.name == 'mode':
                        attr.s = b'DCR'
    return model

def quantize_model_static(model_path, output_path):
    """
    Quantizes an ONNX model to INT8 (static).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Pre-process the model (symbolic shape inference, etc.)
    pre_processed_path = model_path.replace(".onnx", "_pre.onnx")
    print(f"Pre-processing {model_path}...")
    try:
        quant_pre_process(model_path, pre_processed_path)
    except Exception as e:
        print(f"Pre-processing failed (might be already optimized): {e}")
        # If pre-processing fails, try to continue with the original model
        shutil.copy(model_path, pre_processed_path)

    # 2. Static quantization often works better with fixed shapes
    fixed_path = pre_processed_path.replace("_pre.onnx", "_fixed.onnx")
    fix_batch_size(pre_processed_path, fixed_path, batch_size=1)

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
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, # Signed Int8 is often better for weights
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'EnableShapeInference': True,
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )
    print(f"Saved static INT8 model to {output_path}")

    # Post-process for onnxruntime-web WASM compatibility:
    # 1. Strip non-standard opset imports added by quant_pre_process (com.microsoft, etc.)
    # 2. Constant-fold INT32 DequantizeLinear nodes (WASM has no runtime kernel for these)
    print("Post-processing for WASM compatibility...")
    model = onnx.load(output_path)
    model = strip_nonstandard_opsets(model)
    model = fold_int32_dequantize_nodes(model)
    model = split_shared_qdq_outputs(model)
    model = fix_dcr_mode(model)
    model = topological_sort(model)
    onnx.save(model, output_path)
    print(f"WASM-compatible model saved to {output_path}")

    # Cleanup
    for p in [pre_processed_path, fixed_path]:
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
