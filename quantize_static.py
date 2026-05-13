import onnx
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
