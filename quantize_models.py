import onnx
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def fix_batch_size(model_path, output_path, batch_size=1):
    """
    Sets dynamic batch_size to a fixed value to avoid shape inference errors.
    """
    print(f"Fixing batch_size for {model_path}...")
    model = onnx.load(model_path)
    for input in model.graph.input:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_param == 'batch_size':
                dim.dim_value = batch_size
    onnx.save(model, output_path)
    return output_path

def quantize_model(model_path, output_path):
    """
    Quantizes an ONNX model to INT8 (dynamic).
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    fixed_path = model_path.replace(".onnx", "_fixed.onnx")
    fix_batch_size(model_path, fixed_path)

    print(f"Quantizing {fixed_path} to INT8 (dynamic)...")
    quantize_dynamic(
        model_input=fixed_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        extra_options={'EnableShapeInference': False}
    )
    print(f"Saved INT8 model to {output_path}")

    if os.path.exists(fixed_path):
        os.remove(fixed_path)

if __name__ == "__main__":
    base_dir = "enhanced_label_sb/onnx"
    models = ["drift.onnx", "generator.onnx"]

    for m in models:
        src = os.path.join(base_dir, m)
        dst_int8 = os.path.join(base_dir, m.replace(".onnx", "_int8.onnx"))
        quantize_model(src, dst_int8)

    print("\nQuantization complete.")
