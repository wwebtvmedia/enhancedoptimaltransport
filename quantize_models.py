"""
INT8 quantization for ONNX models.

- drift_int8.onnx      : static INT8 (dynamic quantization of Gemm with
                          transB=1 breaks MatMul shapes due to an onnxruntime
                          Gemm-decomposition bug in the two-pass CFG wrapper)
- generator_int8.onnx  : dynamic INT8 (Conv-only; generator has no CFG)
"""
import onnx
import os
import shutil
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_generator_dynamic(model_path, output_path):
    """Dynamic INT8 for the generator (VAE decoder, Conv layers only)."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    print(f"Dynamic INT8: {model_path} -> {output_path}")
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=['Conv'],
        extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT},
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    base_dir = "enhanced_label_sb/onnx"

    # Drift: use static int8 (dynamic int8 breaks due to onnxruntime Gemm bug)
    drift_static = os.path.join(base_dir, "drift_static_int8.onnx")
    drift_int8 = os.path.join(base_dir, "drift_int8.onnx")
    if os.path.exists(drift_static):
        shutil.copy2(drift_static, drift_int8)
        print(f"Drift int8: copied from static -> {drift_int8}")
    else:
        print("Run quantize_static.py first to generate drift_static_int8.onnx")

    # Generator: dynamic int8 works fine
    quantize_generator_dynamic(
        os.path.join(base_dir, "generator.onnx"),
        os.path.join(base_dir, "generator_int8.onnx"),
    )

    print("\nQuantization complete.")
