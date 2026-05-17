#!/bin/bash
# build_and_run_pi5.sh - DEPRECATED legacy stub.
#
# This script builds the OLD vulkan_opencl_app.cpp, which uses hardcoded
# identity weights and CANNOT reproduce the real ONNX models. It is kept
# only as a smoke-test for the legacy display pipeline.
#
# For the real ONNX-driven inference (drift + generator with actual
# weights, golden-tensor verification, per-op logging) use:
#
#     cd opencl/runtime
#     ./build_and_run_pi5.sh

cat <<'WARN'
============================================================
WARNING: This script builds the LEGACY stub (identity weights).
         The new ONNX-driven runtime lives in opencl/runtime/.

         To build/run the real inference:
             cd opencl/runtime
             ./build_and_run_pi5.sh

         Continuing with the legacy build in 5 seconds...
         (Ctrl+C to abort)
============================================================
WARN
sleep 5

# Create build directory
mkdir -p build
cd build

echo "[Pi5] Configuring project..."
cmake ..

echo "[Pi5] Compiling with 4 cores..."
make -j4

if [ $? -eq 0 ]; then
    echo "[Pi5] Build successful. Copying kernels and assets..."
    cp ../dsp_imp.cl .
    
    # Try to find horse_embedding.bin in multiple locations
    if [ -f "../horse_embedding.bin" ]; then
        cp ../horse_embedding.bin .
    elif [ -f "../../horse_embedding.bin" ]; then
        cp "../../horse_embedding.bin" .
    fi
    
    echo "----------------------------------------------------"
    echo "[Pi5] Running inference_app..."
    echo "Note: If you have multiple OpenCL platforms, you can"
    echo "select one using the OCL_ICD_VENDORS variable."
    echo "----------------------------------------------------"
    
    # Run the application
    ./inference_app
    
    if [ $? -eq 0 ]; then
        echo "----------------------------------------------------"
        echo "[Success] Image generated: opencl/build/output_full_pipeline.ppm"
        echo "----------------------------------------------------"
    fi
else
    echo "[Error] Build failed."
    exit 1
fi
