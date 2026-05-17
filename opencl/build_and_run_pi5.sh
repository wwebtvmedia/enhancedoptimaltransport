#!/bin/bash
# build_and_run_pi5.sh - Optimized build and run for Raspberry Pi 5

# Create build directory
mkdir -p build
cd build

echo "[Pi5] Configuring project..."
cmake ..

echo "[Pi5] Compiling with 4 cores..."
make -j4

if [ $? -eq 0 ]; then
    echo "[Pi5] Build successful. Copying kernels..."
    cp ../dsp_imp.cl .
    
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
