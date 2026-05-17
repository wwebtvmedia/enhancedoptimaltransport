#!/bin/bash
# build_and_run.sh - Compile and execute the OpenCL/Vulkan inference application

# Create build directory
mkdir -p build
cd build

echo "Configuring project with CMake..."
cmake ..

echo "Compiling..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful. Copying assets..."
    cp ../dsp_imp.cl .
    cp ../../horse_embedding.bin . 2>/dev/null || true
    
    echo "Running inference_app..."
    ./inference_app
else
    echo "Build failed."
    exit 1
fi
