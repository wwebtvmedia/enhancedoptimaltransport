#!/usr/bin/env bash
# Pi5 build + smoke-test for the ONNX→OpenCL runtime.
#
# Usage:
#   cd opencl/runtime
#   ./build_and_run_pi5.sh                 # full pipeline (20 drift + gen)
#   ./build_and_run_pi5.sh verify-drift    # 1 drift step, diff every op vs golden
#   ./build_and_run_pi5.sh verify-gen      # generator, diff every op vs golden
#   ./build_and_run_pi5.sh both            # verify drift then verify generator
#
# Expectations:
#   - run from   .../opencl/runtime
#   - python3 has: onnx onnxruntime numpy pillow   (only needed once to build assets)
#   - horse_embedding.bin is at the repo root (../../)

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-run}"

ASSETS_OK=1
[[ -f assets/drift/manifest.json     ]] || ASSETS_OK=0
[[ -f assets/generator/manifest.json ]] || ASSETS_OK=0
[[ -d assets/drift/init              ]] || ASSETS_OK=0
[[ -d assets/generator/init          ]] || ASSETS_OK=0

if [[ "$ASSETS_OK" -eq 0 ]]; then
  echo "[Pi5] assets/ missing — running export_onnx_assets.py to build them"
  echo "      (this also dumps ORT golden tensors used by --verify-*)"
  python3 export_onnx_assets.py
fi

mkdir -p build
echo "[Pi5] Configuring CMake..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null

echo "[Pi5] Building (4 cores)..."
cmake --build build -j4

EMB="../../horse_embedding.bin"
[[ -f "$EMB" ]] || { echo "ERR: $EMB not found"; exit 1; }

COMMON=(--drift assets/drift --gen assets/generator
        --embedding "$EMB"
        --kernels src/onnx_kernels.cl
        --log-dir logs)

case "$MODE" in
  verify-drift)
    echo "[Pi5] Drift only, 1 step, diff every op vs golden"
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 1 --drift-only --verify-drift
    ;;
  verify-gen)
    echo "[Pi5] Generator only, diff every op vs golden"
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 20 --gen-only --verify-gen
    ;;
  both)
    echo "[Pi5] Drift verify (1 step)..."
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 1 --drift-only --verify-drift
    echo "[Pi5] Generator verify..."
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 20 --gen-only --verify-gen
    ;;
  run|*)
    echo "[Pi5] Full pipeline: 20 drift steps → generator → output_horse.ppm"
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 20 --out output_horse.ppm
    ;;
esac

echo "[Pi5] Logs in $(ls -dt logs/run_* 2>/dev/null | head -1)"
