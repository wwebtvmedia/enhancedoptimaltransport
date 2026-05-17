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
  numpy-ref)
    echo "[Pi5] Numpy reference walker (validates math without OpenCL)"
    python3 numpy_walker.py drift --continue 2>&1 | tail -30
    ;;
  run|*)
    echo "[Pi5] Full pipeline: 20 drift steps → generator → output_horse.ppm"
    ./build/onnx_opencl_runner "${COMMON[@]}" --steps 20 --out output_horse.ppm
    # Convert PPM → PNG for easy viewing (no Wayland window needed).
    if command -v python3 >/dev/null 2>&1; then
      python3 - <<'PY'
try:
    from PIL import Image
    im = Image.open("output_horse.ppm")
    im.save("output_horse.png")
    arr = im.convert("RGB")
    import numpy as np
    a = np.asarray(arr)
    print(f"[viewer] output_horse.png  shape={a.shape}  "
          f"min={int(a.min())} max={int(a.max())} mean={a.mean():.1f} std={a.std():.1f}")
    if a.std() < 5:
        print("[viewer] WARNING: very low std — image is near-uniform (likely grey).")
        print("         This means the OpenCL runtime is producing zeros for many ops.")
        print("         Re-run with: ./build_and_run_pi5.sh verify-drift")
        print("         and send back the first FAIL/ skip lines.")
except Exception as e:
    print(f"[viewer] couldn't convert: {e}")
PY
    fi
    # If a display is present, try to show it.
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
      if   command -v feh    >/dev/null 2>&1; then feh    output_horse.png &
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open output_horse.png &
      elif command -v eog    >/dev/null 2>&1; then eog    output_horse.png &
      fi
    fi
    ;;
esac

echo "[Pi5] Logs in $(ls -dt logs/run_* 2>/dev/null | head -1)"
