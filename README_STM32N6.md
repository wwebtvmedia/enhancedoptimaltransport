# Schrödinger Bridge Generative Model — STM32N6 NPU Deployment

End-to-end guide: training a latent diffusion model with a Schrödinger Bridge
sampler → quantising to INT8 → compiling for the STM32N6 NPU → displaying
generated images on the MB1860A panel of the **STM32N6570-DK** board.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware](#2-hardware)
3. [Software Requirements](#3-software-requirements)
4. [Repository Structure](#4-repository-structure)
5. [Model Architecture](#5-model-architecture)
6. [Pipeline — PC Side](#6-pipeline--pc-side)
   - 6.1 [Train the model](#61-train-the-model)
   - 6.2 [Export FP32 ONNX](#62-export-fp32-onnx)
   - 6.3 [Static INT8 Quantisation](#63-static-int8-quantisation)
   - 6.4 [Analyse with ST Edge AI](#64-analyse-with-st-edge-ai)
   - 6.5 [Generate NPU binary](#65-generate-npu-binary)
7. [Pipeline — Embedded Build](#7-pipeline--embedded-build)
   - 7.1 [Open projects in STM32CubeIDE](#71-open-projects-in-stm32cubeide)
   - 7.2 [Build ExtMemLoader](#72-build-extmemloader)
   - 7.3 [Build FSBL](#73-build-fsbl)
   - 7.4 [Build Application](#74-build-application)
8. [Flash Memory Layout](#8-flash-memory-layout)
9. [Flashing the Board](#9-flashing-the-board)
   - 9.1 [Using STM32CubeProgrammer GUI](#91-using-stm32cubeprogrammer-gui)
   - 9.2 [Using STM32CubeProgrammer CLI](#92-using-stm32cubeprogrammer-cli)
   - 9.3 [Using STM32CubeIDE Run/Debug](#93-using-stm32cubeide-rundebug)
10. [Expected Behaviour](#10-expected-behaviour)
11. [Memory Budget](#11-memory-budget)
12. [Troubleshooting](#12-troubleshooting)
13. [Key Technical Notes](#13-key-technical-notes)

---

## 1. Overview

This project implements a **Schrödinger Bridge (SB)** latent-space generative
model trained on a small image dataset. The decoder (VAE generator) is
exported to ONNX, quantised to signed INT8, and compiled by **ST Edge AI 4.0**
for the STM32N6 Cortex-M55 + NPU (Neural Processing Unit).

At runtime on the board:
- A random INT8 latent vector `z ~ N(0, 0.8)` is drawn in SRAM.
- The NPU runs the generator in ~TBD ms.
- The float32 output `[1, 3, 96, 96]` is converted to RGB565.
- The 96×96 image is centred and painted on the 800×480 MB1860A display.

**ST Edge AI generate report summary**

| Property | Value |
|----------|-------|
| Model | `generator_static_int8.onnx` |
| Generated | 2026-05-14 |
| ST Edge AI version | 4.0.0-20500 |
| Inputs | `z` int8[1,8,12,12], `text_embedding` int8[1,512] |
| Output | `reconstruction` f32[1,3,96,96] |
| Weights (Flash) | 12,248,936 B (11.68 MiB) |
| Activations (PSRAM) | 7,373,408 B (7.03 MiB) |
| MACC | 559,294,868,307 |
| Compression vs FP32 | −67.4 % |

---

## 2. Hardware

| Item | Details |
|------|---------|
| Board | **STM32N6570-DK** (Discovery Kit) |
| MCU | STM32N6570-HxQ — Cortex-M55 + NPU, 600 MHz |
| Display | **MB1860A** — 5" WVGA 800×480, LTDC parallel RGB565 |
| NOR Flash | MX66UW1G45G 128 MB, connected to **XSPI2** → CPU 0x70000000 |
| PSRAM | APS256XX-OBR 32 MB, connected to **XSPI1** → CPU 0x90000000 |
| Debugger | Embedded ST-LINK v3 (USB-C connector CN6) |

**XSPI routing note:** On the STM32N6570-DK the NOR Flash is on XSPI2 (mapped
at 0x70000000) and the PSRAM is on XSPI1 (mapped at 0x90000000). This is the
reverse of what the STM32N6 datasheet names suggest — the FSBL template in
STM32CubeN6 reflects this correctly.

---

## 3. Software Requirements

### PC (Linux / Ubuntu)

| Tool | Version | Path / Install |
|------|---------|---------------|
| Python | 3.10+ | system |
| PyTorch | 2.x | `pip install torch` |
| onnx, onnxruntime | latest | `pip install onnx onnxruntime onnxsim` |
| **ST Edge AI** | **4.0** | `/home/pc/opt/ST/STEdgeAI/4.0/` |
| **STM32CubeIDE** | **2.1.1** | `/opt/st/stm32cubeide_2.1.1/` |
| STM32CubeN6 repo | main | `/home/pc/sby/STM32CubeN6/` (cloned) |
| ARM GNU Toolchain | 15.2.rel1 | `/home/pc/sby/demoicair_fixed/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi/` |

### Paths used by the build system

```bash
export STEDGEAI=/home/pc/opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai
export STM32CUBE=/home/pc/sby/STM32CubeN6
export TOOLCHAIN=/home/pc/sby/demoicair_fixed/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi
export PATH=$TOOLCHAIN/bin:$PATH
```

---

## 4. Repository Structure

```
enhancedoptimaltransport/
├── config.py                        # Model hyper-parameters & paths
├── models.py                        # SB model, VAEGenerator, DriftWrapper
├── training.py                      # EnhancedLabelTrainer, export_onnx()
├── data_management.py               # Dataset + CLASS_DESCRIPTIONS
│
├── export_for_stm32.py              # Step 1 — export FP32 ONNX (opset 17)
├── quantize_static.py               # Step 2 — INT8 QDQ + STM32 post-processing
├── test_stm32_onnx.py               # CPU inference test (CPU only, no board)
│
├── enhanced_label_sb/onnx/
│   ├── generator.onnx               # FP32 generator (exported)
│   ├── drift.onnx                   # FP32 drift model (exported)
│   ├── generator_static_int8.onnx   # INT8 QDQ generator (quantised + patched)
│   └── drift_static_int8.onnx      # INT8 QDQ drift model
│
├── st_ai_output/                    # Generated by stedgeai
│   ├── network.c                    # ST.AI C runtime code (1.1 MB)
│   ├── network.h                    # Network interface header
│   ├── network_details.h            # Tensor/layer metadata
│   ├── network_data.bin             # INT8 weights binary (12 MB) ← flash at 0x71000000
│   └── network_generate_report.txt  # Full stedgeai report
│
└── stm32n6_app/                     # Embedded application
    ├── App/
    │   ├── app_ai_display.c         # ST.AI inference + display logic
    │   └── app_ai_display.h
    ├── Core/
    │   ├── Src/main.c               # HAL init + main loop
    │   └── Inc/main.h
    ├── Linker/
    │   └── STM32N6570_APP.ld        # Linker script (app at 0x70100400, PSRAM at 0x90000000)
    ├── Makefile                     # Alternative Makefile build
    └── BUILD_GUIDE.md               # Detailed build reference
```

---

## 5. Model Architecture

### Schrödinger Bridge (training)

The training uses the **Enhanced Label SB** framework:
- **NeuralTokenizer** — byte-level text encoder → 512-dim embedding
- **DriftModel** — U-Net style network with SpatialSplitAttention blocks
  predicts the SB drift field; uses Classifier-Free Guidance (CFG scale 6.5)
- **VAEGenerator** — convolutional decoder: latent [1,8,12,12] → image [1,3,96,96]

### Deployed model (generator only)

Only the **VAE decoder** runs on the NPU. The ONNX graph is headless:

```
Inputs:
  z              int8 [1, 8, 12, 12]   scale=0.03618, zp=0
  text_embedding int8 [1, 512]         scale=0.001393, zp=0

Output:
  reconstruction float32 [1, 3, 96, 96]   (tanh range [-1, 1])
```

The NeuralTokenizer and Drift model are **not** in the deployed graph.
For the demo, `text_embedding` is set to all-zeros (unconditional generation).

---

## 6. Pipeline — PC Side

### 6.1 Train the model

```bash
cd enhancedoptimaltransport
python training.py          # trains until convergence, saves checkpoint
```

Checkpoint saved to `enhanced_label_sb/checkpoints/`.

### 6.2 Export FP32 ONNX

```bash
python export_for_stm32.py
```

- Loads checkpoint via `training.EnhancedLabelTrainer.load_for_inference()`
- Forces `ONNX_OPSET_VERSION = 17` so `nn.LayerNorm` exports as a single
  `LayerNormalization` op (avoids float32 intermediate islands in QDQ graph)
- `USE_NEURAL_TOKENIZER = True` in `config.py` makes both models headless
  (no `Gather` op — atonn does not support Gather)
- Outputs: `enhanced_label_sb/onnx/generator.onnx`, `drift.onnx`

### 6.3 Static INT8 Quantisation

```bash
python quantize_static.py
```

Key steps applied in order:

| Step | Function | Reason |
|------|----------|--------|
| ONNX Runtime static quant | `quantize_static()` | Signed INT8, symmetric, QDQ format |
| Split shared Q/DQ fan-out | `split_shared_qdq_outputs()` | atonn requires each consumer has its own Q/DQ pair |
| Fix DCR permutation | `fix_dcr_mode()` | DepthToSpace CRD→DCR weight permutation for correctness |
| Replace DepthToSpace | `replace_depthtospace_with_reshape()` | atonn 1.1.3 does NOT support DepthToSpace op despite listing it |
| Topological sort | `topological_sort()` | New nodes appended at end after DepthToSpace removal |
| Shape inference | `onnx.shape_inference` | Required for stedgeai to resolve tensor shapes |

Outputs: `enhanced_label_sb/onnx/generator_static_int8.onnx`, `drift_static_int8.onnx`

### 6.4 Analyse with ST Edge AI

```bash
$STEDGEAI analyze \
  --model enhanced_label_sb/onnx/generator_static_int8.onnx \
  --target stm32n6 \
  "--st-neural-art=-E" \
  --name network \
  --workspace workspace \
  --output output
```

> **`"--st-neural-art=-E"` is required.** The `-E` flag passes
> `--continue-on-errors` to the atonn compiler. Without it, atonn aborts on
> `Node Transpose_267 not mapped` — a known limitation of atonn 1.1.3's SW
> scheduler which cannot map the `perm=[0,2,3,1]` Transpose that connects
> unquantised LayerNorm float32 regions to INT8 MatMul in the attention blocks.

### 6.5 Generate NPU binary

```bash
$STEDGEAI generate \
  --model enhanced_label_sb/onnx/generator_static_int8.onnx \
  --target stm32n6 \
  "--st-neural-art=-E" \
  --binary \
  --workspace workspace_gen \
  --output st_ai_output
```

Outputs in `st_ai_output/`:

| File | Size | Description |
|------|------|-------------|
| `network.c` | 1.1 MB | ST.AI C runtime — include in your project |
| `network.h` | 21 KB | Network API header |
| `network_details.h` | 197 KB | Per-layer tensor metadata |
| `network_data.bin` | **12 MB** | INT8 weights — flash at `0x71000000` |

---

## 7. Pipeline — Embedded Build

Three STM32CubeIDE projects must be built **in order**.

### Prerequisites

The projects are located inside the cloned STM32CubeN6 repository. They use
Eclipse `PARENT-N-PROJECT_LOC` relative links to reference HAL, CMSIS and BSP
sources — **do not copy them** into the workspace, import in-place.

```
STM32CubeN6 repo: /home/pc/sby/STM32CubeN6/
Template:         Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE/
  Boot/           → Template_XIP_FSBL      (FSBL — runs from internal SRAM)
  AppS/           → Template_XIP_AppS      (application — XIP from Flash)
  ExtMemLoader/   → Template_FSBL_XIP_ExtMemLoader  (Flash programmer loader)
```

### 7.1 Open projects in STM32CubeIDE

1. **File → Import → General → Existing Projects into Workspace → Next**
2. Root directory:
   ```
   /home/pc/sby/STM32CubeN6/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE
   ```
3. Select all three projects, **uncheck "Copy projects into workspace"** → Finish

Or from a terminal (headless import, requires CubeIDE to be closed):

```bash
/opt/st/stm32cubeide_2.1.1/stm32cubeide \
  --launcher.suppressErrors -nosplash \
  -application org.eclipse.cdt.managedbuilder.core.headlessbuild \
  -data /home/pc/STM32CubeIDE/workspace_2.1.1 \
  -import /home/pc/sby/STM32CubeN6/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE/Boot \
  -import /home/pc/sby/STM32CubeN6/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE/AppS \
  -import /home/pc/sby/STM32CubeN6/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE/ExtMemLoader
```

**Fix postbuild script permissions** (needed once after cloning):

```bash
find /home/pc/sby/STM32CubeN6 -name "postbuild.sh" -exec chmod +x {} \;
```

### 7.2 Build ExtMemLoader

Right-click **`Template_FSBL_XIP_ExtMemLoader`** → **Build Project**

This produces the flash programmer loader:
```
STM32CubeIDE/ExtMemLoader/Debug/Template_FSBL_XIP_ExtMemLoader.elf
```
The postbuild script renames it to `.stldr` and copies it into STM32CubeProgrammer's
`ExternalLoader/` directory so the programmer can write to the external NOR Flash.

### 7.3 Build FSBL

Right-click **`Template_XIP_FSBL`** → **Build Project**

Output:
```
STM32CubeIDE/Boot/Debug/Template_XIP_FSBL.bin   ← flash at 0x70000000
```

The FSBL (First Stage Boot Loader):
1. Runs entirely from internal AXISRAM2 (copied there by the Boot ROM)
2. Configures XSPI2 in 8-line Memory-Mapped XIP mode (NOR Flash → 0x70000000)
3. Sets system clock to 600 MHz
4. Jumps to the application entry point at `0x70100400`

> The FSBL does **not** initialise PSRAM. Our application initialises it via
> `BSP_XSPI_RAM_Init(0)` + `BSP_XSPI_RAM_EnableMemoryMappedMode(0)` in
> `App_AI_Display_Init()` before any large buffer is accessed.

### 7.4 Build Application

Add the AI inference + display code to `Template_XIP_AppS`:

1. Copy these files into the AppS project `Core/Src/` and `Core/Inc/`:

   ```bash
   cp stm32n6_app/App/app_ai_display.c  \
      <workspace>/Template_XIP_AppS/Core/Src/
   cp stm32n6_app/App/app_ai_display.h  \
      <workspace>/Template_XIP_AppS/Core/Inc/
   cp st_ai_output/network.c            \
      <workspace>/Template_XIP_AppS/Core/Src/
   cp st_ai_output/network.h            \
      st_ai_output/network_details.h    \
      <workspace>/Template_XIP_AppS/Core/Inc/
   ```

2. Replace (or merge) `Core/Src/main.c` with `stm32n6_app/Core/Src/main.c`.

3. Add ST.AI include + library paths in **Project Properties → C/C++ Build → Settings**:

   | Setting | Value |
   |---------|-------|
   | GCC Compiler → Include paths | `/home/pc/opt/ST/STEdgeAI/4.0/Middlewares/ST/AI/Inc` |
   | GCC Compiler → Include paths | `/home/pc/opt/ST/STEdgeAI/4.0/Middlewares/ST/AI/Npu/ll_aton` |
   | GCC Linker → Library search path | `/home/pc/opt/ST/STEdgeAI/4.0/Middlewares/ST/AI/Lib/GCC/ARMCortexM55` |
   | GCC Linker → Libraries | `NetworkRuntime1200_CM55_GCC` |

4. Also add BSP LCD + XSPI sources to the project (or via **Add Files**):
   ```
   /home/pc/sby/STM32CubeN6/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_lcd.c
   /home/pc/sby/STM32CubeN6/Drivers/BSP/STM32N6570-DK/stm32n6570_discovery_xspi.c
   ```

5. Right-click **`Template_XIP_AppS`** → **Build Project**

Output:
```
STM32CubeIDE/AppS/Debug/Template_XIP_AppS.bin   ← flash at 0x70100400
```

---

## 8. Flash Memory Layout

```
NOR Flash (MX66UW1G45G 128 MB, XSPI2, mapped at 0x70000000)
┌──────────────┬──────────────────────────┬──────────────────────────┐
│  0x70000000  │       0x70100400         │       0x71000000         │
│   FSBL       │    Application (XIP)     │   network_data.bin       │
│  Template_   │    Template_XIP_AppS     │   (INT8 weights, 12 MB)  │
│  XIP_FSBL    │    + app_ai_display      │                          │
│  (~32 KB)    │    + network.c           │                          │
└──────────────┴──────────────────────────┴──────────────────────────┘

PSRAM (APS256XX 32 MB, XSPI1, mapped at 0x90000000)
┌─────────────────────────┬────────────────┬─────────────────────────┐
│  0x90000000             │  0x90720000    │  0x9073B000             │
│  ST.AI activations      │  Output buf    │  RGB565 framebuffer     │
│  (7,373,408 B, 7.03 MB) │  (110,592 B)   │  (800×480×2 = 768 KB)  │
└─────────────────────────┴────────────────┴─────────────────────────┘

Internal SRAM (STM32N657X0)
┌────────────────────┬───────────────────┬───────────────────────────┐
│  AXISRAM2          │  DTCM             │  AXISRAM3-6               │
│  0x34000000 2 MB   │  0x30000000 128K  │  (NPU internal, managed   │
│  .data / .bss      │  Stack            │   by atonn runtime)       │
└────────────────────┴───────────────────┴───────────────────────────┘
```

---

## 9. Flashing the Board

Connect the board via **USB-C on CN6** (ST-LINK connector, left side of board).

### 9.1 Using STM32CubeProgrammer GUI

Launch STM32CubeProgrammer:
```bash
/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin/STM32_Programmer_CLI --version
# or launch the GUI version
```

**Step 1 — Connect**

- Interface: **ST-LINK**, Port: **SWD**, Frequency: **8000 kHz**
- Click **Connect**

**Step 2 — Register the External Loader**

The ExtMemLoader `.stldr` must be registered so the programmer can write to
the NOR Flash. If the postbuild script ran successfully it is already in
CubeProgrammer's `ExternalLoader/` directory as
`Template_FSBL_XIP_ExtMemLoader.stldr`.

In STM32CubeProgrammer: **Settings (gear icon) → External Loaders → Add**
Browse to `Template_FSBL_XIP_ExtMemLoader.stldr`.

Or, add the official loader that ships with CubeProgrammer:
`MX66UW1G45G_STM32N6570-DK.stldr` — it is already installed.

**Step 3 — Flash FSBL** (first time only, or after board reset)

- Erasing: Full chip erase is not needed; sector erase is automatic
- **Download** → File: `Boot/Debug/Template_XIP_FSBL.bin`
- Start address: `0x70000000`
- Click **Start Programming**

**Step 4 — Flash weights**

- **Download** → File: `st_ai_output/network_data.bin`
- Start address: `0x71000000`
- Click **Start Programming** (takes ~60 s for 12 MB)

**Step 5 — Flash application**

- **Download** → File: `AppS/Debug/Template_XIP_AppS.bin`
- Start address: `0x70100400`
- Click **Start Programming**

**Step 6 — Reset and run**

Click **Reset & Run** in STM32CubeProgrammer, or press the **RESET** button
on the board (black button near CN6).

---

### 9.2 Using STM32CubeProgrammer CLI

Set the CLI path:
```bash
export PROG=/opt/st/stm32cubeide_2.1.1/plugins/\
com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/\
tools/bin/STM32_Programmer_CLI

CONNECT="--connect port=SWD freq=8000 reset=HWrst"
TEMPLATE=/home/pc/sby/STM32CubeN6/Projects/STM32N6570-DK/Templates/Template_FSBL_XIP/STM32CubeIDE
STAI=/home/pc/sby/enhancedoptimaltransport/st_ai_output
```

**Flash FSBL** (once):
```bash
$PROG $CONNECT \
  --download $TEMPLATE/Boot/Debug/Template_XIP_FSBL.bin 0x70000000 \
  --verify
```

**Flash weights** (12 MB, ~60 s):
```bash
$PROG $CONNECT \
  --download $STAI/network_data.bin 0x71000000 \
  --verify
```

**Flash application + reset**:
```bash
$PROG $CONNECT \
  --download $TEMPLATE/AppS/Debug/Template_XIP_AppS.bin 0x70100400 \
  --verify --go
```

`--go` performs a hardware reset and starts execution immediately.

> **All three flash operations in one command:**
> ```bash
> $PROG $CONNECT \
>   --download $TEMPLATE/Boot/Debug/Template_XIP_FSBL.bin 0x70000000 \
>   --download $STAI/network_data.bin 0x71000000 \
>   --download $TEMPLATE/AppS/Debug/Template_XIP_AppS.bin 0x70100400 \
>   --verify --go
> ```

---

### 9.3 Using STM32CubeIDE Run/Debug

1. **Run → Run Configurations → STM32 Cortex-M C/C++ Application → New**
2. Project: `Template_XIP_AppS`
3. **Startup** tab:
   - Add memory segment: `network_data.bin` → `0x71000000`
   - Load address for app: `0x70100400`
4. **Debugger** tab: ST-LINK, SWD
5. Click **Run**

---

## 10. Expected Behaviour

After the board is powered and flashed:

```
Time 0 ms    Boot ROM executes (stored in STM32N6 internal ROM)
             ↓
Time ~5 ms   FSBL starts (Template_XIP_FSBL.bin at 0x70000000)
             Configures XSPI2 XIP → NOR Flash accessible
             Sets CPU to 600 MHz
             Jumps to 0x70100400
             ↓
Time ~10 ms  Application starts (Template_XIP_AppS.bin)
             App_AI_Display_Init() called:
               BSP_XSPI_RAM_Init(0)         → PSRAM ready at 0x90000000
               BSP_LCD_Init(0, LANDSCAPE)   → MB1860A panel initialised
               stai_network_init()          → ST.AI context created
               stai_network_set_weights()   → points to 0x71000000 (flash)
               stai_network_set_activations() → points to 0x90000000 (PSRAM)
             ↓
Time ~50 ms  App_AI_Display_Run() called:
               Random z ~ N(0, 0.8) → quantised to INT8
               stai_network_run() → NPU inference (blocking)
               float32[1,3,96,96] → RGB565 conversion
               BSP_LCD_FillRGBRect() → 96×96 tile painted at (352, 192)
             ↓
             HAL_Delay(2000) → wait 2 s → generate next image
```

**Visual result:** a 96×96 generated image centred on the 800×480 display,
with new images appearing every ~2 seconds.

---

## 11. Memory Budget

| Region | Address | Size | Content |
|--------|---------|------|---------|
| FSBL binary | 0x70000000 | ~32 KB | Template_XIP_FSBL.bin |
| App binary | 0x70100400 | ~200 KB | Template_XIP_AppS.bin |
| NPU weights | 0x71000000 | **12.0 MB** | network_data.bin |
| NOR Flash free | 0x71BEDB28 | ~115 MB | available |
| AXISRAM2 .bss | 0x34000000 | <10 KB | small variables |
| DTCM stack | 0x30000000 | 32 KB | call stack |
| PSRAM activations | 0x90000000 | **7.03 MB** | ST.AI scratch |
| PSRAM output buf | 0x90720000 | 108 KB | float32 network output |
| PSRAM framebuffer | 0x9073B000 | 768 KB | RGB565 display buffer |
| **PSRAM total used** | | **~8.0 MB** | of 32 MB available |

---

## 12. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Screen stays black after reset | LTDC not initialised | Check `BSP_LCD_Init()` return code; verify LTDC clock in CubeMX `.ioc` |
| Red 96×96 square on screen | `stai_network_run()` returned error | Verify `network_data.bin` was flashed at exactly `0x71000000`; check weight size |
| Hard fault on startup | PSRAM not accessible | FSBL must complete before app starts; check FSBL binary at `0x70000000`; check XSPI2 8-line config |
| `stai_network_init` returns error | Runtime/generated code mismatch | Regenerate with same ST Edge AI 4.0 version; rebuild project |
| `Error_Handler()` infinite loop | Clock config failed | Use CubeMX-generated `SystemClock_Config()` for your exact HSI/HSE frequency |
| CubeIDE: `No rule to make target extmem.c` | Projects were copied (not imported in-place) | Delete from workspace, re-import without "Copy projects into workspace" |
| postbuild.sh: Permission denied | Script not executable | `find /home/pc/sby/STM32CubeN6 -name "postbuild.sh" -exec chmod +x {} \;` |
| stedgeai: `Node Transpose_267 not mapped` | atonn SW scheduler limitation | Use `"--st-neural-art=-E"` (passes `--continue-on-errors` to atonn) |
| stedgeai: `NOT IMPLEMENTED: only DCR mode` | DepthToSpace not supported by atonn 1.1.3 | Re-run `quantize_static.py` — `replace_depthtospace_with_reshape()` removes all DepthToSpace ops |
| ORT error on `test_stm32_onnx.py` | Model inputs expect INT8 but float32 supplied | The quantised model expects `z` and `text_embedding` as INT8 — the test script handles this |

---

## 13. Key Technical Notes

### Why opset 17 for ONNX export?

`nn.LayerNorm` at opset < 17 decomposes to `ReduceMean + Sub + Pow + ReduceMean + Add + Sqrt + Div + Mul + Add`. These intermediate tensors remain in float32 in the QDQ graph, creating unquantised float32 "islands" that break stedgeai's shape inference. Opset 17 exports a single `LayerNormalization` op which quantises cleanly.

### Why signed INT8 (not unsigned)?

ST Edge AI / atonn requires **signed INT8** (`QuantType.QInt8`, `ActivationSymmetric=True`). Unsigned INT8 (the ONNX Runtime default) is rejected by the stedgeai toolchain with a type error.

### Why are shared Q/DQ fan-out nodes split?

When a single `QuantizeLinear` node feeds multiple consumers (e.g. skip connections in ResNet blocks), atonn requires each consumer to have its own dedicated `QuantizeLinear`/`DequantizeLinear` pair. `split_shared_qdq_outputs()` in `quantize_static.py` clones these nodes.

### Why replace DepthToSpace?

atonn 1.1.3 lists `DepthToSpace` as a supported op but rejects all instances at compile time with `NOT IMPLEMENTED: only DCR mode is supported` (including DCR mode nodes). The decomposition to `Reshape[N,r,r,C,H,W] → Transpose[0,3,4,1,5,2] → Reshape[N,C,Hr,Wr]` (6D intermediates) is mapped correctly as SW epochs.

### Why `--st-neural-art=-E`?

`SpatialSplitAttention` uses `F.linear` on NCHW tensors permuted to NHWC via `perm=[0,2,3,1]` Transpose. The mixed-precision boundary (float32 LayerNorm → INT8 MatMul) creates a Transpose that atonn's SW scheduler cannot schedule in any epoch type. `-E` (`--continue-on-errors`) allows compilation to proceed — those layers fall back to software execution. The correct long-term fix is to replace `F.linear` projections with `Conv2d(1×1)` which avoids the NHWC permutation, but this requires retraining.

### PSRAM initialisation

The standard `Template_XIP_FSBL` only initialises XSPI2 (NOR Flash XIP). PSRAM (XSPI1) is **not** initialised by the FSBL. `App_AI_Display_Init()` calls:
```c
BSP_XSPI_RAM_Init(0);
BSP_XSPI_RAM_EnableMemoryMappedMode(0);
```
before any PSRAM access. These must complete before the first access to `0x90000000`.
