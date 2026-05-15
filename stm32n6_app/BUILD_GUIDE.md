# STM32N6570-DK Display Demo — Complete Build & Flash Guide

This guide builds the Schrödinger Bridge generative model demo that runs on the
STM32N6 NPU and displays a 96×96 generated image centred on the MB1860A panel.

## Prerequisites

| Tool | Version | Source |
|------|---------|--------|
| STM32CubeIDE | 2.1.1 | Downloaded (`.sh.zip`) |
| STM32CubeProgrammer | bundled with IDE | — |
| ARM GNU Toolchain | 15.2.rel1 | `/home/pc/sby/demoicair_fixed/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi/` |
| ST Edge AI 4.0 | installed | `/home/pc/opt/ST/STEdgeAI/4.0/` |
| STM32CubeN6 package | 1.x | [st.com/stm32cuben6](https://www.st.com/en/embedded-software/stm32cuben6.html) |

---

## Step 0 — Install STM32CubeIDE 2.1.1

```bash
# Unzip the downloaded bundle
cd ~/Downloads
unzip st-stm32cubeide_2.1.1_28236_20260312_0043_amd64.deb_bundle.sh.zip

# Make executable and run the installer
chmod +x st-stm32cubeide_2.1.1_28236_20260312_0043_amd64.deb_bundle.sh
sudo ./st-stm32cubeide_2.1.1_28236_20260312_0043_amd64.deb_bundle.sh

# STM32CubeIDE installs to /opt/st/stm32cubeide_2.1.1/
# STM32CubeProgrammer CLI is at:
#   /opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.*/
#   tools/bin/STM32_Programmer_CLI
```

---

## Step 1 — Download STM32CubeN6 firmware package

The BSP drivers and HAL are in the STM32CubeN6 package:

```bash
# Option A: via STM32CubeIDE Package Manager
#   Help → Manage Embedded Software Packages → STM32Cube → STM32CubeN6 → Install

# Option B: direct download from st.com
# Navigate to: https://www.st.com/en/embedded-software/stm32cuben6.html
# Download STM32Cube_FW_N6_V1.x.x.zip, extract to ~/STM32CubeN6/
```

---

## Step 2 — Artifacts produced by stedgeai (already done)

The `stedgeai generate` command already ran successfully and produced:

```
st_ai_output/
├── network.c           ← ST.AI C code (1.1 MB)
├── network.h           ← network interface header
├── network_details.h   ← tensor/layer metadata
└── network_data.bin    ← quantised INT8 weights (12 MB)
```

These files are the "AI model" side of your firmware.

---

## Step 3 — Project structure

```
stm32n6_app/
├── App/
│   ├── app_ai_display.c    ← inference + display logic (this repo)
│   └── app_ai_display.h
├── Core/
│   ├── Inc/main.h
│   └── Src/main.c          ← HAL init + main loop
├── Linker/
│   └── STM32N6570_APP.ld   ← linker script (app only, not FSBL)
├── Makefile
└── BUILD_GUIDE.md          ← this file

# Plus ST.AI generated files (copy or symlink from st_ai_output/):
#   network.c, network.h, network_details.h
```

---

## Step 4 — Understand the two-binary boot model

The STM32N6 **always requires a FSBL** (First Stage Boot Loader).  
The NPU-mapped SRAM clocks and the XSPI memory controllers (Flash + PSRAM)
**must be configured by the FSBL** before your application starts.

```
┌─────────────────────────────────────────────────────────────────┐
│  NOR Flash (64 MB, XSPI1)  ─ mapped at 0x70000000             │
│  ┌────────────┬──────────────────────┬──────────────────────┐  │
│  │  0x70000000│  0x70100400          │  0x71000000          │  │
│  │    FSBL    │  Application binary  │  network_data.bin    │  │
│  │   (~32 KB) │     (your code)      │     (12 MB)          │  │
│  └────────────┴──────────────────────┴──────────────────────┘  │
│                                                                  │
│  PSRAM (32 MB, XSPI2)  ─ mapped at 0x90000000                  │
│  ┌──────────────────────────┬───────────────────────────────┐   │
│  │  0x90000000              │  0x90720000                   │   │
│  │  Activations scratch     │  Output buf + Framebuffer     │   │
│  │  (7.4 MB, network)       │  (110 KB + 768 KB)            │   │
│  └──────────────────────────┴───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Getting the FSBL

ST provides a ready-made FSBL example with the STM32N6570-DK Getting Started package:

```bash
# Clone the ST image classification example (includes FSBL)
git clone https://github.com/STMicroelectronics/STM32N6-GettingStarted-ImageClassification.git

# The FSBL project is at:
#   STM32N6-GettingStarted-ImageClassification/FSBL/
# Pre-built binary (if provided):
#   STM32N6-GettingStarted-ImageClassification/Binary/fsbl.bin
```

The FSBL does:
1. Configures XSPI2 in Memory-Mapped XIP mode (NOR Flash accessible at 0x70000000)
2. Jumps to application entry at `0x70100400`

> **Important:** The standard FSBL template does **NOT** initialise PSRAM (XSPI1).
> Our `App_AI_Display_Init()` calls `BSP_XSPI_RAM_Init(0)` +
> `BSP_XSPI_RAM_EnableMemoryMappedMode(0)` at startup to enable PSRAM access
> at `0x90000000` before any large buffer is touched.

---

## Step 5 — Option A: Build with STM32CubeIDE (recommended)

### 5A.1 — Create a new STM32 project

1. Open STM32CubeIDE
2. **File → New → STM32 Project**
3. Board selector: search `STM32N6570-DK`, select it, click **Next**
4. Project name: `sb_display`, C project, **Finish**
5. When asked "Initialize all peripherals with default mode?": click **Yes** (gives you a base CubeMX config)

### 5A.2 — Configure peripherals in CubeMX (`.ioc` file)

Double-click the `.ioc` file to open CubeMX:

| Peripheral | Setting |
|-----------|---------|
| XSPI1 | Memory-mapped NOR Flash (configured by FSBL — leave as-is or match FSBL config) |
| XSPI2 | Memory-mapped PSRAM (configured by FSBL — leave as-is or match FSBL config) |
| LTDC | Enable, pixel format RGB565, width 800, height 480 |
| NPU | Enable |
| RCC | PLL1 → 480 MHz SYSCLK |

Click **Generate Code**.

### 5A.3 — Add application files to the project

In the Project Explorer, right-click `Core/Src` → **Import → File System**:
- Add `app_ai_display.c` from this repo

Right-click `Core/Inc` → **Import**: add `app_ai_display.h`

Add ST.AI generated files:
- Copy `st_ai_output/network.c` → project `Core/Src/`
- Copy `st_ai_output/network.h`, `network_details.h` → project `Core/Inc/`

### 5A.4 — Add ST.AI runtime library

The ST.AI runtime is bundled with STM32CubeIDE:

```
/opt/st/stm32cubeide_2.1.1/plugins/
  com.st.stm32cube.ide.mcu.externaltools.ai.*/
    tools/Lib/GCC/libStAI_CM55_fp_rel.a   ← link this
    tools/Inc/                             ← include this dir
```

In CubeIDE: **Project → Properties → C/C++ Build → Settings → GCC Linker → Libraries**:
- Add library path: (path above ending in `GCC/`)
- Add library: `StAI_CM55_fp_rel`

Include path: **GCC Compiler → Include paths** → add the `tools/Inc/` path.

### 5A.5 — Update `main.c`

Replace the generated `main.c` with (or merge into) `Core/Src/main.c` from this repo.
Key additions:
```c
#include "app_ai_display.h"
// in main(): call App_AI_Display_Init() then App_AI_Display_Run() in while(1)
```

### 5A.6 — Set linker script for application start at 0x70100400

**Project → Properties → C/C++ Build → Settings → GCC Linker → General**:
- Linker script: point to `Linker/STM32N6570_APP.ld` from this repo

Or in the CubeMX-generated `.ld`, change the FLASH origin from `0x70000000` to `0x70100400`.

### 5A.7 — Build

**Project → Build Project** (or Ctrl+B)

Expected output in `Debug/` or `Release/`:
```
sb_display.elf
sb_display.bin
sb_display.hex
```

---

## Step 6 — Option B: Build with Makefile

```bash
# Set paths
export STM32CUBE_DIR=~/STM32CubeN6
export TOOLCHAIN_DIR=/home/pc/sby/demoicair_fixed/tools/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi
export STAI_DIR=/home/pc/sby/enhancedoptimaltransport/st_ai_output

# Copy ST.AI generated files into project
cp $STAI_DIR/network.c  stm32n6_app/Core/Src/
cp $STAI_DIR/network.h  stm32n6_app/Core/Inc/
cp $STAI_DIR/network_details.h stm32n6_app/Core/Inc/

# Build
cd stm32n6_app
make STM32CUBE_DIR=$STM32CUBE_DIR \
     TOOLCHAIN_DIR=$TOOLCHAIN_DIR \
     STAI_DIR=$STAI_DIR

# Output: build/sb_display.bin, build/sb_display.hex
```

---

## Step 7 — Flash the board

Connect the STM32N6570-DK via USB-C (STLINK connector on the board).

### 7.1 — Flash the FSBL (first time only)

```bash
STM32_Programmer_CLI \
  --connect port=SWD freq=8000 reset=HWrst \
  --download /path/to/fsbl.bin 0x70000000 \
  --verify
```

### 7.2 — Flash the weight binary (network_data.bin)

```bash
STM32_Programmer_CLI \
  --connect port=SWD freq=8000 reset=HWrst \
  --download /home/pc/sby/enhancedoptimaltransport/st_ai_output/network_data.bin \
             0x71000000 \
  --verify
```

> **Note:** The weight binary is 12 MB. The NOR Flash on the DK is 64 MB total,
> so 0x71000000 (16 MB offset from base) is well within range.

### 7.3 — Flash the application binary

```bash
STM32_Programmer_CLI \
  --connect port=SWD freq=8000 reset=HWrst \
  --download build/sb_display.bin 0x70100400 \
  --verify --go
```

`--go` performs a hardware reset and starts execution.

### 7.4 — Via STM32CubeIDE (alternative)

In CubeIDE: **Run → Run Configurations → STM32 Cortex-M C/C++ Application**
- Set load address to `0x70100400` in the **Startup** tab
- Add a second memory segment for `network_data.bin` at `0x71000000`

---

## Step 8 — Expected behaviour

After `--go`:
1. FSBL runs (~50 ms): configures clocks, XSPI, PSRAM
2. Application starts at `0x70100400`
3. `App_AI_Display_Init()`:
   - NPU clocks enabled
   - LCD initialised (panel goes white or black briefly)
   - ST.AI network context bound to PSRAM activations and Flash weights
4. `App_AI_Display_Run()`:
   - Random INT8 latent `z` generated (~N(0, 0.8) in float space)
   - NPU inference runs (~TBD ms depending on clock speed)
   - 96×96 float32 output converted to RGB565
   - Image painted centred on the 800×480 panel (at pixel 352, 192)
5. 2-second pause, then a new image is generated

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Screen stays black | LTDC not initialised | Check BSP_LCD_Init return code; verify LTDC clocks in CubeMX |
| Red square on screen | ST.AI returned non-SUCCESS | Check `stai_network_run` error; verify weights at 0x71000000 |
| Hard fault at startup | PSRAM not accessible | FSBL must configure XSPI2 before app starts |
| `stai_network_init` returns error | ST.AI library version mismatch | Match `libStAI_CM55_fp_rel.a` version to `network.c` generation date |
| `Error_Handler()` called | Clock config failed | Use CubeMX-generated SystemClock_Config for exact HSE/HSI values |

---

## Memory budget summary

| Region | Address | Size | Content |
|--------|---------|------|---------|
| Flash code | 0x70100400 | ~200 KB | Application binary |
| Flash weights | 0x71000000 | 12 MB | network_data.bin |
| AXIRAM1 .bss | 0x24000000 | <10 KB | Variables, stack overflow |
| DTCM stack | 0x20000000 | 32 KB | Call stack |
| PSRAM activations | 0x90000000 | 7.4 MB | ST.AI scratch |
| PSRAM output buf | 0x90720000 | 110 KB | float32[1,3,96,96] |
| PSRAM framebuffer | 0x9073B000 | 768 KB | RGB565[800×480] |
| **PSRAM total used** | | **~8.3 MB** | of 32 MB available |
