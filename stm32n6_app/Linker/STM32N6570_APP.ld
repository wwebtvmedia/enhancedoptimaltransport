/*
 * STM32N6570_APP.ld — Application XIP linker script
 *
 * Memory layout verified against ST official template:
 *   Templates/Template_FSBL_XIP/STM32CubeIDE/AppS/STM32N657XX_XIP.ld
 *   Applications/VENC/VENC_USB/STM32CubeIDE/Appli/STM32N657XX.ld
 *
 * STM32N6570-DK memory map:
 *   XSPI2 → NOR Flash (MX66UW1G45G 128MB) → CPU address 0x70000000
 *   XSPI1 → PSRAM (APS256XX 32MB)          → CPU address 0x90000000
 *
 * The FSBL (First Stage Boot Loader) must configure XSPI2 in XIP mode
 * before jumping to 0x70100400.  PSRAM is initialised by the application
 * via BSP_XSPI_RAM_Init(0) + BSP_XSPI_RAM_EnableMemoryMappedMode(0).
 */

ENTRY(Reset_Handler)

_Min_Heap_Size  = 0x4000;   /* 16 KB heap  */
_Min_Stack_Size = 0x8000;   /* 32 KB stack */

_estack = ORIGIN(RAM) + LENGTH(RAM);
_sstack = _estack - _Min_Stack_Size;

/* Export PSRAM bounds for optional MPU config */
__psram_region_start__ = ORIGIN(PSRAM);
__psram_region_end__   = ORIGIN(PSRAM) + LENGTH(PSRAM);

MEMORY
{
  /* NOR Flash XIP — app starts at +0x400 (boot header skip, per ST template) */
  ROM    (rx)   : ORIGIN = 0x70100400, LENGTH = 511K

  /* AXISRAM (2 MB internal SRAM on STM32N657X0) */
  RAM    (rw)   : ORIGIN = 0x34000000, LENGTH = 2048K

  /* DTCM (128 KB) — stack lives here */
  DTCM   (rw)   : ORIGIN = 0x30000000, LENGTH = 128K

  /* External PSRAM via XSPI1 — activations + framebuffers */
  PSRAM  (rw)   : ORIGIN = 0x90000000, LENGTH = 16M
}

SECTIONS
{
  /* ── Vector table + code + rodata in Flash (XIP) ───────────── */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > ROM

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    *(.RamFunc)
    *(.RamFunc*)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } > ROM

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } > ROM

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > ROM
  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } > ROM

  .preinit_array :
  {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
  } > ROM

  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > ROM

  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > ROM

  /* ── Initialised data loaded from Flash, runs in AXISRAM ────── */
  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT > ROM

  /* ── Zero-initialised BSS in AXISRAM ─────────────────────────── */
  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } > RAM

  /* ── PSRAM: large buffers (activations + framebuffers) ────────── */
  .psram_section (NOLOAD) :
  {
    . = ALIGN(32);
    *(.psram_bss)
    *(.psram_bss*)
    . = ALIGN(32);
  } > PSRAM

  /* ── Stack in DTCM ────────────────────────────────────────────── */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE(end = .);
    PROVIDE(_end = .);
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } > DTCM

  /DISCARD/ : { libc.a(*) libm.a(*) libgcc.a(*) }
  .ARM.attributes 0 : { *(.ARM.attributes) }
}
