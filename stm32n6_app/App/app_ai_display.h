#ifndef APP_AI_DISPLAY_H
#define APP_AI_DISPLAY_H

#include <stdint.h>

/* ── Display geometry ─────────────────────────────────────────── */
#define APP_IMG_W              96
#define APP_IMG_H              96
#define DISPLAY_W              800
#define DISPLAY_H              480
/* Centre the 96×96 tile on the 800×480 panel */
#define DISPLAY_X_OFFSET       ((DISPLAY_W - APP_IMG_W) / 2)   /* 352 */
#define DISPLAY_Y_OFFSET       ((DISPLAY_H - APP_IMG_H) / 2)   /* 192 */

/* ── ST.AI quantisation parameters (from network.h) ──────────── */
/* Input 1 – z latent:           INT8 symmetric, shape [1,8,12,12] */
#define Z_SCALE                0.03618268296122551f
/* Input 2 – text embedding:     INT8 symmetric, shape [1,512]     */
#define EMB_SCALE              0.0013926321407780051f
/* Gaussian prior scale (matches Python config.CST_COEF_GAUSSIAN_PRIO) */
#define CST_GAUSSIAN_PRIOR     0.8f

/* ── Buffer sizes ─────────────────────────────────────────────── */
#define NET_IN1_SIZE           1152    /* 1×8×12×12 INT8 bytes      */
#define NET_IN2_SIZE           512     /* 1×512 INT8 bytes           */
#define NET_OUT_SIZE_BYTES     110592  /* 1×3×96×96 float32 bytes   */
#define NET_ACT_SIZE           7373408 /* activation scratch (PSRAM) */

/* ── Public API ───────────────────────────────────────────────── */
/**
 * Initialise the ST.AI network context and display panel.
 * Must be called once after HAL_Init() and SystemClock_Config().
 * Requires:
 *   - XSPI1 in XIP mode (code + weights flash accessible)
 *   - XSPI2/AHB in Memory-Mapped mode (PSRAM accessible at 0x90000000)
 *   - LTDC clock enabled, BSP drivers linked
 */
void App_AI_Display_Init(void);

/**
 * Run one inference and paint the result on the MB1860A panel.
 * Generates random INT8 latent z, uses zero text-embedding (unconditional).
 * Call in a loop or once from main().
 */
void App_AI_Display_Run(void);

#endif /* APP_AI_DISPLAY_H */
