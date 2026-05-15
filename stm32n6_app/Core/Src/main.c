/**
 * main.c — STM32N6570-DK Schrödinger Bridge image display demo
 *
 * Boots via FSBL, then:
 *   1. Configures system clocks (480 MHz CPU, NPU, LTDC)
 *   2. Initialises the ST.AI generator network + MB1860A LCD
 *   3. Loops: run inference → display generated 96×96 image
 *
 * Prereqs handled by the FSBL (first-stage boot loader):
 *   - XSPI1 Memory-Mapped (NOR Flash with app + weights at 0x70000000)
 *   - XSPI2 Memory-Mapped (32 MB PSRAM at 0x90000000)
 *   - TrustZone disabled (TZEN=0) or non-secure state entered
 */

#include "main.h"
#include "app_ai_display.h"

/* ── forward declarations ─────────────────────────────────────── */
static void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/* ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();

    /* Init AI inference engine + display panel */
    App_AI_Display_Init();

    /* Generate and display images continuously */
    while (1) {
        App_AI_Display_Run();
        HAL_Delay(2000);  /* pause 2 s between generations */
    }
}

/* ── Clock configuration ──────────────────────────────────────── */
/*
 * STM32N6 target: CPU = 480 MHz, AHB = 240 MHz, APB = 120 MHz
 * Adjust PLL dividers to match your exact HSE / MSIS source.
 * If you used STM32CubeMX to generate this project, replace this
 * function with the generated SystemClock_Config() from that output.
 */
static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    /* Enable HSI as source */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState       = RCC_HSI_ON;
    RCC_OscInitStruct.HSIDiv         = RCC_HSI_DIV1;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    /* PLL1 → SYSCLK = 480 MHz (HSI 64 MHz × 15 / 2) */
    RCC_OscInitStruct.PLL1.PLLState  = RCC_PLL_ON;
    RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL1.PLLM     = 4;
    RCC_OscInitStruct.PLL1.PLLN     = 75;
    RCC_OscInitStruct.PLL1.PLLP     = 1;  /* VCO/1 → 480 MHz */
    RCC_OscInitStruct.PLL1.PLLQ     = 4;
    RCC_OscInitStruct.PLL1.PLLR     = 4;
    RCC_OscInitStruct.PLL1.PLLFRACN = 0;
    RCC_OscInitStruct.PLL1.PLLClockOut = RCC_PLL1_DIVP;

    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    /* HCLK = SYSCLK/2 = 240 MHz, PCLK1/2/4/5 = HCLK/2 = 120 MHz */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2
                                | RCC_CLOCKTYPE_PCLK4 | RCC_CLOCKTYPE_PCLK5;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV2;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB4CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB5CLKDivider = RCC_HCLK_DIV2;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
        Error_Handler();
    }
}

/* ── GPIO init (USER LED on PC13 for heartbeat) ──────────────── */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);

    GPIO_InitStruct.Pin   = GPIO_PIN_13;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

/* ── Error handler ────────────────────────────────────────────── */
void Error_Handler(void)
{
    __disable_irq();
    while (1) {
        /* Toggle LED fast to signal error */
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        for (volatile uint32_t d = 0; d < 400000UL; d++);
    }
}
