#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>

#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "csi_receiver";

// ==================== 1. 串口配置参数 ====================
#define CSI_UART_NUM            UART_NUM_1
#define CSI_UART_RX_GPIO        GPIO_NUM_21
#define CSI_UART_TX_GPIO        GPIO_NUM_22
#define CSI_UART_BAUD_RATE      921600
#define CSI_UART_RX_BUFFER_SIZE (16 * 1024)

// ==================== 2. 数据协议帧定义 ====================
static const uint8_t CSI_FRAME_MAGIC[4] = {'C', 'S', 'I', '1'};
#define CSI_FRAME_MAX_RAW_LEN   1024
#define CSI_FRAME_HEADER_SIZE   10
#define CSI_FRAME_CHECKSUM_SIZE 2

// ==================== 3. 辅助转换函数 ====================
static uint16_t read_le16(const uint8_t *src)
{
    return ((uint16_t)src[0]) | (((uint16_t)src[1]) << 8);
}

static uint32_t read_le32(const uint8_t *src)
{
    return ((uint32_t)src[0]) |
           (((uint32_t)src[1]) << 8) |
           (((uint32_t)src[2]) << 16) |
           (((uint32_t)src[3]) << 24);
}

// ==================== 4. 驱动与接收逻辑 ====================

/**
 * @brief 初始化接收端 UART 驱动
 */
static esp_err_t init_csi_uart(void)
{
    const uart_config_t uart_config = {
        .baud_rate = CSI_UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
    };

    esp_err_t ret = uart_driver_install(CSI_UART_NUM, CSI_UART_RX_BUFFER_SIZE, 0, 0, NULL, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to install CSI UART driver: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_param_config(CSI_UART_NUM, &uart_config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to configure CSI UART: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = uart_set_pin(CSI_UART_NUM,
                       CSI_UART_TX_GPIO,
                       CSI_UART_RX_GPIO,
                       UART_PIN_NO_CHANGE,
                       UART_PIN_NO_CHANGE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to set CSI UART pins: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG,
             "CSI UART%d initialized: RX GPIO%d, TX GPIO%d, baud=%d",
             CSI_UART_NUM,
             CSI_UART_RX_GPIO,
             CSI_UART_TX_GPIO,
             CSI_UART_BAUD_RATE);
    return ESP_OK;
}

/**
 * @brief 阻塞读取精确长度的串口数据
 */
static bool uart_read_exact(uint8_t *dst, size_t len)
{
    size_t received = 0;
    while (received < len) {
        const int ret = uart_read_bytes(CSI_UART_NUM,
                                        dst + received,
                                        len - received,
                                        pdMS_TO_TICKS(1000));
        if (ret < 0) {
            ESP_LOGE(TAG, "CSI UART read failed");
            return false;
        }
        received += (size_t)ret;
    }
    return true;
}

/**
 * @brief 从串口读取并校验一帧 CSI 数据
 */
static bool read_csi_frame(uint32_t *seq, int8_t *raw_payload, size_t *raw_len)
{
    uint8_t frame_header[CSI_FRAME_HEADER_SIZE] = {0};
    size_t matched = 0;

    // 1. 同步魔数
    while (matched < sizeof(CSI_FRAME_MAGIC)) {
        uint8_t byte = 0;
        if (!uart_read_exact(&byte, 1)) {
            return false;
        }

        if (byte == CSI_FRAME_MAGIC[matched]) {
            frame_header[matched++] = byte;
        } else {
            matched = (byte == CSI_FRAME_MAGIC[0]) ? 1 : 0;
            if (matched == 1) {
                frame_header[0] = byte;
            }
        }
    }

    // 2. 读取头部剩余 6 字节
    if (!uart_read_exact(frame_header + sizeof(CSI_FRAME_MAGIC),
                         CSI_FRAME_HEADER_SIZE - sizeof(CSI_FRAME_MAGIC))) {
        return false;
    }

    // 3. 读取长度并检查
    const uint16_t len = read_le16(&frame_header[8]);
    if (len == 0 || len > CSI_FRAME_MAX_RAW_LEN) {
        ESP_LOGW(TAG, "invalid raw_len=%u (max=%u)", len, CSI_FRAME_MAX_RAW_LEN);
        return false;
    }

    // 4. 读取载荷数据
    if (!uart_read_exact((uint8_t *)raw_payload, len)) {
        return false;
    }

    // 5. 读取 2 字节校验和
    uint8_t checksum_bytes[CSI_FRAME_CHECKSUM_SIZE] = {0};
    if (!uart_read_exact(checksum_bytes, sizeof(checksum_bytes))) {
        return false;
    }

    // 6. 校验数据
    uint32_t sum = 0;
    for (size_t i = 0; i < CSI_FRAME_HEADER_SIZE; ++i) {
        sum += frame_header[i];
    }
    for (size_t i = 0; i < len; ++i) {
        sum += (uint8_t)raw_payload[i];
    }
    const uint16_t expected = read_le16(checksum_bytes);
    const uint16_t actual = (uint16_t)(sum & 0xffff);
    if (expected != actual) {
        ESP_LOGW(TAG, "checksum failed: expected=0x%04x actual=0x%04x", expected, actual);
        return false;
    }

    *seq = read_le32(&frame_header[4]);
    *raw_len = len;
    return true;
}

// ==================== 5. 主函数入口 ====================
void app_main(void)
{
    ESP_LOGI(TAG, "======= CSI UART Receiver Startup =======");

    // 1. 初始化 UART 串口外设
    if (init_csi_uart() != ESP_OK) {
        ESP_LOGE(TAG, "CSI UART Init failed! System halted.");
        return;
    }

    // 用于暂存接收到的每一帧载荷的缓冲区
    int8_t csi_raw_payload[CSI_FRAME_MAX_RAW_LEN] = {0};
    uint32_t seq = 0;
    size_t raw_len = 0;

    ESP_LOGI(TAG, "Listening for incoming CSI frames...");

    TickType_t last_tick = xTaskGetTickCount();
    uint32_t frame_count = 0;

    // 2. 主循环：不断接收串口数据并输出
    while (true) {
        if (read_csi_frame(&seq, csi_raw_payload, &raw_len)) {
            // 接收成功，累加帧数统计
            frame_count++;
        }
        
        TickType_t current_tick = xTaskGetTickCount();
        if ((current_tick - last_tick) >= pdMS_TO_TICKS(1000)) {
            float elapsed_sec = (float)(current_tick - last_tick) / configTICK_RATE_HZ;
            float fps = (float)frame_count / elapsed_sec;
            ESP_LOGI(TAG, "CSI RX FPS: %.2f (received %" PRIu32 " frames in %.2f s)", fps, frame_count, elapsed_sec);
            frame_count = 0;
            last_tick = current_tick;
        }
        
        // 适当避让，防止看门狗复位（如果串口没有数据，uart_read_bytes 会阻塞，这里有防备性延时）
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
