#include "csi_source.h"
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "csi_source";

#define CSI_UART_NUM            UART_NUM_1
#define CSI_UART_RX_GPIO        GPIO_NUM_21
#define CSI_UART_TX_GPIO        GPIO_NUM_22
#define CSI_UART_BAUD_RATE      921600
#define CSI_UART_RX_BUFFER_SIZE (16 * 1024)

static const uint8_t CSI_FRAME_MAGIC[4] = {'C', 'S', 'I', '1'};
#define CSI_FRAME_HEADER_SIZE   10
#define CSI_FRAME_CHECKSUM_SIZE 2

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

esp_err_t csi_source_init(void)
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

bool csi_source_read_frame(csi_frame_t *out_frame)
{
    uint8_t frame_header[CSI_FRAME_HEADER_SIZE] = {0};
    size_t matched = 0;

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

    if (!uart_read_exact(frame_header + sizeof(CSI_FRAME_MAGIC),
                         CSI_FRAME_HEADER_SIZE - sizeof(CSI_FRAME_MAGIC))) {
        return false;
    }

    const uint16_t len = read_le16(&frame_header[8]);
    if (len == 0 || len > CSI_FRAME_MAX_RAW_LEN) {
        ESP_LOGW(TAG, "invalid raw_len=%u (max=%u)", len, CSI_FRAME_MAX_RAW_LEN);
        return false;
    }

    if (!uart_read_exact((uint8_t *)out_frame->payload, len)) {
        return false;
    }

    uint8_t checksum_bytes[CSI_FRAME_CHECKSUM_SIZE] = {0};
    if (!uart_read_exact(checksum_bytes, sizeof(checksum_bytes))) {
        return false;
    }

    uint32_t sum = 0;
    for (size_t i = 0; i < CSI_FRAME_HEADER_SIZE; ++i) {
        sum += frame_header[i];
    }
    for (size_t i = 0; i < len; ++i) {
        sum += (uint8_t)out_frame->payload[i];
    }
    const uint16_t expected = read_le16(checksum_bytes);
    const uint16_t actual = (uint16_t)(sum & 0xffff);
    if (expected != actual) {
        ESP_LOGW(TAG, "checksum failed: expected=0x%04x actual=0x%04x", expected, actual);
        return false;
    }

    out_frame->seq = read_le32(&frame_header[4]);
    out_frame->raw_len = len;
    return true;
}
