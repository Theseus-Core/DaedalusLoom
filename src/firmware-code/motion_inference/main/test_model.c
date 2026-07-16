//模型测试的例程

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "nn_model.h"

#define TAG "main"

// UART Configuration
#define EX_UART_NUM UART_NUM_0
#define TXD_PIN UART_PIN_NO_CHANGE
#define RXD_PIN UART_PIN_NO_CHANGE
#define RX_BUF_SIZE 2048

static QueueHandle_t coord_queue = NULL;

#define NUM_BUFFERS 3
static float *csi_buffers[NUM_BUFFERS] = {NULL};
static bool csi_buffer_busy[NUM_BUFFERS] = {false};
static portMUX_TYPE buffer_mux = portMUX_INITIALIZER_UNLOCKED;

static float *get_free_csi_buffer(void) {
    float *buf = NULL;
    portENTER_CRITICAL(&buffer_mux);
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (!csi_buffer_busy[i]) {
            csi_buffer_busy[i] = true;
            buf = csi_buffers[i];
            break;
        }
    }
    portEXIT_CRITICAL(&buffer_mux);
    return buf;
}

static void release_csi_buffer(float *buf) {
    if (!buf) return;
    portENTER_CRITICAL(&buffer_mux);
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (csi_buffers[i] == buf) {
            csi_buffer_busy[i] = false;
            break;
        }
    }
    portEXIT_CRITICAL(&buffer_mux);
}

typedef enum {
    STATE_WAIT_HEADER_0,
    STATE_WAIT_HEADER_1,
    STATE_WAIT_HEADER_2,
    STATE_WAIT_HEADER_3,
    STATE_WAIT_LENGTH_0,
    STATE_WAIT_LENGTH_1,
    STATE_WAIT_LENGTH_2,
    STATE_WAIT_LENGTH_3,
    STATE_READ_PAYLOAD,
    STATE_DISCARD_PAYLOAD,
    STATE_WAIT_CHECKSUM_0,
    STATE_WAIT_CHECKSUM_1,
    STATE_WAIT_CHECKSUM_2,
    STATE_WAIT_CHECKSUM_3,
} parse_state_t;

// Core 0 Task: UART Coordinate Receiver (Binary framing protocol parser)
static void uart_recv_test_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting UART Receive Task on Core %d...", xPortGetCoreID());
    
    // Configure UART parameters
    uart_config_t uart_config = {
        .baud_rate = 921600,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    ESP_ERROR_CHECK(uart_driver_install(EX_UART_NUM, RX_BUF_SIZE * 4, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(EX_UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(EX_UART_NUM, TXD_PIN, RXD_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    
    uint8_t *temp_rx_buf = (uint8_t *)malloc(RX_BUF_SIZE);
    if (!temp_rx_buf) {
        ESP_LOGE(TAG, "Failed to allocate temp RX buffer!");
        vTaskDelete(NULL);
        return;
    }
    
    parse_state_t state = STATE_WAIT_HEADER_0;
    uint32_t payload_len = 0;
    uint32_t bytes_read = 0;
    uint8_t *payload_buf = NULL;
    uint32_t calc_checksum = 0;
    uint32_t recv_checksum = 0;
    
    while (1) {
        // Read chunks from the UART
        int read_len = uart_read_bytes(EX_UART_NUM, temp_rx_buf, RX_BUF_SIZE, pdMS_TO_TICKS(10));
        for (int i = 0; i < read_len; i++) {
            uint8_t byte = temp_rx_buf[i];
            
            switch (state) {
                case STATE_WAIT_HEADER_0:
                    if (byte == 0xAA) state = STATE_WAIT_HEADER_1;
                    break;
                case STATE_WAIT_HEADER_1:
                    if (byte == 0xBB) state = STATE_WAIT_HEADER_2;
                    else state = (byte == 0xAA) ? STATE_WAIT_HEADER_1 : STATE_WAIT_HEADER_0;
                    break;
                case STATE_WAIT_HEADER_2:
                    if (byte == 0xCC) state = STATE_WAIT_HEADER_3;
                    else state = (byte == 0xAA) ? STATE_WAIT_HEADER_1 : STATE_WAIT_HEADER_0;
                    break;
                case STATE_WAIT_HEADER_3:
                    if (byte == 0xDD) {
                        state = STATE_WAIT_LENGTH_0;
                        payload_len = 0;
                    } else {
                        state = (byte == 0xAA) ? STATE_WAIT_HEADER_1 : STATE_WAIT_HEADER_0;
                    }
                    break;
                case STATE_WAIT_LENGTH_0:
                    payload_len = (uint32_t)byte;
                    state = STATE_WAIT_LENGTH_1;
                    break;
                case STATE_WAIT_LENGTH_1:
                    payload_len |= ((uint32_t)byte << 8);
                    state = STATE_WAIT_LENGTH_2;
                    break;
                case STATE_WAIT_LENGTH_2:
                    payload_len |= ((uint32_t)byte << 16);
                    state = STATE_WAIT_LENGTH_3;
                    break;
                case STATE_WAIT_LENGTH_3:
                    payload_len |= ((uint32_t)byte << 24);
                    
                    // Validate payload length: must be exactly 45600 bytes
                    if (payload_len == 45600) {
                        payload_buf = (uint8_t *)get_free_csi_buffer();
                        if (payload_buf == NULL) {
                            ESP_LOGW(TAG, "No free CSI buffer! Dropping packet.");
                            bytes_read = 0;
                            state = STATE_DISCARD_PAYLOAD;
                        } else {
                            bytes_read = 0;
                            calc_checksum = 0;
                            state = STATE_READ_PAYLOAD;
                        }
                    } else {
                        ESP_LOGE(TAG, "Invalid length: %u (expected 45600)", payload_len);
                        state = STATE_WAIT_HEADER_0;
                    }
                    break;
                case STATE_READ_PAYLOAD:
                    payload_buf[bytes_read++] = byte;
                    calc_checksum += byte;
                    if (bytes_read >= payload_len) {
                        state = STATE_WAIT_CHECKSUM_0;
                        recv_checksum = 0;
                    }
                    break;
                case STATE_DISCARD_PAYLOAD:
                    bytes_read++;
                    // Discard payload + 4 bytes checksum
                    if (bytes_read >= payload_len + 4) {
                        state = STATE_WAIT_HEADER_0;
                    }
                    break;
                case STATE_WAIT_CHECKSUM_0:
                    recv_checksum = (uint32_t)byte;
                    state = STATE_WAIT_CHECKSUM_1;
                    break;
                case STATE_WAIT_CHECKSUM_1:
                    recv_checksum |= ((uint32_t)byte << 8);
                    state = STATE_WAIT_CHECKSUM_2;
                    break;
                case STATE_WAIT_CHECKSUM_2:
                    recv_checksum |= ((uint32_t)byte << 16);
                    state = STATE_WAIT_CHECKSUM_3;
                    break;
                case STATE_WAIT_CHECKSUM_3:
                    recv_checksum |= ((uint32_t)byte << 24);
                    
                    if (recv_checksum == calc_checksum) {
                        ESP_LOGI(TAG, "Valid CSI packet received. Pushing to Queue...");
                        if (xQueueSend(coord_queue, &payload_buf, pdMS_TO_TICKS(100)) != pdPASS) {
                            ESP_LOGW(TAG, "Queue full! Dropping packet.");
                            release_csi_buffer((float *)payload_buf);
                        }
                    } else {
                        ESP_LOGE(TAG, "Checksum failed! Calc: 0x%08X, Recv: 0x%08X", calc_checksum, recv_checksum);
                        release_csi_buffer((float *)payload_buf);
                    }
                    payload_buf = NULL;
                    state = STATE_WAIT_HEADER_0;
                    break;
            }
        }
    }
    free(temp_rx_buf);
    vTaskDelete(NULL);
}


static void uart_recv_task(void *pvParameters){

}

// Core 1 Task: AI Model Inference
static void ai_inference_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting AI Inference Task on Core %d...", xPortGetCoreID());
    float *input_features = NULL;
    
    const char *gesture_names[NUM_CLASSES] = {
        "挥手 (Wave/Cut)",
        "抓握 (Grip)",
        "画圈 (Circle/Draw_o)",
        "未知 (Unknown)"
    };
    
    while (1) {
        // Block indefinitely until coordinates are received
        if (xQueueReceive(coord_queue, &input_features, portMAX_DELAY) == pdPASS) {
            ESP_LOGI(TAG, "Core 1 dequeued sample. Executing CNN inference...");
            
            // Measure execution time
            int64_t start = esp_timer_get_time();
            int pred_class = nn_model_predict_cnn(input_features);
            int64_t end = esp_timer_get_time();
            
            if (pred_class >= 0 && pred_class < NUM_CLASSES) {
                const char *name = gesture_names[pred_class] ? gesture_names[pred_class] : "Unknown";
                ESP_LOGI(TAG, ">> Inference Result: %s (Time taken: %lld us)", 
                         name, (end - start));
            } else {
                ESP_LOGE(TAG, "Inference failed with code: %d", pred_class);
            }
            
            // Free memory back to the buffer pool
            release_csi_buffer(input_features);
        }
    }
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting Dual-Core Parallel Scenario with 1D-CNN Model...");
    
    // Initialize neural network model
    nn_model_init();
    
    // Allocate CSI buffer pool in internal SRAM
    for (int i = 0; i < NUM_BUFFERS; i++) {
        csi_buffers[i] = (float *)heap_caps_malloc(11400 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!csi_buffers[i]) {
            ESP_LOGE(TAG, "Failed to allocate CSI buffer %d in internal SRAM!", i);
            return;
        }
    }
    
    // 2. Create FreeRTOS Queue for core-to-core pointer communication
    coord_queue = xQueueCreate(5, sizeof(float *));
    if (coord_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create queue!");
        return;
    }
    
    // 3. Create core-pinned tasks
    // Pinned to Core 0 (PRO_CPU) for UART IO
    xTaskCreatePinnedToCore(
        uart_recv_test_task,
        "uart_recv_test_task",
        4096,
        NULL,
        5,
        NULL,
        0
    );
    
    // Pinned to Core 1 (APP_CPU) for heavy neural network calculations
    xTaskCreatePinnedToCore(
        ai_inference_task,
        "ai_inference_task",
        8192,
        NULL,
        5,
        NULL,
        1
    );
}