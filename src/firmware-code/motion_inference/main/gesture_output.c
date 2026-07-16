#include "gesture_output.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include <string.h>

#define TAG "gesture_output"
#define MAX_OBSERVERS 4
#define OUTPUT_QUEUE_LEN 16

typedef struct {
    gesture_output_callback_t callback;
    void *user_data;
} observer_entry_t;

static observer_entry_t g_observers[MAX_OBSERVERS];
static int g_observer_count = 0;
static SemaphoreHandle_t g_observer_mutex = NULL;
static QueueHandle_t g_output_queue = NULL;

static void gesture_output_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting Gesture Output Dispatcher Task on Core %d...", xPortGetCoreID());
    gesture_result_t result;
    
    while (1) {
        if (xQueueReceive(g_output_queue, &result, portMAX_DELAY) == pdTRUE) {
            if (xSemaphoreTake(g_observer_mutex, portMAX_DELAY) == pdTRUE) {
                for (int i = 0; i < g_observer_count; i++) {
                    if (g_observers[i].callback) {
                        g_observers[i].callback(&result, g_observers[i].user_data);
                    }
                }
                xSemaphoreGive(g_observer_mutex);
            }
        }
    }
}

void gesture_output_init(void)
{
    g_observer_mutex = xSemaphoreCreateMutex();
    g_output_queue = xQueueCreate(OUTPUT_QUEUE_LEN, sizeof(gesture_result_t));
    
    if (g_output_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create output queue!");
        return;
    }
    
    // 启动低优先级的异步分发任务，绑定到 Core 0
    xTaskCreatePinnedToCore(
        gesture_output_task,
        "gesture_output_task",
        4096,
        NULL,
        3, // 优先级低于数据接收(6)和AI推理(5)
        NULL,
        0
    );
}

bool gesture_output_register_observer(gesture_output_callback_t callback, void *user_data)
{
    if (g_observer_mutex == NULL) {
        return false;
    }
    
    bool ret = false;
    if (xSemaphoreTake(g_observer_mutex, portMAX_DELAY) == pdTRUE) {
        if (g_observer_count < MAX_OBSERVERS) {
            g_observers[g_observer_count].callback = callback;
            g_observers[g_observer_count].user_data = user_data;
            g_observer_count++;
            ret = true;
            ESP_LOGI(TAG, "Observer registered successfully (count=%d).", g_observer_count);
        } else {
            ESP_LOGE(TAG, "Failed to register observer: slot full!");
        }
        xSemaphoreGive(g_observer_mutex);
    }
    return ret;
}

void gesture_output_dispatch(const gesture_result_t *result)
{
    if (g_output_queue == NULL) {
        return;
    }
    
    // 非阻塞发送，防止卡死 AI 推理主流程
    if (xQueueSend(g_output_queue, result, 0) != pdPASS) {
        ESP_LOGW(TAG, "Output queue full! Dropping dispatch result.");
    }
}
