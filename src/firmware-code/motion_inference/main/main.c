#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "sys_config.h"
#include "csi_source.h"
#include "csi_dsp.h"
#include "gesture_detector.h"
#include "gesture_output.h"
#include "nn_model.h"
#include "ui_controller.h"
#include "wifi_manager.h"
#include "upload_api.h"
#include "sdkconfig.h"

static const char *TAG = "main";

// ==================== 1. 并发队列及 Job 定义 ====================
typedef struct {
    float *buffer;
    float motion_val;
    bool is_ended;
} inference_job_t;

static QueueHandle_t xEmptyBufQueue = NULL;
static QueueHandle_t xReadyInferenceQueue = NULL;

#define NUM_BUFFERS 3

static int gesture_severity_from_name(const char *class_name)
{
    if (!class_name) {
        return 1;
    }

    if (strcmp(class_name, "fall_down") == 0 ||
        strcmp(class_name, "跌倒") == 0 ||
        strcmp(class_name, "wave_hand") == 0 ||
        strcmp(class_name, "举手呼叫") == 0) {
        return 5;
    }

    if (strcmp(class_name, "waveup") == 0 ||
        strcmp(class_name, "起身") == 0) {
        return 3;
    }

    if (strcmp(class_name, "roll") == 0 ||
        strcmp(class_name, "翻身") == 0) {
        return 1;
    }

    return 1;
}

static void sent_web_observer(const gesture_result_t *result, void *user_data)
{
    if (!result->is_realtime) {
        char url[128];
        char json_buffer[256];  
        int severity = gesture_severity_from_name(result->class_name);
        
        // 构造URL
        snprintf(url, sizeof(url), "%s/api/sensor/signal", CONFIG_ddloom_base_url);
        
        char description[128];
        int patient_id = 1;
        if (result->is_multi_branch) {
            snprintf(description, sizeof(description), "动作结束 (人员: %s)", result->person_class_name);
            char *endptr;
            long val = strtol(result->person_class_name, &endptr, 10);
            if (endptr != result->person_class_name && *endptr == '\0') {
                patient_id = (int)val;
            }
        } else {
            snprintf(description, sizeof(description), "动作结束");
        }

        // 构造JSON - 注意true/false不加引号
        snprintf(json_buffer, sizeof(json_buffer),
                 "{\"patientId\": %d, \"behaviorType\": \"%s\", "
                 "\"description\": \"%s\", \"isAbnormal\": %s, "
                 "\"severity\": %d}",
                 patient_id,  // patientId
                 result->class_name,
                 description,
                 "true",  // 
                 severity);
        
        // 发送数据
        upload_info(url, json_buffer);
    }
}


// ==================== 2. 默认控制台日志观察者 ====================
static void console_log_observer(const gesture_result_t *result, void *user_data)
{
    if (result->is_realtime) {
        if (result->is_multi_branch) {
            ESP_LOGI(TAG, " [ACTIVE] Real-time Pred - Action: %s (Conf: %.1f%%) | Person: %s (Conf: %.1f%%)", 
                     result->class_name, result->confidence * 100.0f,
                     result->person_class_name, result->person_confidence * 100.0f);
        } else {
            if (result->class_id != -1) {
                ESP_LOGI(TAG, " [ACTIVE] Real-time Pred: %s (Conf: %.1f%%)", 
                         result->class_name, result->confidence * 100.0f);
            } else {
                ESP_LOGI(TAG, " [ACTIVE] Real-time Pred: UNCERTAIN");
            }
        }
    } else {
        ESP_LOGI(TAG, "================ GESTURE DETECTED ================");
        if (result->is_multi_branch) {
            ESP_LOGI(TAG, "Final Result - Action: %s (Score: %.1f%%) | Person: %s (Score: %.1f%%)", 
                     result->class_name, result->confidence * 100.0f,
                     result->person_class_name, result->person_confidence * 100.0f);
            ESP_LOGI(TAG, "      [Action Probabilities]");
            for (int c = 0; c < result->class_count; c++) {
                const char *name = nn_model_get_action_class_name(c);
                ESP_LOGI(TAG, "      - Class %d (%s) Prob: %.2f%%", c, name, result->final_probs[c] * 100.0f);
            }
            ESP_LOGI(TAG, "      [Person Probabilities]");
            for (int c = 0; c < result->person_class_count; c++) {
                const char *name = nn_model_get_person_class_name(c);
                ESP_LOGI(TAG, "      - Class %d (%s) Prob: %.2f%%", c, name, result->person_final_probs[c] * 100.0f);
            }
        } else {
            ESP_LOGI(TAG, " Final Result: %s (Score: %.1f%%)", result->class_name, result->confidence * 100.0f);
            for (int c = 0; c < result->class_count; c++) {
                const char *name = nn_model_get_class_name(c);
                ESP_LOGI(TAG, "      - Class %d (%s) Prob: %.2f%%", c, name, result->final_probs[c] * 100.0f);
            }
        }
        ESP_LOGI(TAG, "      - Accum Frames: %d, Total Weight: %.2f", result->gesture_frames_count, result->total_weight);
        ESP_LOGI(TAG, "==================================================");
    }
}

// ==================== 3. 核心任务定义 ====================

/**
 * @brief Core 1 任务: AI模型推理
 */
static void ai_inference_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting AI Inference Task on Core %d...", xPortGetCoreID());
    
    // 初始化神经网络模型（从Flash/RODATA加载）
    nn_model_init();
    
    gesture_detector_t action_detector;
    gesture_detector_t person_detector;
    
    // We will initialize them dynamically when the model ID changes.
    int last_model_id = -1;
    bool current_is_multi_branch = false;
    
    float action_probs[NN_MODEL_MAX_CLASS_COUNT] = {0.0f};
    float person_probs[NN_MODEL_MAX_CLASS_COUNT] = {0.0f};
    inference_job_t job;
    
    while (1) {
        if (xQueueReceive(xReadyInferenceQueue, &job, portMAX_DELAY) == pdTRUE) {
            int active_model_id = nn_model_get_active_id();
            if (active_model_id != last_model_id) {
                last_model_id = active_model_id;
                current_is_multi_branch = nn_model_is_multi_branch();
                gesture_detector_init(&action_detector, current_is_multi_branch ? nn_model_get_action_class_count() : nn_model_get_class_count());
                gesture_detector_init(&person_detector, current_is_multi_branch ? nn_model_get_person_class_count() : 0);
            }

            if (job.is_ended) {
                // 动作结束，进行加权总决策
                int final_action_class = 0;
                float final_action_prob = 0.0f;
                float final_action_probs[NN_MODEL_MAX_CLASS_COUNT] = {0.0f};
                
                int final_person_class = 0;
                float final_person_prob = 0.0f;
                float final_person_probs[NN_MODEL_MAX_CLASS_COUNT] = {0.0f};
                
                bool action_ok = gesture_detector_get_final(&action_detector, &final_action_class, &final_action_prob, final_action_probs);
                bool person_ok = false;
                if (current_is_multi_branch) {
                    person_ok = gesture_detector_get_final(&person_detector, &final_person_class, &final_person_prob, final_person_probs);
                }
                
                if (action_ok) {
                    gesture_result_t result = {
                        .is_multi_branch = current_is_multi_branch,
                        .class_id = final_action_class,
                        .class_count = current_is_multi_branch ? nn_model_get_action_class_count() : nn_model_get_class_count(),
                        .class_name = current_is_multi_branch ? nn_model_get_action_class_name(final_action_class) : nn_model_get_class_name(final_action_class),
                        .confidence = final_action_prob,
                        
                        .person_class_id = current_is_multi_branch ? final_person_class : -1,
                        .person_class_count = current_is_multi_branch ? nn_model_get_person_class_count() : 0,
                        .person_class_name = (current_is_multi_branch && person_ok) ? nn_model_get_person_class_name(final_person_class) : "Unknown",
                        .person_confidence = current_is_multi_branch ? final_person_prob : 0.0f,
                        
                        .is_realtime = false,
                        .gesture_frames_count = action_detector.gesture_frames_count,
                        .total_weight = action_detector.total_weight
                    };
                    memcpy(result.final_probs, final_action_probs, sizeof(final_action_probs));
                    if (current_is_multi_branch) {
                        memcpy(result.person_final_probs, final_person_probs, sizeof(final_person_probs));
                    }
                    gesture_output_dispatch(&result);
                } else {
                    ESP_LOGI(TAG, "================ GESTURE DETECTED ================");
                    ESP_LOGI(TAG, "Finished, but no high-confidence frames accumulated.");
                    ESP_LOGI(TAG, "==================================================");
                }
                
                // 重置累加状态与平滑历史
                gesture_detector_reset(&action_detector, current_is_multi_branch ? nn_model_get_action_class_count() : nn_model_get_class_count());
                if (current_is_multi_branch) {
                    gesture_detector_reset(&person_detector, nn_model_get_person_class_count());
                }
            } else {
                // 执行 AI 推理
                memset(action_probs, 0, sizeof(action_probs));
                memset(person_probs, 0, sizeof(person_probs));
                int pred_action_class = -1;
                int pred_person_class = -1;
                
                if (current_is_multi_branch) {
                    pred_action_class = nn_model_predict_cnn_multi(job.buffer, action_probs, person_probs, &pred_person_class);
                } else {
                    pred_action_class = nn_model_predict_cnn_with_probs(job.buffer, action_probs);
                }
                
                // 立即将使用的缓冲区归还到空闲队列，缩短被占用的时间
                float *buf_ptr = job.buffer;
                xQueueSend(xEmptyBufQueue, &buf_ptr, 0);
                
                sys_config_t config;
                sys_config_get(&config);
                
                int smoothed_action_class = -1;
                float action_confidence = 0.0f;
                gesture_detector_accumulate(&action_detector, action_probs, job.motion_val, &config, &smoothed_action_class, &action_confidence);
                
                int smoothed_person_class = -1;
                float person_confidence = 0.0f;
                if (current_is_multi_branch) {
                    gesture_detector_accumulate(&person_detector, person_probs, job.motion_val, &config, &smoothed_person_class, &person_confidence);
                }
                
                // 实时分发推理结果
                gesture_result_t result = {
                    .is_multi_branch = current_is_multi_branch,
                    .class_id = smoothed_action_class,
                    .class_count = current_is_multi_branch ? nn_model_get_action_class_count() : nn_model_get_class_count(),
                    .class_name = current_is_multi_branch ? nn_model_get_action_class_name(smoothed_action_class) : nn_model_get_class_name(smoothed_action_class),
                    .confidence = action_confidence,
                    
                    .person_class_id = current_is_multi_branch ? smoothed_person_class : -1,
                    .person_class_count = current_is_multi_branch ? nn_model_get_person_class_count() : 0,
                    .person_class_name = current_is_multi_branch ? nn_model_get_person_class_name(smoothed_person_class) : "Unknown",
                    .person_confidence = person_confidence,
                    
                    .is_realtime = true
                };
                gesture_output_dispatch(&result);
            }
        }
    }
}

/**
 * @brief Core 0 任务: UART 接收和运动监测状态机
 */
static void csi_uart_rx_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting CSI UART RX Task on Core %d...", xPortGetCoreID());
    
    csi_frame_t *frame = (csi_frame_t *)malloc(sizeof(csi_frame_t));
    if (!frame) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer on heap!");
        vTaskDelete(NULL);
        return;
    }
    static float new_frame[CSI_DSP_FUSION_CHANNELS];
    
    // 动态分配 Core 0 本地的滑动窗口
    float *csi_window = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!csi_window) {
        ESP_LOGE(TAG, "Failed to allocate local CSI window buffer in PSRAM!");
        free(frame);
        vTaskDelete(NULL);
        return;
    }
    memset(csi_window, 0, CSI_DSP_TOTAL_FEATURES * sizeof(float));
    int csi_window_count = 0;
    
    bool is_moving = false;
    int motion_trigger_counter = 0;
    int idle_counter = 0;
    TickType_t cooldown_until = 0;
    
    while (true) {
        if (csi_source_read_frame(frame)) {
            // 解析并对齐子载波
            csi_dsp_parse_frame(frame->payload, frame->raw_len, new_frame);
            
            // 写入滑动窗口
            if (csi_window_count < CSI_DSP_NUM_FRAMES) {
                memcpy(csi_window + csi_window_count * CSI_DSP_FUSION_CHANNELS, new_frame, sizeof(new_frame));
                csi_window_count++;
            } else {
                // 滑动左移1帧，并将新帧追加在尾部
                memmove(csi_window, csi_window + CSI_DSP_FUSION_CHANNELS, (CSI_DSP_NUM_FRAMES - 1) * CSI_DSP_FUSION_CHANNELS * sizeof(float));
                memcpy(csi_window + (CSI_DSP_NUM_FRAMES - 1) * CSI_DSP_FUSION_CHANNELS, new_frame, sizeof(new_frame));
            }
            
            // 当滑动窗口攒满50帧后，触发计算标准差与状态机
            if (csi_window_count == CSI_DSP_NUM_FRAMES) {
                float motion_val = csi_dsp_calculate_motion(csi_window);
                
                sys_config_t config;
                sys_config_get(&config);
                TickType_t now = xTaskGetTickCount();
                bool cooldown_active = (cooldown_until != 0) &&
                                       ((int32_t)(cooldown_until - now) > 0);
                if (!cooldown_active) {
                    cooldown_until = 0;
                }
                
                if (cooldown_active) {
                    is_moving = false;
                    motion_trigger_counter = 0;
                    idle_counter = 0;
                } else if (motion_val >= config.motion_threshold) {
                    idle_counter = 0;
                    if (!is_moving) {
                        motion_trigger_counter++;
                        if (motion_trigger_counter >= config.required_trigger_frames) {
                            is_moving = true;
                            ESP_LOGI(TAG, "[MOTION DETECTED] Gesture started! (Motion Level: %.2f)", motion_val);
                        }
                    }
                    
                    if (is_moving) {
                        // 利用缓冲池零拷贝安全将数据传给 Core 1 推理，不会产生大拷贝锁死中断的问题
                        float *job_buf = NULL;
                        if (xQueueReceive(xEmptyBufQueue, &job_buf, 0) == pdTRUE) {
                            memcpy(job_buf, csi_window, CSI_DSP_TOTAL_FEATURES * sizeof(float));
                            inference_job_t job = {
                                .buffer = job_buf,
                                .motion_val = motion_val,
                                .is_ended = false
                            };
                            if (xQueueSend(xReadyInferenceQueue, &job, 0) != pdPASS) {
                                // 队列已满则将缓冲区送回空闲池，防泄漏
                                xQueueSend(xEmptyBufQueue, &job_buf, 0);
                            }
                        } else {
                            ESP_LOGD(TAG, "Buffer pool empty, dropping frame!");
                        }
                    }
                } else {
                    if (is_moving) {
                        motion_trigger_counter = 0;
                        idle_counter++;
                        
                        if (idle_counter >= config.debounce_frames) {
                            is_moving = false;
                            
                            // 通知 Core 1 动作结束
                            inference_job_t job = {
                                .buffer = NULL,
                                .motion_val = motion_val,
                                .is_ended = true
                            };
                            xQueueSend(xReadyInferenceQueue, &job, portMAX_DELAY);
                            if (config.gesture_cooldown_sec > 0.0f) {
                                uint32_t cooldown_ms = (uint32_t)(config.gesture_cooldown_sec * 1000.0f);
                                cooldown_until = xTaskGetTickCount() + pdMS_TO_TICKS(cooldown_ms);
                            } else {
                                cooldown_until = 0;
                            }
                            
                            ESP_LOGI(TAG, "[MOTION ENDED] Returning to standby.");
                        }
                    } else {
                        motion_trigger_counter = 0;
                    }
                }
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    
    free(csi_window);
    free(frame);
    vTaskDelete(NULL);
}

// ==================== 4. 主函数入口 ====================
void app_main(void)
{
    ESP_LOGI(TAG, "======= ESP32-P4 WiFi CSI Inference Startup =======");

    //  初始化阈值配置中心
    sys_config_init();

    //  初始化 UI 控制器 (OLED + 旋转编码器)
    if (ui_controller_init() != ESP_OK) {
        ESP_LOGE(TAG, "UI Controller Init failed!");
    }
    wifi_manager_init();
    
    //  初始化输出观察者分发系统，并注册控制台输出作为默认通道
    gesture_output_init();
    gesture_output_register_observer(console_log_observer, NULL);
    gesture_output_register_observer(sent_web_observer, NULL);
    //gesture_output_register_observer();
    //创建指针缓冲池队列（无拷贝多核交互）
    xEmptyBufQueue = xQueueCreate(NUM_BUFFERS, sizeof(float *));
    xReadyInferenceQueue = xQueueCreate(NUM_BUFFERS, sizeof(inference_job_t));
    if (!xEmptyBufQueue || !xReadyInferenceQueue) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS queues!");
        return;
    }


    // 初始化 Wi-Fi 连接管理器
    //wifi_manager_init();
    // 在 PSRAM 中分配缓冲池的各个块并加入队列
    for (int i = 0; i < NUM_BUFFERS; i++) {
        float *buf = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
        if (!buf) {
            ESP_LOGE(TAG, "Failed to allocate buffer pool block %d in PSRAM!", i);
            return;
        }
        memset(buf, 0, CSI_DSP_TOTAL_FEATURES * sizeof(float));
        xQueueSend(xEmptyBufQueue, &buf, portMAX_DELAY);
    }

    // 初始化 UART 串口接收驱动
    if (csi_source_init() != ESP_OK) {
        ESP_LOGE(TAG, "CSI Source Init failed! System halted.");
        return;
    }
    
    // 建核心任务并绑定 CPU
    // 绑定 Core 1 进行神经网络推理
    xTaskCreatePinnedToCore(
        ai_inference_task,
        "ai_inference_task",
        8192, // 增加栈空间以防推理/格式化输出时溢出
        NULL,
        5,
        NULL,
        1
    );

    // 绑定 Core 0 进行数据接收与运动检测
    xTaskCreatePinnedToCore(
        csi_uart_rx_task,
        "csi_uart_rx_task",
        4096, // 增加栈空间以防串口驱动/队列发送时溢出
        NULL,
        6, // 接收任务稍高优先级，保证不丢数据
        NULL,
        0
    );
}
