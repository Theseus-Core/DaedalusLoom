#ifndef GESTURE_OUTPUT_H
#define GESTURE_OUTPUT_H

#include <stdint.h>
#include <stdbool.h>
#include "nn_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int class_id;
    int class_count;
    const char *class_name;
    float confidence;
    bool is_realtime; // true 代表每帧实时预测，false 代表动作结束的最终判定结果
    
    // 最终判决时的概率与帧统计
    float final_probs[NN_MODEL_MAX_CLASS_COUNT];
    int gesture_frames_count;
    float total_weight;
} gesture_result_t;

typedef void (*gesture_output_callback_t)(const gesture_result_t *result, void *user_data);

/**
 * @brief 初始化输出分发模块（创建队列和异步任务）
 */
void gesture_output_init(void);

/**
 * @brief 注册输出通道观察者（例如 OLED 刷新、HTTP 发送任务）
 * 
 * @param callback 观察者回调函数
 * @param user_data 用户私有数据指针
 * @return true 注册成功
 * @return false 注册满或失败
 */
bool gesture_output_register_observer(gesture_output_callback_t callback, void *user_data);

/**
 * @brief 非阻塞地分发推理结果给所有观察者
 * 
 * @param result 推理结果指针
 */
void gesture_output_dispatch(const gesture_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // GESTURE_OUTPUT_H
