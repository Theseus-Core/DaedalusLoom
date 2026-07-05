#ifndef GESTURE_DETECTOR_H
#define GESTURE_DETECTOR_H

#include <stdbool.h>
#include "sys_config.h"
#include "nn_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GESTURE_SMOOTHING_FRAMES 7

typedef struct {
    float accumulated_probs[NN_MODEL_MAX_CLASS_COUNT];
    float unweighted_probs[NN_MODEL_MAX_CLASS_COUNT];
    int class_count;
    float total_weight;
    int gesture_frames_count;
    
    int pred_history[GESTURE_SMOOTHING_FRAMES];
    int pred_history_count;
} gesture_detector_t;

/**
 * @brief 初始化动作检测器状态
 */
void gesture_detector_init(gesture_detector_t *detector);

/**
 * @brief 重置动作检测器状态以开始新动作的统计
 */
void gesture_detector_reset(gesture_detector_t *detector);

/**
 * @brief 累加当前帧的概率分布并进行消抖平滑
 * 
 * @param detector 检测器指针
 * @param probs 当前帧的概率数组
 * @param motion_val 当前帧的运动强度值
 * @param config 当前配置快照
 * @param out_smoothed_class 输出实时投票滤波后的预测类别，若无法确定则返回 -1
 * @param out_confidence 输出实时最高置信度
 */
void gesture_detector_accumulate(gesture_detector_t *detector, 
                                 const float *probs, 
                                 float motion_val, 
                                 const sys_config_t *config,
                                 int *out_smoothed_class,
                                 float *out_confidence);

/**
 * @brief 获取当前动作周期的加权总判决结果
 * 
 * @param detector 检测器指针
 * @param out_final_class 输出判决结果类别
 * @param out_final_prob 输出判决结果的加权概率值
 * @param out_probs 输出所有类别的加权后概率数组 (大小需为 NN_MODEL_MAX_CLASS_COUNT，可传 NULL)
 * @return true 判决成功（有高置信度帧积累）
 * @return false 判决失败（无积累帧）
 */
bool gesture_detector_get_final(const gesture_detector_t *detector,
                                 int *out_final_class,
                                 float *out_final_prob,
                                 float *out_probs);

#ifdef __cplusplus
}
#endif

#endif // GESTURE_DETECTOR_H
