#include "gesture_detector.h"
#include <string.h>

void gesture_detector_init(gesture_detector_t *detector)
{
    gesture_detector_reset(detector);
}

void gesture_detector_reset(gesture_detector_t *detector)
{
    memset(detector->accumulated_probs, 0, sizeof(detector->accumulated_probs));
    memset(detector->unweighted_probs, 0, sizeof(detector->unweighted_probs));
    detector->class_count = nn_model_get_class_count();
    if (detector->class_count <= 0 || detector->class_count > NN_MODEL_MAX_CLASS_COUNT) {
        detector->class_count = 0;
    }
    detector->total_weight = 0.0f;
    detector->gesture_frames_count = 0;
    detector->pred_history_count = 0;
    memset(detector->pred_history, 0, sizeof(detector->pred_history));
}

void gesture_detector_accumulate(gesture_detector_t *detector, 
                                 const float *probs, 
                                 float motion_val, 
                                 const sys_config_t *config,
                                 int *out_smoothed_class,
                                 float *out_confidence)
{
    int class_count = nn_model_get_class_count();
    if (class_count <= 0 || class_count > NN_MODEL_MAX_CLASS_COUNT) {
        *out_smoothed_class = -1;
        *out_confidence = 0.0f;
        return;
    }
    if (detector->class_count != class_count) {
        gesture_detector_reset(detector);
    }

    // 寻找当前帧最大概率对应的类别和置信度
    int pred_class = 0;
    float confidence = probs[0];
    for (int c = 1; c < class_count; c++) {
        if (probs[c] > confidence) {
            confidence = probs[c];
            pred_class = c;
        }
    }
    
    *out_confidence = confidence;
    
    // 计算加权系数 (motion_val - threshold)^2
    float weight = (motion_val - config->motion_threshold) * (motion_val - config->motion_threshold);
    
    // 过滤置信度低于 0.50 的帧，仅让大于等于 0.50 的帧参与总决策累加
    if (confidence >= 0.50f) {
        for (int c = 0; c < class_count; c++) {
            detector->accumulated_probs[c] += probs[c] * weight;
            detector->unweighted_probs[c] += probs[c];
        }
        detector->total_weight += weight;
        detector->gesture_frames_count++;
    }
    
    // 根据置信度阈值推入平滑历史
    if (confidence >= config->min_confidence) {
        if (detector->pred_history_count < GESTURE_SMOOTHING_FRAMES) {
            detector->pred_history[detector->pred_history_count++] = pred_class;
        } else {
            for (int i = 0; i < GESTURE_SMOOTHING_FRAMES - 1; i++) {
                detector->pred_history[i] = detector->pred_history[i + 1];
            }
            detector->pred_history[GESTURE_SMOOTHING_FRAMES - 1] = pred_class;
        }
    }
    
    // 根据历史记录投票
    int smoothed_class = -1;
    if (detector->pred_history_count > 0) {
        int counts[NN_MODEL_MAX_CLASS_COUNT] = {0};
        for (int i = 0; i < detector->pred_history_count; i++) {
            int history_class = detector->pred_history[i];
            if (history_class >= 0 && history_class < class_count) {
                counts[history_class]++;
            }
        }
        int max_count = 0;
        for (int c = 0; c < class_count; c++) {
            if (counts[c] > max_count) {
                max_count = counts[c];
                smoothed_class = c;
            }
        }
    }
    *out_smoothed_class = smoothed_class;
}

bool gesture_detector_get_final(const gesture_detector_t *detector,
                                 int *out_final_class,
                                 float *out_final_prob,
                                 float *out_probs)
{
    if (detector->gesture_frames_count <= 0) {
        return false;
    }

    int class_count = detector->class_count;
    if (class_count <= 0 || class_count > NN_MODEL_MAX_CLASS_COUNT) {
        class_count = nn_model_get_class_count();
    }
    if (class_count <= 0 || class_count > NN_MODEL_MAX_CLASS_COUNT) {
        return false;
    }
    
    float final_probs[NN_MODEL_MAX_CLASS_COUNT] = {0.0f};
    if (detector->total_weight > 0.0f) {
        for (int c = 0; c < class_count; c++) {
            final_probs[c] = detector->accumulated_probs[c] / detector->total_weight;
        }
    } else {
        for (int c = 0; c < class_count; c++) {
            final_probs[c] = detector->unweighted_probs[c] / (float)detector->gesture_frames_count;
        }
    }
    
    // argmax
    int final_class = 0;
    float max_prob = final_probs[0];
    for (int c = 1; c < class_count; c++) {
        if (final_probs[c] > max_prob) {
            max_prob = final_probs[c];
            final_class = c;
        }
    }
    
    *out_final_class = final_class;
    *out_final_prob = max_prob;
    
    if (out_probs) {
        memcpy(out_probs, final_probs, NN_MODEL_MAX_CLASS_COUNT * sizeof(float));
    }
    return true;
}
