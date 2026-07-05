#ifndef SYS_CONFIG_H
#define SYS_CONFIG_H

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SYS_CONFIG_PARAM_MOTION_THRESHOLD = 0,
    SYS_CONFIG_PARAM_MIN_CONFIDENCE,
    SYS_CONFIG_PARAM_REQUIRED_TRIGGER_FRAMES,
    SYS_CONFIG_PARAM_DEBOUNCE_FRAMES,
    SYS_CONFIG_PARAM_GESTURE_COOLDOWN,
    SYS_CONFIG_PARAM_COUNT
} sys_config_param_id_t;

// 系统配置参数结构体
typedef struct {
    float motion_threshold;        // 运动标准差触发阈值
    int debounce_frames;           // 运动结束消抖帧数
    int required_trigger_frames;   // 运动开始确认帧数
    float min_confidence;          // 实时预测最小置信度过滤阈值
    float gesture_cooldown_sec;    // 动作结束后的下一次识别冷却时间（秒）
    int active_model_id;           // 当前启用的模型槽位 ID
} sys_config_t;

/**
 * @brief 初始化配置模块（设置默认值）
 */
void sys_config_init(void);

/**
 * @brief 获取完整的系统配置结构体（线程安全）
 * 
 * @param out_config 输出配置结构体指针
 */
void sys_config_get(sys_config_t *out_config);

/**
 * @brief 设置完整的系统配置结构体（线程安全）
 * 
 * @param in_config 输入配置结构体指针
 */
void sys_config_set(const sys_config_t *in_config);

/**
 * @brief 获取当前运动阈值
 */
float sys_config_get_motion_threshold(void);

/**
 * @brief 设置新运动阈值
 */
void sys_config_set_motion_threshold(float val);

/**
 * @brief 获取运动结束消抖帧数
 */
int sys_config_get_debounce_frames(void);

/**
 * @brief 设置运动结束消抖帧数
 */
void sys_config_set_debounce_frames(int val);

/**
 * @brief 获取运动开始确认帧数
 */
int sys_config_get_required_trigger_frames(void);

/**
 * @brief 设置运动开始确认帧数
 */
void sys_config_set_required_trigger_frames(int val);

/**
 * @brief 获取实时预测过滤的最小置信度阈值
 */
float sys_config_get_min_confidence(void);

/**
 * @brief 设置实时预测过滤的最小置信度阈值
 */
void sys_config_set_min_confidence(float val);

/**
 * @brief 获取动作识别冷却时间（秒）
 */
float sys_config_get_gesture_cooldown_sec(void);

/**
 * @brief 设置动作识别冷却时间（秒）
 */
void sys_config_set_gesture_cooldown_sec(float val);

/**
 * @brief 获取当前启用的模型槽位 ID
 */
int sys_config_get_active_model_id(void);

/**
 * @brief 设置当前启用的模型槽位 ID
 */
void sys_config_set_active_model_id(int model_id);

/**
 * @brief 按 OLED 显示单位调整某个参数，内部负责单位转换与范围裁剪
 */
esp_err_t sys_config_adjust_param(sys_config_param_id_t param_id, int delta_display_units);

/**
 * @brief 获取参数的整数显示值
 */
int sys_config_get_param_display_value(sys_config_param_id_t param_id);

/**
 * @brief 获取参数显示名
 */
const char *sys_config_get_param_label(sys_config_param_id_t param_id);

/**
 * @brief 获取参数显示单位
 */
const char *sys_config_get_param_unit(sys_config_param_id_t param_id);

#ifdef __cplusplus
}
#endif

#endif // SYS_CONFIG_H
