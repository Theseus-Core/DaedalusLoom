#include "sys_config.h"

#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nn_model.h"

#define MOTION_THRESHOLD_MIN_X10 10
#define MOTION_THRESHOLD_MAX_X10 100
#define MIN_CONFIDENCE_MIN_PCT 50
#define MIN_CONFIDENCE_MAX_PCT 99
#define REQUIRED_TRIGGER_MIN_FRAMES 1
#define REQUIRED_TRIGGER_MAX_FRAMES 20
#define DEBOUNCE_MIN_FRAMES 1
#define DEBOUNCE_MAX_FRAMES 50
#define MODEL_ID_MIN 0
#define MODEL_ID_MAX (NN_MODEL_MAX_SLOT_COUNT - 1)

typedef struct {
    const char *label;
    const char *unit;
} param_meta_t;

static const param_meta_t g_param_meta[SYS_CONFIG_PARAM_COUNT] = {
    [SYS_CONFIG_PARAM_MOTION_THRESHOLD] = {"Motion", "x0.1"},
    [SYS_CONFIG_PARAM_MIN_CONFIDENCE] = {"Conf", "%"},
    [SYS_CONFIG_PARAM_REQUIRED_TRIGGER_FRAMES] = {"Trig", "frm"},
    [SYS_CONFIG_PARAM_DEBOUNCE_FRAMES] = {"Idle", "frm"},
};

// 全局配置实例，默认值与重构前一致
static sys_config_t g_config = {
    .motion_threshold = 2.5f,
    .debounce_frames = 15,
    .required_trigger_frames = 3,
    .min_confidence = 0.80f,
    .active_model_id = 0,
};

// 用于多核保护的自旋锁
static portMUX_TYPE g_config_mux = portMUX_INITIALIZER_UNLOCKED;

static int clamp_int(int val, int min_val, int max_val)
{
    if (val < min_val) {
        return min_val;
    }
    if (val > max_val) {
        return max_val;
    }
    return val;
}

static float clamp_float(float val, float min_val, float max_val)
{
    if (val < min_val) {
        return min_val;
    }
    if (val > max_val) {
        return max_val;
    }
    return val;
}

static void normalize_config_locked(void)
{
    g_config.motion_threshold = clamp_float(g_config.motion_threshold,
                                            MOTION_THRESHOLD_MIN_X10 / 10.0f,
                                            MOTION_THRESHOLD_MAX_X10 / 10.0f);
    g_config.min_confidence = clamp_float(g_config.min_confidence,
                                          MIN_CONFIDENCE_MIN_PCT / 100.0f,
                                          MIN_CONFIDENCE_MAX_PCT / 100.0f);
    g_config.required_trigger_frames = clamp_int(g_config.required_trigger_frames,
                                                 REQUIRED_TRIGGER_MIN_FRAMES,
                                                 REQUIRED_TRIGGER_MAX_FRAMES);
    g_config.debounce_frames = clamp_int(g_config.debounce_frames,
                                         DEBOUNCE_MIN_FRAMES,
                                         DEBOUNCE_MAX_FRAMES);
    g_config.active_model_id = clamp_int(g_config.active_model_id, MODEL_ID_MIN, MODEL_ID_MAX);
}

void sys_config_init(void)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.motion_threshold = 2.5f;
    g_config.debounce_frames = 15;
    g_config.required_trigger_frames = 3;
    g_config.min_confidence = 0.80f;
    g_config.active_model_id = 0;
    normalize_config_locked();
    portEXIT_CRITICAL(&g_config_mux);
}

void sys_config_get(sys_config_t *out_config)
{
    if (out_config) {
        portENTER_CRITICAL(&g_config_mux);
        *out_config = g_config;
        portEXIT_CRITICAL(&g_config_mux);
    }
}

void sys_config_set(const sys_config_t *in_config)
{
    if (in_config) {
        portENTER_CRITICAL(&g_config_mux);
        g_config = *in_config;
        normalize_config_locked();
        portEXIT_CRITICAL(&g_config_mux);
    }
}

float sys_config_get_motion_threshold(void)
{
    float val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.motion_threshold;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_motion_threshold(float val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.motion_threshold = clamp_float(val,
                                            MOTION_THRESHOLD_MIN_X10 / 10.0f,
                                            MOTION_THRESHOLD_MAX_X10 / 10.0f);
    portEXIT_CRITICAL(&g_config_mux);
}

int sys_config_get_debounce_frames(void)
{
    int val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.debounce_frames;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_debounce_frames(int val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.debounce_frames = clamp_int(val, DEBOUNCE_MIN_FRAMES, DEBOUNCE_MAX_FRAMES);
    portEXIT_CRITICAL(&g_config_mux);
}

int sys_config_get_required_trigger_frames(void)
{
    int val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.required_trigger_frames;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_required_trigger_frames(int val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.required_trigger_frames = clamp_int(val,
                                                 REQUIRED_TRIGGER_MIN_FRAMES,
                                                 REQUIRED_TRIGGER_MAX_FRAMES);
    portEXIT_CRITICAL(&g_config_mux);
}

float sys_config_get_min_confidence(void)
{
    float val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.min_confidence;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_min_confidence(float val)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.min_confidence = clamp_float(val,
                                          MIN_CONFIDENCE_MIN_PCT / 100.0f,
                                          MIN_CONFIDENCE_MAX_PCT / 100.0f);
    portEXIT_CRITICAL(&g_config_mux);
}

int sys_config_get_active_model_id(void)
{
    int val;
    portENTER_CRITICAL(&g_config_mux);
    val = g_config.active_model_id;
    portEXIT_CRITICAL(&g_config_mux);
    return val;
}

void sys_config_set_active_model_id(int model_id)
{
    portENTER_CRITICAL(&g_config_mux);
    g_config.active_model_id = clamp_int(model_id, MODEL_ID_MIN, MODEL_ID_MAX);
    portEXIT_CRITICAL(&g_config_mux);
}

esp_err_t sys_config_adjust_param(sys_config_param_id_t param_id, int delta_display_units)
{
    if (param_id < 0 || param_id >= SYS_CONFIG_PARAM_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }

    portENTER_CRITICAL(&g_config_mux);
    switch (param_id) {
    case SYS_CONFIG_PARAM_MOTION_THRESHOLD: {
        int display = (int)lroundf(g_config.motion_threshold * 10.0f);
        display = clamp_int(display + delta_display_units,
                            MOTION_THRESHOLD_MIN_X10,
                            MOTION_THRESHOLD_MAX_X10);
        g_config.motion_threshold = display / 10.0f;
        break;
    }
    case SYS_CONFIG_PARAM_MIN_CONFIDENCE: {
        int display = (int)lroundf(g_config.min_confidence * 100.0f);
        display = clamp_int(display + delta_display_units,
                            MIN_CONFIDENCE_MIN_PCT,
                            MIN_CONFIDENCE_MAX_PCT);
        g_config.min_confidence = display / 100.0f;
        break;
    }
    case SYS_CONFIG_PARAM_REQUIRED_TRIGGER_FRAMES:
        g_config.required_trigger_frames = clamp_int(g_config.required_trigger_frames + delta_display_units,
                                                     REQUIRED_TRIGGER_MIN_FRAMES,
                                                     REQUIRED_TRIGGER_MAX_FRAMES);
        break;
    case SYS_CONFIG_PARAM_DEBOUNCE_FRAMES:
        g_config.debounce_frames = clamp_int(g_config.debounce_frames + delta_display_units,
                                             DEBOUNCE_MIN_FRAMES,
                                             DEBOUNCE_MAX_FRAMES);
        break;
    default:
        portEXIT_CRITICAL(&g_config_mux);
        return ESP_ERR_INVALID_ARG;
    }
    portEXIT_CRITICAL(&g_config_mux);
    return ESP_OK;
}

int sys_config_get_param_display_value(sys_config_param_id_t param_id)
{
    int display = 0;
    portENTER_CRITICAL(&g_config_mux);
    switch (param_id) {
    case SYS_CONFIG_PARAM_MOTION_THRESHOLD:
        display = (int)lroundf(g_config.motion_threshold * 10.0f);
        break;
    case SYS_CONFIG_PARAM_MIN_CONFIDENCE:
        display = (int)lroundf(g_config.min_confidence * 100.0f);
        break;
    case SYS_CONFIG_PARAM_REQUIRED_TRIGGER_FRAMES:
        display = g_config.required_trigger_frames;
        break;
    case SYS_CONFIG_PARAM_DEBOUNCE_FRAMES:
        display = g_config.debounce_frames;
        break;
    default:
        display = 0;
        break;
    }
    portEXIT_CRITICAL(&g_config_mux);
    return display;
}

const char *sys_config_get_param_label(sys_config_param_id_t param_id)
{
    if (param_id < 0 || param_id >= SYS_CONFIG_PARAM_COUNT) {
        return "Invalid";
    }
    return g_param_meta[param_id].label;
}

const char *sys_config_get_param_unit(sys_config_param_id_t param_id)
{
    if (param_id < 0 || param_id >= SYS_CONFIG_PARAM_COUNT) {
        return "";
    }
    return g_param_meta[param_id].unit;
}
