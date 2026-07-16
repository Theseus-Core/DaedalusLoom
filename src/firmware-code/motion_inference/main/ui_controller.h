#ifndef UI_CONTROLLER_H
#define UI_CONTROLLER_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化 EC11 编码器、SSD1306 OLED 屏幕，并创建 UI 控制后台任务
 */
esp_err_t ui_controller_init(void);

#ifdef __cplusplus
}
#endif

#endif // UI_CONTROLLER_H
