#ifndef CSI_SOURCE_H
#define CSI_SOURCE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_FRAME_MAX_RAW_LEN 1024

typedef struct {
    uint32_t seq;
    size_t raw_len;
    int8_t payload[CSI_FRAME_MAX_RAW_LEN];
} csi_frame_t;

/**
 * @brief 初始化 CSI 串口数据接收驱动
 */
esp_err_t csi_source_init(void);

/**
 * @brief 从串口读取并校验一帧 CSI 数据 (阻塞读取，有超时)
 * 
 * @param out_frame 指向接收帧的指针
 * @return true 成功读取并校验一帧
 * @return false 读取失败或校验失败
 */
bool csi_source_read_frame(csi_frame_t *out_frame);

#ifdef __cplusplus
}
#endif

#endif // CSI_SOURCE_H
