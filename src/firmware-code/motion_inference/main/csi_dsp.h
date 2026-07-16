#ifndef CSI_DSP_H
#define CSI_DSP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_DSP_NUM_FRAMES 50
#define CSI_DSP_NUM_SUBCARRIERS 114
#define CSI_DSP_FUSION_CHANNELS 228
#define CSI_DSP_TOTAL_FEATURES (CSI_DSP_NUM_FRAMES * CSI_DSP_FUSION_CHANNELS)

/**
 * @brief 解析原始CSI载荷并填充当前帧的228个双通道数据 (I/Q)
 */
void csi_dsp_parse_frame(const int8_t *raw_payload, size_t raw_len, float *out_frame_228);

/**
 * @brief 计算给定滑动窗口的平均运动水平强度
 */
float csi_dsp_calculate_motion(const float *window_buf);

/**
 * @brief 对输入的原始滑动窗口数据进行完整的预处理（幅度和相位提取、相位展开、线性校准、SR-Std标准化以及融合）
 * 
 * @param raw_csi 长度为 50 * 228 = 11400 的浮点数输入窗口
 * @param out_features 长度为 11400 的输出预处理特征数组
 */
void csi_dsp_preprocess_fusion(const float *raw_csi, float *out_features);

#ifdef __cplusplus
}
#endif

#endif // CSI_DSP_H
