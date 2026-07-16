#include "csi_dsp.h"
#include "esp_heap_caps.h"
#include <cmath>
#include <cstring>

static const float CONST_D = 123452.5f;

static void unwrap_phase_frame(const float *p_in, float *p_out) {
    p_out[0] = p_in[0];
    float cum_shift = 0.0f;
    const float inv_two_pi = 1.0f / (2.0f * (float)M_PI);
    const float two_pi = 2.0f * (float)M_PI;
    
    for (int i = 1; i < CSI_DSP_NUM_SUBCARRIERS; i++) {
        float diff = p_in[i] - p_in[i-1];
        float shift = 0.0f;
        if (diff > (float)M_PI) {
            shift = -two_pi * (int)((diff + (float)M_PI) * inv_two_pi);
        } else if (diff < -(float)M_PI) {
            shift = two_pi * (int)((-diff + (float)M_PI) * inv_two_pi);
        }
        cum_shift += shift;
        p_out[i] = p_in[i] + cum_shift;
    }
}

void csi_dsp_parse_frame(const int8_t *raw_payload, size_t raw_len, float *out_frame_228)
{
    int num_in = raw_len / 2;
    int count = 0;
    for (int i = 0; i < num_in && count < CSI_DSP_NUM_SUBCARRIERS; i++) {
        int8_t imag = raw_payload[2 * i];
        int8_t real = raw_payload[2 * i + 1];
        if (real != 0 || imag != 0) {
            out_frame_228[2 * count] = (float)imag;
            out_frame_228[2 * count + 1] = (float)real;
            count++;
        }
    }
    
    if (count < CSI_DSP_NUM_SUBCARRIERS) {
        if (count > 0) {
            float last_imag = out_frame_228[2 * (count - 1)];
            float last_real = out_frame_228[2 * (count - 1) + 1];
            for (int i = count; i < CSI_DSP_NUM_SUBCARRIERS; i++) {
                out_frame_228[2 * i] = last_imag;
                out_frame_228[2 * i + 1] = last_real;
            }
        } else {
            memset(out_frame_228, 0, CSI_DSP_FUSION_CHANNELS * sizeof(float));
        }
    }
}

float csi_dsp_calculate_motion(const float *window_buf)
{
    float std_sum = 0.0f;
    for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
            float imag = window_buf[t * CSI_DSP_FUSION_CHANNELS + 2 * s];
            float real = window_buf[t * CSI_DSP_FUSION_CHANNELS + 2 * s + 1];
            float amp = sqrtf(real * real + imag * imag);
            sum += amp;
            sum_sq += amp * amp;
        }
        float mean = sum / (float)CSI_DSP_NUM_FRAMES;
        float var = (sum_sq / (float)CSI_DSP_NUM_FRAMES) - (mean * mean);
        if (var < 0.0f) {
            var = 0.0f;
        }
        float std = sqrtf(var);
        std_sum += std;
    }
    return std_sum / (float)CSI_DSP_NUM_SUBCARRIERS;
}

void csi_dsp_preprocess_fusion(const float *raw_csi, float *out_features)
{
    static float *amp = nullptr;
    static float *phase = nullptr;
    static float *unwrapped = nullptr;
    static float *phase_cal = nullptr;

    if (amp == nullptr) {
        amp = (float *)heap_caps_malloc(CSI_DSP_NUM_FRAMES * CSI_DSP_NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        phase = (float *)heap_caps_malloc(CSI_DSP_NUM_FRAMES * CSI_DSP_NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        unwrapped = (float *)heap_caps_malloc(CSI_DSP_NUM_FRAMES * CSI_DSP_NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
        phase_cal = (float *)heap_caps_malloc(CSI_DSP_NUM_FRAMES * CSI_DSP_NUM_SUBCARRIERS * sizeof(float), MALLOC_CAP_SPIRAM);
    }

    // 1. Extract Amplitude and Phase
    for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
        int t_offset = t * CSI_DSP_NUM_SUBCARRIERS;
        int t_fusion_offset = t * CSI_DSP_FUSION_CHANNELS;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            float imag = raw_csi[t_fusion_offset + 2 * s];
            float real = raw_csi[t_fusion_offset + 2 * s + 1];
            
            amp[t_offset + s] = sqrtf(real * real + imag * imag);
            phase[t_offset + s] = atan2f(imag, real);
        }
    }

    // 2. Phase Unwrapping and Linear Calibration (Frame by frame)
    for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
        float *p_in = &phase[t * CSI_DSP_NUM_SUBCARRIERS];
        float *p_unwrapped = &unwrapped[t * CSI_DSP_NUM_SUBCARRIERS];
        float *p_cal = &phase_cal[t * CSI_DSP_NUM_SUBCARRIERS];
        
        unwrap_phase_frame(p_in, p_unwrapped);
        
        float y_sum = 0.0f;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            y_sum += p_unwrapped[s];
        }
        float y_mean = y_sum * (1.0f / (float)CSI_DSP_NUM_SUBCARRIERS);
        
        float num_sum = 0.0f;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            float y_dev = p_unwrapped[s] - y_mean;
            float x_dev = (float)s - 56.5f;
            num_sum += y_dev * x_dev;
        }
        float a = num_sum * (1.0f / CONST_D);
        
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            float y_dev = p_unwrapped[s] - y_mean;
            float x_dev = (float)s - 56.5f;
            p_cal[s] = y_dev - a * x_dev;
        }
    }

    // 3. Standardization (SR-Std) over time dimension and Feature Fusion
    const float eps = 2.0f;
    
    static float amp_mean[CSI_DSP_NUM_SUBCARRIERS];
    static float phase_mean[CSI_DSP_NUM_SUBCARRIERS];
    static float amp_var[CSI_DSP_NUM_SUBCARRIERS];
    static float phase_var[CSI_DSP_NUM_SUBCARRIERS];
    
    memset(amp_mean, 0, sizeof(amp_mean));
    memset(phase_mean, 0, sizeof(phase_mean));
    memset(amp_var, 0, sizeof(amp_var));
    memset(phase_var, 0, sizeof(phase_var));
    
    // Sum over time
    for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
        int t_offset = t * CSI_DSP_NUM_SUBCARRIERS;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            amp_mean[s] += amp[t_offset + s];
            phase_mean[s] += phase_cal[t_offset + s];
        }
    }
    
    // Compute means
    const float inv_num_frames = 1.0f / (float)CSI_DSP_NUM_FRAMES;
    for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
        amp_mean[s] *= inv_num_frames;
        phase_mean[s] *= inv_num_frames;
    }
    
    // Sum variance
    for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
        int t_offset = t * CSI_DSP_NUM_SUBCARRIERS;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            float amp_diff = amp[t_offset + s] - amp_mean[s];
            amp_var[s] += amp_diff * amp_diff;
            
            float phase_diff = phase_cal[t_offset + s] - phase_mean[s];
            phase_var[s] += phase_diff * phase_diff;
        }
    }
    
    // Compute std and scales (reciprocals)
    static float amp_scale[CSI_DSP_NUM_SUBCARRIERS];
    static float phase_scale[CSI_DSP_NUM_SUBCARRIERS];
    for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
        float amp_std = sqrtf(amp_var[s] * inv_num_frames);
        float phase_std = sqrtf(phase_var[s] * inv_num_frames);
        amp_scale[s] = 1.0f / (amp_std + eps);
        phase_scale[s] = 1.0f / (phase_std + eps);
    }
    
    // Normalize and fuse (sequential access, cache friendly, multiplication only)
    for (int t = 0; t < CSI_DSP_NUM_FRAMES; t++) {
        int t_offset_in = t * CSI_DSP_NUM_SUBCARRIERS;
        int t_offset_out = t * CSI_DSP_FUSION_CHANNELS;
        for (int s = 0; s < CSI_DSP_NUM_SUBCARRIERS; s++) {
            float norm_amp = (amp[t_offset_in + s] - amp_mean[s]) * amp_scale[s];
            float norm_phase = (phase_cal[t_offset_in + s] - phase_mean[s]) * phase_scale[s];
            
            out_features[t_offset_out + s] = norm_amp;
            out_features[t_offset_out + s + CSI_DSP_NUM_SUBCARRIERS] = norm_phase;
        }
    }
}
