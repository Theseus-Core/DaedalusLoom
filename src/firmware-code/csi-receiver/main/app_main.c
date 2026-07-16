/*
 * SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
// build test

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_err.h"
#include "nvs_flash.h"

#include "driver/gpio.h"
#include "driver/uart.h"
#include "soc/gpio_num.h"


#include "esp_csi_gain_ctrl.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "rom/ets_sys.h"

#include "storage.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL 11
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 ||                                     \
    (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_PHYMODE WIFI_PHY_MODE_HT40
#define CONFIG_ESP_NOW_RATE WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN 0

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF 0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 ||         \
    CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL 1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

#define IS_PRINT_CSI_INFO 1

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";
static int g_csi_package_count = 0;

#define EX_UART_NUM UART_NUM_1
#define UART_TXD_PIN 11
#define UART_RXD_PIN 12
#define UART_BAUD_RATE 921600
#define CSI_TO_UART_ENABLED 1

#define SWITCH_GPIO GPIO_NUM_23

typedef enum
{
    TO_CONSOLE = 0,
    TO_UART = 1,
} CSI_DATA_OUTPUT_MODE;

static void uart_init(void)
{
    const uart_config_t uart_config = {
        .baud_rate = UART_BAUD_RATE,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    ESP_ERROR_CHECK(uart_driver_install(EX_UART_NUM, 1024, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(EX_UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(EX_UART_NUM, UART_TXD_PIN, UART_RXD_PIN, UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE));
}

static void uart_send_csi_binary(uint32_t seq, int8_t *buf, uint16_t len, float compensate_gain)
{
    if (len > 1024)
    {
        len = 1024;
    }

    uint8_t header[10];
    header[0] = 'C';
    header[1] = 'S';
    header[2] = 'I';
    header[3] = '1';

    // seq (4字节，小端)
    header[4] = (uint8_t)(seq & 0xFF);
    header[5] = (uint8_t)((seq >> 8) & 0xFF);
    header[6] = (uint8_t)((seq >> 16) & 0xFF);
    header[7] = (uint8_t)((seq >> 24) & 0xFF);

    // len (2字节，小端)
    header[8] = (uint8_t)(len & 0xFF);
    header[9] = (uint8_t)((len >> 8) & 0xFF);

    int8_t payload[1024];
    for (uint16_t i = 0; i < len; ++i)
    {
        int val = (int)(compensate_gain * buf[i]);
        if (val > 127)
            val = 127;
        if (val < -128)
            val = -128;
        payload[i] = (int8_t)val;
    }

    // 校验和 (头10字节之和 + 载荷之和，低16位)
    uint32_t sum = 0;
    for (int i = 0; i < 10; ++i)
    {
        sum += header[i];
    }
    for (uint16_t i = 0; i < len; ++i)
    {
        sum += (uint8_t)payload[i];
    }

    uint8_t checksum_bytes[2];
    checksum_bytes[0] = (uint8_t)(sum & 0xFF);
    checksum_bytes[1] = (uint8_t)((sum >> 8) & 0xFF);

    uart_write_bytes(EX_UART_NUM, (const char *)header, 10);
    uart_write_bytes(EX_UART_NUM, (const char *)payload, len);
    uart_write_bytes(EX_UART_NUM, (const char *)checksum_bytes, 2);
}

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {.ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
                                  .ghz_5g = CONFIG_WIFI_5G_PROTOCOL};
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {.ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
                                   .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS};
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) ||            \
    CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
#if CONFIG_IDF_TARGET_ESP32C5
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY &&
         CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) ||
        (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY &&
         CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20))
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }
    else
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) ||            \
    CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY &&
        CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }
    else
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20)
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    }
    else
    {
        ESP_ERROR_CHECK(
            esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {.phymode = CONFIG_ESP_NOW_PHYMODE,
                                         .rate = CONFIG_ESP_NOW_RATE, //  WIFI_PHY_RATE_MCS0_LGI,
                                         .ersu = false,
                                         .dcm = false};
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

__attribute__((unused)) static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf)
    {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6))
    {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100)
    {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    }
    else if (s_count == 100)
    {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
        ESP_LOGD(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
    // ESP_LOGI(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d", compensate_gain,
    // agc_gain,fft_gain);
#endif

    uint32_t rx_id = *(uint32_t *)(info->payload + 15);
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
    if (!s_count)
    {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,"
                   "sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d", rx_id, MAC2STR(info->mac),
               rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->noise_floor, fft_gain, agc_gain,
               rx_ctrl->channel, rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state);
#else
    if (!s_count)
    {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,"
                   "aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_"
                   "channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel,
               rx_ctrl->secondary_channel, rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len,
               rx_ctrl->rx_state);

#endif

#define SEND_TO_CONSOLE 0
#if SEND_TO_CONSOLE == 1
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
    int16_t csi = ((int16_t)(((((uint16_t)info->buf[1]) << 8) | info->buf[0]) << 4) >> 4);
    ets_printf(",%d,%d,\"[%d", (info->len - 2) / 2, info->first_word_invalid,
               (int16_t)(compensate_gain * csi));
    for (int i = 2; i < (info->len - 2); i += 2)
    {
        csi = ((int16_t)(((((uint16_t)info->buf[i + 1]) << 8) | info->buf[i]) << 4) >> 4);
        ets_printf(",%d", (int16_t)(compensate_gain * csi));
    }
#else
    printf(",%d,%d,\"[%d", info->len, info->first_word_invalid,
           (int16_t)(compensate_gain * info->buf[0]));

    for (int i = 1; i < info->len; i++)
    {

        printf(",%d", (int16_t)(compensate_gain * info->buf[i]));
    }
#endif
    printf("]\"\n");
#endif

#if CSI_TO_UART_ENABLED
    uart_send_csi_binary(rx_id, info->buf, info->len, compensate_gain);
#endif
    s_count++;
}

void callback2console(void *ctx, wifi_csi_info_t *info)
{

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6))
    {
        return;
    }
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;

    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (g_csi_package_count < 100)
    {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    }
    else if (g_csi_package_count == 100)
    {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain,
                                            fft_gain); // 各种增益补偿计算
    // ESP_LOGI(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d", compensate_gain,agc_gain,
    // fft_gain);

    // ESP_LOGI(TAG, "info.len: %d", info->len);

    g_csi_package_count++;
    printf("index:%d len:%d compensate_gain:%f data:[", g_csi_package_count, info->len,
           compensate_gain);

    if (info->len > 0)
    {

        printf("%d", (int16_t)(compensate_gain * info->buf[0]));
        // 剩下的元素前面加逗号
        for (int i = 1; i < info->len; i++)
        {

            printf(",%d", (int16_t)(compensate_gain * info->buf[i]));
        }
    }

    printf("]\n");


}


void callback2uart(void *ctx, wifi_csi_info_t *info)
{

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6))
    {
        return;
    }
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;

    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (g_csi_package_count < 100)
    {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    }
    else if (g_csi_package_count == 100)
    {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain,
                                            fft_gain); // 各种增益补偿计算

    g_csi_package_count++;
    uart_send_csi_binary(g_csi_package_count, info->buf, info->len, compensate_gain);

}




static void wifi_csi_init(wifi_csi_cb_t recv_data_cb)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    /**< default config */
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {.enable = true,
                                    .acquire_csi_legacy = false,
                                    .acquire_csi_force_lltf = CSI_FORCE_LLTF,
                                    .acquire_csi_ht20 = true,
                                    .acquire_csi_ht40 = true,
                                    .acquire_csi_vht = false,
                                    .acquire_csi_su = false,
                                    .acquire_csi_mu = false,
                                    .acquire_csi_dcm = false,
                                    .acquire_csi_beamformed = false,
                                    .acquire_csi_he_stbc_mode = 2,
                                    .val_scale_cfg = 0,
                                    .dump_ack_en = false,
                                    .reserved = false};
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {.enable = true,
                                    .acquire_csi_legacy = false,
                                    .acquire_csi_ht20 = true,
                                    .acquire_csi_ht40 = true,
                                    .acquire_csi_su = true,
                                    .acquire_csi_mu = true,
                                    .acquire_csi_dcm = true,
                                    .acquire_csi_beamformed = true,
                                    .acquire_csi_he_stbc = 2,
                                    .val_scale_cfg = false,
                                    .dump_ack_en = false,
                                    .reserved = false};
#else
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = false,
        .shift = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(recv_data_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void gpio_init(void)
{
    // 1. 设置引脚为输入模式
    gpio_set_direction(SWITCH_GPIO, GPIO_MODE_INPUT);
    // 2. 启用内部上拉电阻（以按键接GND为例）
    gpio_set_pull_mode(SWITCH_GPIO, GPIO_PULLUP_ONLY);
}

void app_main()
{
    /**
     * @brief Initialize NVS
     */
    storage_init();
    gpio_init();

    int enable_switch = gpio_get_level(SWITCH_GPIO);
    ESP_LOGI(TAG, "Switch GPIO level: %d", enable_switch);
    if (enable_switch == 0)
    {
        int8_t output_mode;
        esp_err_t ret = easy_read_i8("system_config", "output_mode", &output_mode);
        if (ret == ESP_OK)
        {
            if (output_mode == (int8_t)TO_CONSOLE)
            {
                ESP_LOGI(TAG, "Output mode: TO_CONSOLE");
                output_mode = TO_UART;
                easy_save_i8("system_config", "output_mode", &output_mode);
            }
            else if (output_mode == (int8_t)TO_UART)
            {
                ESP_LOGI(TAG, "Output mode: TO_UART");
                output_mode = TO_CONSOLE;
                easy_save_i8("system_config", "output_mode", &output_mode);
            }
            else
            {
                ESP_LOGW(TAG, "Unknown output mode: %d", output_mode);
            }
        }
        else
        {
            int8_t default_mode = TO_CONSOLE;
            ret = easy_save_i8("system_config", "output_mode", &default_mode);
        }
    }
    int8_t output_mode;
    easy_read_i8("system_config", "output_mode", &output_mode);
    ESP_LOGD(TAG, "Output mode: %d", output_mode);
    if (output_mode == (int8_t)TO_UART)
    {
        ESP_LOGI(TAG, "Setting callback to UART");
        uart_init();
       
    }
    wifi_init();

    /**
     * @brief Initialize ESP-NOW
     *        ESP-NOW protocol see:
     * https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
     */

    esp_now_peer_info_t peer = {
        .channel = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);
    if (output_mode == (int8_t)TO_UART)
    {
        ESP_LOGI(TAG, "Setting callback to UART");
        wifi_csi_init(callback2uart);
    }
    else
    {
        ESP_LOGI(TAG, "Setting callback to Console");
        wifi_csi_init(callback2console);
    }
    
}
