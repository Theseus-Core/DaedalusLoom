#include "wifi_manager.h"
#include <stdio.h>
#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_log_level.h"
#include "esp_netif.h"
#include "esp_netif_types.h"
#include "esp_wifi_default.h"
#include "esp_wifi_types_generic.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"

#include "sdkconfig.h" // 必须引入此头文件
#include "upload_api.h"

#define WIFI_SSID CONFIG_ddloom_WIFI_SSID 
#define WIFI_PASSWORD CONFIG_ddloom_WIFI_PASSWORD
#define TAG "wifi_manager"

static EventGroupHandle_t wifi_event_group;
static uint8_t WIFI_AUTO_RECONNECT_ENABLED = 0;

static void wifi_event_handler_STA(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                ESP_LOGI(TAG, "WiFi驱动就绪");
                break;
            case WIFI_EVENT_STA_CONNECTED:
                ESP_LOGI(TAG, "已连接到热点");
                break;
            case WIFI_EVENT_STA_DISCONNECTED:
                ESP_LOGW(TAG, "热点断开, 尝试重连...");
                WIFI_AUTO_RECONNECT_ENABLED = 1;
                // Try to reconnect
                esp_wifi_connect();
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "成功获取IP:" IPSTR, IP2STR(&event->ip_info.ip));
        
    }
}

void wifi_manager_init(void) {
    // 1. Initialize NVS (essential for Wi-Fi storage)


    ESP_LOGI(TAG, "Initializing Wi-Fi STA Mode...");

    wifi_event_group = xEventGroupCreate();
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_LOGI(TAG, "wifi init successful");

    esp_event_handler_instance_t wifi_event_handle, ip_event_handle;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler_STA, NULL, &wifi_event_handle));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler_STA, NULL, &ip_event_handle));
    
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    
    if (esp_wifi_start() == ESP_OK) {
        ESP_LOGI(TAG, "wifi start successful");

    }
    
    esp_wifi_connect();

}


bool is_wifi_connected(void)
{
    // 1. 检查WiFi状态
    wifi_ap_record_t ap_info;
    if (esp_wifi_sta_get_ap_info(&ap_info) != ESP_OK) {
        ESP_LOGW("WIFI", "Not connected to any AP");
        return false;
    }
    
    // 2. 检查是否获得了IP地址
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (!netif) {
        ESP_LOGW("WIFI", "Network interface not available");
        return false;
    }
    
    esp_netif_ip_info_t ip_info;
    if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
        ESP_LOGW("WIFI", "Failed to get IP info");
        return false;
    }
    
    // 3. 检查IP地址是否有效（不是0.0.0.0）
    if (ip_info.ip.addr == 0) {
        ESP_LOGW("WIFI", "IP address is 0.0.0.0 (not assigned yet)");
        return false;
    }
    
    ESP_LOGI("WIFI", "Connected to SSID: %s, IP: " IPSTR, 
             ap_info.ssid, IP2STR(&ip_info.ip));
    return true;
}