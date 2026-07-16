#include "esp_http_client.h"
#include "esp_log.h"
#include "lwip/sockets.h"
#include "sdkconfig.h"
#include <stdio.h>
#include <string.h>

static const char *TAG = "UPLOAD API";

#define BASE_URL CONFIG_ddloom_base_url

void upload_info(const char *url, const char *data)
{   
    
    esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_POST,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, data, strlen(data));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK)
    {
        ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
                 esp_http_client_get_status_code(client),
                 esp_http_client_get_content_length(client));
    }
    else
    {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
}

void test_upload()
{

    char endpoint[] = "/api/sensor/signal";
    char url[128];
    snprintf(url, sizeof(url), "%s%s", BASE_URL, endpoint);
    const char *jsonStr = "{\
        \"patientId\": 3,\
        \"behaviorType\": \"起身\",\
        \"description\": \"患者苏醒\",\
        \"isAbnormal\": true,\
        \"severity\": 1\
    }";
    ESP_LOGI(TAG, "Uploading to URL: %s", url);
    upload_info(url, jsonStr);
    
}