#include "nvs_flash.h"
#include "esp_log.h"


esp_err_t storage_init(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    return ret;
}

esp_err_t storage_erase(void)
{
    return nvs_flash_erase();
}

esp_err_t easy_save_i8(char *namespace, const char *key, int8_t *value)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(namespace, NVS_READWRITE, &handle);
    if (err != ESP_OK)
    {
        ESP_LOGE(namespace, "Error opening NVS");
        return err;
    }

    err = nvs_set_i8(handle, key, *value);
    if (err != ESP_OK)
    {
        ESP_LOGE(namespace, "Error setting NVS ");
        nvs_close( handle);
        return err;
    }

    err = nvs_commit(handle);
    if (err != ESP_OK)
    {
        ESP_LOGE(namespace, "Error committing NVS");
        nvs_close(handle);
        return err;
    }

    nvs_close(handle);
    return ESP_OK;
}

esp_err_t easy_read_i8(char *namespace, const char *key, int8_t *value)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open(namespace, NVS_READONLY, &handle);
    if (err != ESP_OK)
    {
        ESP_LOGE(namespace, "Error opening NVS");
        return err;
    }

    err = nvs_get_i8(handle, key, value);
    if (err != ESP_OK)
    {
        ESP_LOGE(namespace, "Error getting NVS i8");
        nvs_close(handle);
        return err;
    }

    nvs_close(handle);
    return ESP_OK;
}