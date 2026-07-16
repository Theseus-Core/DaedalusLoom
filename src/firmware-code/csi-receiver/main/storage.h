#include "nvs_flash.h"
#include "esp_log.h"


esp_err_t storage_init(void);

esp_err_t storage_erase(void);

esp_err_t easy_save_i8(char *namespace, const char *key, int8_t *value);

esp_err_t easy_read_i8(char *namespace, const char *key, int8_t *value);