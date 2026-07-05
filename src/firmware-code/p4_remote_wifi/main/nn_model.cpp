#include "nn_model.h"
#include "sys_config.h"
#include "csi_dsp.h"
#include "dl_model_base.hpp"
#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"
#include "cJSON.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <cstdlib>
#include <map>
#include <strings.h>
#include <sys/stat.h>

#define TAG "nn_model"

#define SD_MOUNT_POINT "/sdcard"
#define SD_MODEL_DIR SD_MOUNT_POINT "/models"
#define SD_LEGACY_MODEL_DIR SD_MOUNT_POINT "/sdcard_models"
#define MODEL_NAME_MAX_LEN 24
#define MODEL_PATH_MAX_LEN 96
#define MODEL_META_MAX_BYTES (8 * 1024)

static constexpr gpio_num_t SD_PWR_EN_GPIO = GPIO_NUM_45;
static constexpr int SDMMC_IO_LDO_CHAN_ID = 4;
static constexpr int SD_CARD_SDMMC_SLOT = SDMMC_HOST_SLOT_0;

typedef struct {
    char name[MODEL_NAME_MAX_LEN];
    char path[MODEL_PATH_MAX_LEN];
    char meta_path[MODEL_PATH_MAX_LEN];
    long size;
    int class_count;
    bool metadata_loaded;
    char class_names[NN_MODEL_MAX_CLASS_COUNT][NN_MODEL_CLASS_NAME_MAX_LEN];
    dl::Model *model;
    dl::TensorBase *input;
    dl::TensorBase *output;
} model_slot_t;

static model_slot_t g_model_slots[NN_MODEL_MAX_SLOT_COUNT];
static int g_model_slot_count = 0;
static bool g_sdcard_mounted = false;
static sdmmc_card_t *g_sdcard = nullptr;
static sd_pwr_ctrl_handle_t g_sd_pwr_ctrl = nullptr;

// Active model pointers kept hot for real-time inference.
static int g_active_model_id = 0;
static dl::Model *g_model = nullptr;
static dl::TensorBase *g_model_input = nullptr;
static dl::TensorBase *g_model_output = nullptr;
static int g_active_class_count = 4;

static const char *const DEFAULT_CLASS_NAMES[] = {
    "高抬腿", "展臂", "深蹲", "Unknown"
};
static constexpr int DEFAULT_CLASS_COUNT = sizeof(DEFAULT_CLASS_NAMES) / sizeof(DEFAULT_CLASS_NAMES[0]);

static bool is_valid_model_id(int model_id)
{
    return model_id >= 0 && model_id < g_model_slot_count;
}

static bool has_espdl_extension(const char *name)
{
    if (!name) {
        return false;
    }

    const char *dot = strrchr(name, '.');
    return dot && strcasecmp(dot, ".espdl") == 0;
}

static const char *basename_from_path(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static void copy_truncated(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0) {
        return;
    }

    const char *safe_src = src ? src : "";
    size_t len = strlen(safe_src);
    if (len >= dst_size) {
        len = dst_size - 1;
    }
    memcpy(dst, safe_src, len);
    dst[len] = '\0';
}

static void copy_model_display_name(char *dst, size_t dst_size, const char *filename)
{
    if (!dst || dst_size == 0) {
        return;
    }

    const char *base = basename_from_path(filename);
    copy_truncated(dst, dst_size, base ? base : "MODEL");

    char *dot = strrchr(dst, '.');
    if (dot && strcasecmp(dot, ".espdl") == 0) {
        *dot = '\0';
    }

    if (dst[0] == '\0') {
        copy_truncated(dst, dst_size, "MODEL");
    }
}

static void set_default_class_names(model_slot_t *slot)
{
    if (!slot) {
        return;
    }

    slot->class_count = DEFAULT_CLASS_COUNT;
    slot->metadata_loaded = false;
    for (int i = 0; i < NN_MODEL_MAX_CLASS_COUNT; i++) {
        if (i < DEFAULT_CLASS_COUNT) {
            copy_truncated(slot->class_names[i], sizeof(slot->class_names[i]), DEFAULT_CLASS_NAMES[i]);
        } else {
            snprintf(slot->class_names[i], sizeof(slot->class_names[i]), "Class %d", i);
        }
    }
}

static bool replace_extension(char *dst, size_t dst_size, const char *path, const char *extension)
{
    if (!dst || dst_size == 0 || !path || !extension) {
        return false;
    }

    copy_truncated(dst, dst_size, path);
    char *dot = strrchr(dst, '.');
    if (!dot) {
        return false;
    }

    size_t prefix_len = (size_t)(dot - dst);
    size_t ext_len = strlen(extension);
    if (prefix_len + ext_len >= dst_size) {
        dst[0] = '\0';
        return false;
    }

    memcpy(dst + prefix_len, extension, ext_len + 1);
    return true;
}

static bool file_exists(const char *path)
{
    struct stat st;
    return path && stat(path, &st) == 0 && !S_ISDIR(st.st_mode);
}

static cJSON *read_json_file(const char *path)
{
    if (!path || !file_exists(path)) {
        return nullptr;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        ESP_LOGW(TAG, "metadata open failed: %s", path);
        return nullptr;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return nullptr;
    }
    long size = ftell(fp);
    if (size <= 0 || size > MODEL_META_MAX_BYTES) {
        fclose(fp);
        ESP_LOGD(TAG, "skip non-metadata-sized json: %s (%ld bytes)", path, size);
        return nullptr;
    }
    rewind(fp);

    char *buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(fp);
        ESP_LOGW(TAG, "metadata allocation failed: %s", path);
        return nullptr;
    }

    size_t read_len = fread(buf, 1, (size_t)size, fp);
    fclose(fp);
    buf[read_len] = '\0';

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) {
        ESP_LOGW(TAG, "metadata parse failed: %s", path);
    }
    return root;
}

static cJSON *get_json_item_any(cJSON *root, const char *name_a, const char *name_b)
{
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, name_a);
    if (!item && name_b) {
        item = cJSON_GetObjectItemCaseSensitive(root, name_b);
    }
    return item;
}

static bool load_model_metadata_from_json(model_slot_t *slot, const char *path)
{
    cJSON *root = read_json_file(path);
    if (!root) {
        return false;
    }

    cJSON *labels = get_json_item_any(root, "labels", "classes");
    if (!cJSON_IsArray(labels)) {
        cJSON_Delete(root);
        return false;
    }

    int label_count = cJSON_GetArraySize(labels);
    if (label_count <= 0 || label_count > NN_MODEL_MAX_CLASS_COUNT) {
        ESP_LOGW(TAG, "metadata label count out of range: %s (%d)", path, label_count);
        cJSON_Delete(root);
        return false;
    }

    cJSON *class_count_item = get_json_item_any(root, "num_classes", "class_count");
    int class_count = cJSON_IsNumber(class_count_item) ? class_count_item->valueint : label_count;
    if (class_count != label_count) {
        ESP_LOGW(TAG, "metadata class count mismatch: %s (%d labels, count=%d)",
                 path,
                 label_count,
                 class_count);
        cJSON_Delete(root);
        return false;
    }

    cJSON *name = get_json_item_any(root, "name", "display_name");
    if (cJSON_IsString(name) && name->valuestring && name->valuestring[0] != '\0') {
        copy_truncated(slot->name, sizeof(slot->name), name->valuestring);
    }

    slot->class_count = class_count;
    for (int i = 0; i < class_count; i++) {
        cJSON *label = cJSON_GetArrayItem(labels, i);
        if (!cJSON_IsString(label) || !label->valuestring || label->valuestring[0] == '\0') {
            ESP_LOGW(TAG, "metadata label %d invalid: %s", i, path);
            cJSON_Delete(root);
            set_default_class_names(slot);
            return false;
        }
        copy_truncated(slot->class_names[i], sizeof(slot->class_names[i]), label->valuestring);
    }
    for (int i = class_count; i < NN_MODEL_MAX_CLASS_COUNT; i++) {
        snprintf(slot->class_names[i], sizeof(slot->class_names[i]), "Class %d", i);
    }

    copy_truncated(slot->meta_path, sizeof(slot->meta_path), path);
    slot->metadata_loaded = true;
    ESP_LOGI(TAG, "loaded metadata for %s: %d classes from %s", slot->name, slot->class_count, path);
    cJSON_Delete(root);
    return true;
}

static void load_model_metadata(model_slot_t *slot)
{
    char meta_path[MODEL_PATH_MAX_LEN];

    if (replace_extension(meta_path, sizeof(meta_path), slot->path, ".meta.json") &&
        load_model_metadata_from_json(slot, meta_path)) {
        return;
    }

    if (replace_extension(meta_path, sizeof(meta_path), slot->path, ".labels.json") &&
        load_model_metadata_from_json(slot, meta_path)) {
        return;
    }

    if (replace_extension(meta_path, sizeof(meta_path), slot->path, ".json") &&
        load_model_metadata_from_json(slot, meta_path)) {
        return;
    }

    ESP_LOGW(TAG,
             "no usable metadata found for %s; using default %d-class labels",
             slot->path,
             DEFAULT_CLASS_COUNT);
}

static int infer_output_class_count(dl::TensorBase *output)
{
    if (!output) {
        return 0;
    }

    int size = output->get_size();
    if (size <= 0 || size > NN_MODEL_MAX_CLASS_COUNT) {
        ESP_LOGW(TAG, "model output size out of supported range: %d (max=%d)",
                 size,
                 NN_MODEL_MAX_CLASS_COUNT);
        return 0;
    }
    return size;
}

static void free_model_slots(void)
{
    for (int i = 0; i < g_model_slot_count; i++) {
        if (g_model_slots[i].model != nullptr) {
            delete g_model_slots[i].model;
            g_model_slots[i].model = nullptr;
        }
        g_model_slots[i].input = nullptr;
        g_model_slots[i].output = nullptr;
    }
    g_model = nullptr;
    g_model_input = nullptr;
    g_model_output = nullptr;
    g_active_model_id = 0;
    g_active_class_count = DEFAULT_CLASS_COUNT;
}

static float tensor_value_as_float(dl::TensorBase *tensor, int index)
{
    if (!tensor) {
        return 0.0f;
    }

    switch (tensor->get_dtype()) {
    case dl::DATA_TYPE_INT8:
        return (float)tensor->get_element_ptr<int8_t>()[index] * std::pow(2.0f, tensor->get_exponent());
    case dl::DATA_TYPE_INT16:
        return (float)tensor->get_element_ptr<int16_t>()[index] * std::pow(2.0f, tensor->get_exponent());
    case dl::DATA_TYPE_INT32:
        return (float)tensor->get_element_ptr<int32_t>()[index] * std::pow(2.0f, tensor->get_exponent());
    case dl::DATA_TYPE_FLOAT:
        return tensor->get_element_ptr<float>()[index];
    default:
        ESP_LOGW(TAG, "unsupported output dtype: %s", tensor->get_dtype_string());
        return 0.0f;
    }
}

static esp_err_t enable_sdcard_power(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << SD_PWR_EN_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
#if SOC_GPIO_SUPPORT_PIN_HYS_FILTER
        .hys_ctrl_mode = GPIO_HYS_SOFT_DISABLE,
#endif
    };
    esp_err_t ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to configure SD power gpio: %s", esp_err_to_name(ret));
        return ret;
    }

    // ESP32-P4-Function-EV-Board uses an active-low SD_PWRn enable on GPIO45.
    ret = gpio_set_level(SD_PWR_EN_GPIO, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to enable SD power: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t mount_sdcard_once(void)
{
    if (g_sdcard_mounted) {
        return ESP_OK;
    }

    esp_err_t ret = enable_sdcard_power();
    if (ret != ESP_OK) {
        return ret;
    }

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 6,
        .allocation_unit_size = 16 * 1024,
        .disk_status_check_enable = false,
        .use_one_fat = false,
    };

    sd_pwr_ctrl_ldo_config_t ldo_config = {
        .ldo_chan_id = SDMMC_IO_LDO_CHAN_ID,
    };
    ret = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &g_sd_pwr_ctrl);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to create SDMMC IO LDO power control: %s", esp_err_to_name(ret));
        g_sd_pwr_ctrl = nullptr;
        return ret;
    }

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    // ESP32-P4 board SD-card socket is on slot 0; ESP-Hosted Wi-Fi uses slot 1.
    host.slot = SD_CARD_SDMMC_SLOT;
    host.pwr_ctrl_handle = g_sd_pwr_ctrl;
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    ESP_LOGI(TAG, "mounting SD card at %s via SDMMC slot %d", SD_MOUNT_POINT, host.slot);
    ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &g_sdcard);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "failed to mount FAT filesystem on SD card");
        } else {
            ESP_LOGE(TAG, "failed to initialize SD card: %s", esp_err_to_name(ret));
        }
        sd_pwr_ctrl_del_on_chip_ldo(g_sd_pwr_ctrl);
        g_sd_pwr_ctrl = nullptr;
        g_sdcard = nullptr;
        return ret;
    }

    sdmmc_card_print_info(stdout, g_sdcard);
    g_sdcard_mounted = true;
    return ESP_OK;
}

static bool add_model_file(const char *path)
{
    if (!path || g_model_slot_count >= NN_MODEL_MAX_SLOT_COUNT) {
        return false;
    }

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        ESP_LOGW(TAG, "skip unreadable model file: %s", path);
        return false;
    }
    if (S_ISDIR(st.st_mode)) {
        return false;
    }

    model_slot_t *slot = &g_model_slots[g_model_slot_count];
    memset(slot, 0, sizeof(*slot));
    copy_model_display_name(slot->name, sizeof(slot->name), path);
    copy_truncated(slot->path, sizeof(slot->path), path);
    slot->size = (long)st.st_size;
    set_default_class_names(slot);
    load_model_metadata(slot);

    ESP_LOGI(TAG, "found SD model %d: %s (%ld bytes, %d classes)",
             g_model_slot_count,
             slot->path,
             slot->size,
             slot->class_count);
    g_model_slot_count++;
    return true;
}

static void scan_model_dir(const char *dir_path)
{
    DIR *dir = opendir(dir_path);
    if (!dir) {
        ESP_LOGW(TAG, "model directory not found or unreadable: %s", dir_path);
        return;
    }

    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (g_model_slot_count >= NN_MODEL_MAX_SLOT_COUNT) {
            ESP_LOGW(TAG, "model slot limit reached (%d)", NN_MODEL_MAX_SLOT_COUNT);
            break;
        }
        if (entry->d_name[0] == '.' || !has_espdl_extension(entry->d_name)) {
            continue;
        }

        char path[MODEL_PATH_MAX_LEN];
        int written = snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
        if (written < 0 || written >= (int)sizeof(path)) {
            ESP_LOGW(TAG, "skip model with too long path: %s/%s", dir_path, entry->d_name);
            continue;
        }
        add_model_file(path);
    }
    closedir(dir);
}

static void scan_sd_models(void)
{
    free_model_slots();
    g_model_slot_count = 0;
    memset(g_model_slots, 0, sizeof(g_model_slots));

    if (mount_sdcard_once() != ESP_OK) {
        return;
    }

    scan_model_dir(SD_MOUNT_POINT);
    scan_model_dir(SD_MODEL_DIR);
    scan_model_dir(SD_LEGACY_MODEL_DIR);

    if (g_model_slot_count == 0) {
        ESP_LOGW(TAG,
                 "no .espdl models found in %s, %s or %s",
                 SD_MOUNT_POINT,
                 SD_MODEL_DIR,
                 SD_LEGACY_MODEL_DIR);
    }
}

static esp_err_t ensure_model_loaded(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return ESP_ERR_INVALID_ARG;
    }

    model_slot_t *slot = &g_model_slots[model_id];
    if (slot->path[0] == '\0' || slot->size <= 0) {
        ESP_LOGW(TAG, "%s is not installed", slot->name);
        return ESP_ERR_NOT_FOUND;
    }

    if (slot->model != nullptr) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Loading %s from SD card: %s", slot->name, slot->path);
    slot->model = new dl::Model(slot->path,
                                fbs::MODEL_LOCATION_IN_SDCARD,
                                16 * 1024,
                                dl::MEMORY_MANAGER_GREEDY,
                                nullptr,
                                false);
    if (!slot->model) {
        ESP_LOGE(TAG, "Failed to load %s", slot->name);
        return ESP_FAIL;
    }

    std::map<std::string, dl::TensorBase *> model_inputs = slot->model->get_inputs();
    std::map<std::string, dl::TensorBase *> model_outputs = slot->model->get_outputs();
    if (model_inputs.empty() || model_outputs.empty()) {
        ESP_LOGE(TAG, "%s has invalid tensor metadata", slot->name);
        delete slot->model;
        slot->model = nullptr;
        return ESP_FAIL;
    }

    slot->input = model_inputs.begin()->second;
    slot->output = model_outputs.begin()->second;
    int output_class_count = infer_output_class_count(slot->output);
    if (output_class_count <= 0) {
        delete slot->model;
        slot->model = nullptr;
        slot->input = nullptr;
        slot->output = nullptr;
        return ESP_FAIL;
    }

    if (slot->metadata_loaded) {
        if (slot->class_count != output_class_count) {
            ESP_LOGE(TAG,
                     "%s metadata class count (%d) does not match model output (%d)",
                     slot->name,
                     slot->class_count,
                     output_class_count);
            delete slot->model;
            slot->model = nullptr;
            slot->input = nullptr;
            slot->output = nullptr;
            return ESP_ERR_INVALID_SIZE;
        }
    } else {
        slot->class_count = output_class_count;
    }

    ESP_LOGI(TAG, "%s loaded. Size: %ld bytes, Classes: %d, Input Exp: %d, Output Exp: %d",
             slot->name,
             slot->size,
             slot->class_count,
             slot->input->get_exponent(),
             slot->output->get_exponent());
    return ESP_OK;
}

void nn_model_init(void)
{
    scan_sd_models();

    if (g_model_slot_count <= 0) {
        ESP_LOGE(TAG, "No SD-card models available; inference will stay disabled.");
        return;
    }

    int requested_id = sys_config_get_active_model_id();
    if (!is_valid_model_id(requested_id)) {
        requested_id = 0;
    }

    if (nn_model_select(requested_id) != ESP_OK) {
        ESP_LOGW(TAG, "Falling back to first SD-card model");
        (void)nn_model_select(0);
    }
}

void nn_model_rescan(void)
{
    scan_sd_models();
}

int nn_model_get_slot_count(void)
{
    return g_model_slot_count;
}

bool nn_model_is_installed(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return false;
    }
    return g_model_slots[model_id].path[0] != '\0' && g_model_slots[model_id].size > 0;
}

esp_err_t nn_model_select(int model_id)
{
    esp_err_t ret = ensure_model_loaded(model_id);
    if (ret != ESP_OK) {
        return ret;
    }

    model_slot_t *slot = &g_model_slots[model_id];
    g_active_model_id = model_id;
    g_model = slot->model;
    g_model_input = slot->input;
    g_model_output = slot->output;
    g_active_class_count = slot->class_count;
    sys_config_set_active_model_id(model_id);
    ESP_LOGI(TAG, "Active model switched to %s (%d classes)", slot->name, g_active_class_count);
    return ESP_OK;
}

int nn_model_get_active_id(void)
{
    return g_active_model_id;
}

const char *nn_model_get_active_name(void)
{
    return nn_model_get_slot_name(g_active_model_id);
}

int nn_model_get_class_count(void)
{
    return g_active_class_count;
}

int nn_model_get_slot_class_count(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return 0;
    }
    return g_model_slots[model_id].class_count;
}

const char *nn_model_get_slot_name(int model_id)
{
    if (!is_valid_model_id(model_id)) {
        return "NO MODEL";
    }
    return g_model_slots[model_id].name;
}

const char *nn_model_get_class_name(int class_id)
{
    if (class_id < 0 || class_id >= g_active_class_count || !is_valid_model_id(g_active_model_id)) {
        return "Unknown";
    }
    const char *name = g_model_slots[g_active_model_id].class_names[class_id];
    return name[0] != '\0' ? name : "Unknown";
}

int nn_model_predict(float x, float y)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }

    int input_exponent = g_model_input->get_exponent();

    // Quantize inputs: float_val / (2^input_exponent)
    int8_t q_x = std::round(x / std::pow(2.0f, input_exponent));
    int8_t q_y = std::round(y / std::pow(2.0f, input_exponent));

    // Fill input tensor data
    int8_t *input_data = (int8_t *)g_model_input->data;
    input_data[0] = q_x;
    input_data[1] = q_y;

    // Run forward pass
    g_model->run();

    // Read quantized outputs
    int class_count = nn_model_get_class_count();
    if (class_count <= 0) {
        return -1;
    }

    // Find class with the maximum score (argmax)
    int max_class = 0;
    float max_val = tensor_value_as_float(g_model_output, 0);
    for (int c = 1; c < class_count; c++) {
        float val = tensor_value_as_float(g_model_output, c);
        if (val > max_val) {
            max_val = val;
            max_class = c;
        }
    }

    return max_class;
}

int nn_model_predict_cnn(float *raw_csi)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }

    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
        if (preprocessed_features == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate preprocessed feature buffer in PSRAM");
            return -1;
        }
    }

    // Perform on-board preprocessing
    csi_dsp_preprocess_fusion(raw_csi, preprocessed_features);

    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent);

    int8_t *input_data = (int8_t *)g_model_input->data;

    // Quantize 11400 features into the input tensor using fast multiplication
    for (int i = 0; i < CSI_DSP_TOTAL_FEATURES; i++) {
        float q_val = std::round(preprocessed_features[i] * inv_scale);
        int32_t qi = (int32_t)q_val;
        if (qi > 127) qi = 127;
        else if (qi < -128) qi = -128;
        input_data[i] = (int8_t)qi;
    }

    // Run forward pass
    g_model->run();

    // Read output tensor data
    int class_count = nn_model_get_class_count();
    if (class_count <= 0) {
        return -1;
    }

    // Find class with the maximum score (argmax among classes)
    int max_class = 0;
    float max_val = tensor_value_as_float(g_model_output, 0);
    for (int c = 1; c < class_count; c++) {
        float val = tensor_value_as_float(g_model_output, c);
        if (val > max_val) {
            max_val = val;
            max_class = c;
        }
    }

    return max_class;
}

int nn_model_predict_cnn_with_probs(float *raw_csi, float *out_probs)
{
    if (g_model == nullptr || g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "Model not initialized!");
        return -1;
    }
    if (raw_csi == nullptr || out_probs == nullptr) {
        return -1;
    }

    static float *preprocessed_features = nullptr;
    if (preprocessed_features == nullptr) {
        preprocessed_features = (float *)heap_caps_malloc(CSI_DSP_TOTAL_FEATURES * sizeof(float), MALLOC_CAP_SPIRAM);
        if (preprocessed_features == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate preprocessed feature buffer in PSRAM");
            return -1;
        }
    }

    // Perform on-board preprocessing
    csi_dsp_preprocess_fusion(raw_csi, preprocessed_features);

    int input_exponent = g_model_input->get_exponent();
    float inv_scale = 1.0f / std::pow(2.0f, input_exponent);

    int8_t *input_data = (int8_t *)g_model_input->data;

    // Quantize features using fast multiplication
    for (int i = 0; i < CSI_DSP_TOTAL_FEATURES; i++) {
        float q_val = std::round(preprocessed_features[i] * inv_scale);
        int32_t qi = (int32_t)q_val;
        if (qi > 127) qi = 127;
        else if (qi < -128) qi = -128;
        input_data[i] = (int8_t)qi;
    }

    // Run forward pass
    g_model->run();

    int class_count = nn_model_get_class_count();
    if (class_count <= 0) {
        return -1;
    }

    // Calculate Softmax probabilities
    float sum_exp = 0.0f;
    float float_logits[NN_MODEL_MAX_CLASS_COUNT];
    float max_logit = tensor_value_as_float(g_model_output, 0);
    for (int c = 1; c < class_count; c++) {
        float val = tensor_value_as_float(g_model_output, c);
        if (val > max_logit) {
            max_logit = val;
        }
    }
    for (int c = 0; c < class_count; c++) {
        float_logits[c] = tensor_value_as_float(g_model_output, c);
        sum_exp += expf(float_logits[c] - max_logit);
    }
    for (int c = 0; c < class_count; c++) {
        out_probs[c] = expf(float_logits[c] - max_logit) / sum_exp;
    }
    for (int c = class_count; c < NN_MODEL_MAX_CLASS_COUNT; c++) {
        out_probs[c] = 0.0f;
    }

    // Find class with the maximum score
    int max_class = 0;
    float max_prob = out_probs[0];
    for (int c = 1; c < class_count; c++) {
        if (out_probs[c] > max_prob) {
            max_prob = out_probs[c];
            max_class = c;
        }
    }

    return max_class;
}
