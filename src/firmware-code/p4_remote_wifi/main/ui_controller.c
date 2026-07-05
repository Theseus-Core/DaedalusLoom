#include "ui_controller.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "nn_model.h"
#include "sys_config.h"

#define TAG "ui_controller"

#define ENCODER_A_GPIO GPIO_NUM_6
#define ENCODER_B_GPIO GPIO_NUM_5
#define ENCODER_SW_GPIO GPIO_NUM_4

#define OLED_SCK_GPIO GPIO_NUM_24
#define OLED_MOSI_GPIO GPIO_NUM_25
#define OLED_RST_GPIO GPIO_NUM_32
#define OLED_DC_GPIO GPIO_NUM_33
#define OLED_CS_GPIO GPIO_NUM_36
#define OLED_SPI_HOST SPI2_HOST
#define OLED_SPI_CLOCK_HZ (10 * 1000 * 1000)

#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_PAGES (OLED_HEIGHT / 8)
#define OLED_FB_SIZE (OLED_WIDTH * OLED_PAGES)

#ifndef BUTTON_DEBOUNCE_MS
#define BUTTON_DEBOUNCE_MS 25
#endif

#ifndef BUTTON_LONG_PRESS_MS
#define BUTTON_LONG_PRESS_MS 800
#endif

#define UI_EVENT_QUEUE_LEN 32
#define UI_TASK_PRIORITY 6

typedef enum {
    UI_GPIO_EVENT_ENCODER = 0,
    UI_GPIO_EVENT_BUTTON,
} ui_gpio_event_type_t;

typedef struct {
    ui_gpio_event_type_t type;
    TickType_t tick;
    uint8_t encoder_state;
    bool button_pressed;
} ui_gpio_event_t;

typedef enum {
    BUTTON_EVENT_NONE = 0,
    BUTTON_EVENT_CLICK,
    BUTTON_EVENT_LONG_PRESS,
} button_event_t;

typedef enum {
    UI_SCREEN_PARAMS = 0,
    UI_SCREEN_MODEL_SELECT,
    UI_SCREEN_NOTICE,
} ui_screen_t;

static uint8_t g_oled_fb[OLED_FB_SIZE];
static spi_device_handle_t g_oled_spi = NULL;
static bool g_oled_ready = false;

static ui_screen_t g_screen = UI_SCREEN_PARAMS;
static sys_config_param_id_t g_selected_param = SYS_CONFIG_PARAM_MOTION_THRESHOLD;
static int g_model_cursor = 0;
static char g_notice[24] = {0};
static TickType_t g_notice_until = 0;
static QueueHandle_t g_ui_event_queue = NULL;

static bool g_button_raw_pressed_last = false;
static bool g_button_stable_pressed = false;
static bool g_button_long_press_reported = false;
static TickType_t g_button_last_raw_change_tick = 0;
static TickType_t g_button_press_start_tick = 0;

static bool g_encoder_initialized = false;
static uint8_t g_encoder_last_state = 0;
static int g_encoder_accumulator = 0;

static const int8_t g_encoder_table[16] = {
    0, -1, 1, 0,
    1, 0, 0, -1,
    -1, 0, 0, 1,
    0, 1, -1, 0,
};

static const uint8_t FONT_SPACE[5] = {0x00, 0x00, 0x00, 0x00, 0x00};
static const uint8_t FONT_UNKNOWN[5] = {0x02, 0x01, 0x51, 0x09, 0x06};

typedef struct {
    char ch;
    uint8_t col[5];
} font_glyph_t;

static const font_glyph_t FONT_5X7[] = {
    {'!', {0x00, 0x00, 0x5f, 0x00, 0x00}},
    {'%', {0x23, 0x13, 0x08, 0x64, 0x62}},
    {'-', {0x08, 0x08, 0x08, 0x08, 0x08}},
    {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
    {'/', {0x20, 0x10, 0x08, 0x04, 0x02}},
    {'0', {0x3e, 0x51, 0x49, 0x45, 0x3e}},
    {'1', {0x00, 0x42, 0x7f, 0x40, 0x00}},
    {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
    {'3', {0x21, 0x41, 0x45, 0x4b, 0x31}},
    {'4', {0x18, 0x14, 0x12, 0x7f, 0x10}},
    {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
    {'6', {0x3c, 0x4a, 0x49, 0x49, 0x30}},
    {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
    {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
    {'9', {0x06, 0x49, 0x49, 0x29, 0x1e}},
    {':', {0x00, 0x36, 0x36, 0x00, 0x00}},
    {'>', {0x41, 0x22, 0x14, 0x08, 0x00}},
    {'A', {0x7e, 0x11, 0x11, 0x11, 0x7e}},
    {'B', {0x7f, 0x49, 0x49, 0x49, 0x36}},
    {'C', {0x3e, 0x41, 0x41, 0x41, 0x22}},
    {'D', {0x7f, 0x41, 0x41, 0x22, 0x1c}},
    {'E', {0x7f, 0x49, 0x49, 0x49, 0x41}},
    {'F', {0x7f, 0x09, 0x09, 0x09, 0x01}},
    {'G', {0x3e, 0x41, 0x49, 0x49, 0x7a}},
    {'H', {0x7f, 0x08, 0x08, 0x08, 0x7f}},
    {'I', {0x00, 0x41, 0x7f, 0x41, 0x00}},
    {'J', {0x20, 0x40, 0x41, 0x3f, 0x01}},
    {'K', {0x7f, 0x08, 0x14, 0x22, 0x41}},
    {'L', {0x7f, 0x40, 0x40, 0x40, 0x40}},
    {'M', {0x7f, 0x02, 0x0c, 0x02, 0x7f}},
    {'N', {0x7f, 0x04, 0x08, 0x10, 0x7f}},
    {'O', {0x3e, 0x41, 0x41, 0x41, 0x3e}},
    {'P', {0x7f, 0x09, 0x09, 0x09, 0x06}},
    {'Q', {0x3e, 0x41, 0x51, 0x21, 0x5e}},
    {'R', {0x7f, 0x09, 0x19, 0x29, 0x46}},
    {'S', {0x46, 0x49, 0x49, 0x49, 0x31}},
    {'T', {0x01, 0x01, 0x7f, 0x01, 0x01}},
    {'U', {0x3f, 0x40, 0x40, 0x40, 0x3f}},
    {'V', {0x1f, 0x20, 0x40, 0x20, 0x1f}},
    {'W', {0x3f, 0x40, 0x38, 0x40, 0x3f}},
    {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},
    {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}},
    {'Z', {0x61, 0x51, 0x49, 0x45, 0x43}},
    {'_', {0x40, 0x40, 0x40, 0x40, 0x40}},
};

static const uint8_t *font_lookup(char ch)
{
    if (ch >= 'a' && ch <= 'z') {
        ch = (char)(ch - 'a' + 'A');
    }
    if (ch == ' ') {
        return FONT_SPACE;
    }
    for (size_t i = 0; i < sizeof(FONT_5X7) / sizeof(FONT_5X7[0]); i++) {
        if (FONT_5X7[i].ch == ch) {
            return FONT_5X7[i].col;
        }
    }
    return FONT_UNKNOWN;
}

static void oled_clear_fb(void)
{
    memset(g_oled_fb, 0, sizeof(g_oled_fb));
}

static void oled_draw_pixel(int x, int y, bool color)
{
    if (x < 0 || x >= OLED_WIDTH || y < 0 || y >= OLED_HEIGHT) {
        return;
    }

    uint8_t *dst = &g_oled_fb[(y / 8) * OLED_WIDTH + x];
    uint8_t mask = (uint8_t)(1U << (y % 8));
    if (color) {
        *dst |= mask;
    } else {
        *dst &= (uint8_t)~mask;
    }
}

static void oled_fill_rect(int x, int y, int w, int h, bool color)
{
    for (int yy = y; yy < y + h; yy++) {
        for (int xx = x; xx < x + w; xx++) {
            oled_draw_pixel(xx, yy, color);
        }
    }
}

static void oled_draw_rect(int x, int y, int w, int h, bool color)
{
    for (int xx = x; xx < x + w; xx++) {
        oled_draw_pixel(xx, y, color);
        oled_draw_pixel(xx, y + h - 1, color);
    }
    for (int yy = y; yy < y + h; yy++) {
        oled_draw_pixel(x, yy, color);
        oled_draw_pixel(x + w - 1, yy, color);
    }
}

static void oled_fill_page(int page, uint8_t value)
{
    if (page < 0 || page >= OLED_PAGES) {
        return;
    }
    memset(&g_oled_fb[page * OLED_WIDTH], value, OLED_WIDTH);
}

static esp_err_t oled_write_command(uint8_t command)
{
    if (!g_oled_spi) {
        return ESP_ERR_INVALID_STATE;
    }

    spi_transaction_t transaction = {
        .flags = SPI_TRANS_USE_TXDATA,
        .length = 8,
        .tx_data = {command},
    };

    gpio_set_level(OLED_DC_GPIO, 0);
    return spi_device_transmit(g_oled_spi, &transaction);
}

static esp_err_t oled_write_data(const uint8_t *data, size_t length)
{
    if (!g_oled_spi) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!data || length == 0) {
        return ESP_OK;
    }

    spi_transaction_t transaction = {
        .length = length * 8,
        .tx_buffer = data,
    };

    gpio_set_level(OLED_DC_GPIO, 1);
    return spi_device_transmit(g_oled_spi, &transaction);
}

static void oled_draw_char_ex(int x, int page, char ch, bool inverted)
{
    if (page < 0 || page >= OLED_PAGES || x >= OLED_WIDTH) {
        return;
    }
    const uint8_t *glyph = font_lookup(ch);
    for (int col = 0; col < 6; col++) {
        int dst_x = x + col;
        if (dst_x >= 0 && dst_x < OLED_WIDTH) {
            uint8_t pixels = (col < 5) ? glyph[col] : 0x00;
            g_oled_fb[page * OLED_WIDTH + dst_x] = inverted ? (uint8_t)~pixels : pixels;
        }
    }
}

static void oled_draw_text_ex(int x, int page, const char *text, bool inverted)
{
    if (!text) {
        return;
    }
    while (*text && x < OLED_WIDTH) {
        oled_draw_char_ex(x, page, *text++, inverted);
        x += 6;
    }
}

static void oled_draw_text(int x, int page, const char *text)
{
    oled_draw_text_ex(x, page, text, false);
}

static int oled_text_width(const char *text)
{
    return text ? (int)strlen(text) * 6 : 0;
}

static void oled_draw_text_right(int right_x, int page, const char *text, bool inverted)
{
    int x = right_x - oled_text_width(text);
    if (x < 0) {
        x = 0;
    }
    oled_draw_text_ex(x, page, text, inverted);
}

static void oled_draw_text_center(int page, const char *text, bool inverted)
{
    int x = (OLED_WIDTH - oled_text_width(text)) / 2;
    if (x < 0) {
        x = 0;
    }
    oled_draw_text_ex(x, page, text, inverted);
}

static void oled_flush(void)
{
    if (!g_oled_ready) {
        return;
    }

    esp_err_t ret = oled_write_command(0x21);
    if (ret == ESP_OK) {
        ret = oled_write_command(0x00);
    }
    if (ret == ESP_OK) {
        ret = oled_write_command(OLED_WIDTH - 1);
    }
    if (ret == ESP_OK) {
        ret = oled_write_command(0x22);
    }
    if (ret == ESP_OK) {
        ret = oled_write_command(0x00);
    }
    if (ret == ESP_OK) {
        ret = oled_write_command(OLED_PAGES - 1);
    }
    if (ret == ESP_OK) {
        ret = oled_write_data(g_oled_fb, sizeof(g_oled_fb));
    }
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "OLED hardware SPI flush failed: %s", esp_err_to_name(ret));
    }
}

static void format_param_value(sys_config_param_id_t param, char *buf, size_t buf_size)
{
    int value = sys_config_get_param_display_value(param);
    if (param == SYS_CONFIG_PARAM_MOTION_THRESHOLD) {
        if (value < 10) {
            value = 10;
        } else if (value > 100) {
            value = 100;
        }
        snprintf(buf, buf_size, "%u.%u", (unsigned)(value / 10), (unsigned)(value % 10));
    } else if (param == SYS_CONFIG_PARAM_GESTURE_COOLDOWN) {
        snprintf(buf,
                 buf_size,
                 "%u.%us",
                 (unsigned)(value / 2),
                 (unsigned)((value % 2) * 5));
    } else {
        snprintf(buf, buf_size, "%u%s", (unsigned)value, sys_config_get_param_unit(param));
    }
}

static void get_param_range(sys_config_param_id_t param, int *min_value, int *max_value)
{
    switch (param) {
    case SYS_CONFIG_PARAM_MOTION_THRESHOLD:
        *min_value = 10;
        *max_value = 100;
        break;
    case SYS_CONFIG_PARAM_MIN_CONFIDENCE:
        *min_value = 50;
        *max_value = 99;
        break;
    case SYS_CONFIG_PARAM_REQUIRED_TRIGGER_FRAMES:
        *min_value = 1;
        *max_value = 20;
        break;
    case SYS_CONFIG_PARAM_GESTURE_COOLDOWN:
        *min_value = 0;
        *max_value = 20;
        break;
    case SYS_CONFIG_PARAM_DEBOUNCE_FRAMES:
    default:
        *min_value = 1;
        *max_value = 50;
        break;
    }
}

static int param_progress_width(sys_config_param_id_t param, int max_width)
{
    int min_value = 0;
    int max_value = 1;
    get_param_range(param, &min_value, &max_value);

    int value = sys_config_get_param_display_value(param);
    if (value < min_value) {
        value = min_value;
    }
    if (value > max_value) {
        value = max_value;
    }

    int range = max_value - min_value;
    if (range <= 0) {
        return 0;
    }
    return ((value - min_value) * max_width) / range;
}

static void oled_draw_param_screen(void)
{
    char line[48];
    char value_text[16];

    oled_clear_fb();
    snprintf(line, sizeof(line), "M%u  PARAM %u/%u",
             (unsigned)(nn_model_get_active_id() + 1),
             (unsigned)((int)g_selected_param + 1),
             (unsigned)SYS_CONFIG_PARAM_COUNT);
    oled_draw_text(0, 0, line);
    oled_draw_text_right(OLED_WIDTH, 0, nn_model_get_active_name(), false);

    for (int i = 0; i < SYS_CONFIG_PARAM_COUNT; i++) {
        sys_config_param_id_t param = (sys_config_param_id_t)i;
        int page = 1 + i;
        bool selected = (param == g_selected_param);

        if (selected) {
            oled_fill_page(page, 0xff);
        }

        snprintf(line, sizeof(line), "%s", sys_config_get_param_label(param));
        format_param_value(param, value_text, sizeof(value_text));
        oled_draw_text_ex(6, page, line, selected);
        oled_draw_text_right(124, page, value_text, selected);
    }

    int bar_width = OLED_WIDTH - 4;
    int fill_width = param_progress_width(g_selected_param, bar_width - 4);
    oled_draw_rect(2, 54, bar_width, 7, true);
    oled_fill_rect(4, 56, fill_width, 3, true);
    oled_flush();
}

static void oled_draw_model_screen(void)
{
    char line[32];
    int count = nn_model_get_slot_count();
    const int visible_rows = 4;

    oled_clear_fb();
    oled_draw_text_center(0, "SELECT MODEL", false);
    if (count <= 0) {
        oled_draw_text_center(3, "NO SD MODEL", false);
        oled_draw_text_center(6, "CHECK SD", false);
        oled_flush();
        return;
    }

    int start = g_model_cursor - visible_rows + 1;
    if (start < 0) {
        start = 0;
    }
    if (start > count - visible_rows) {
        start = count - visible_rows;
    }
    if (start < 0) {
        start = 0;
    }

    for (int row = 0; row < visible_rows && start + row < count; row++) {
        int model_id = start + row;
        bool selected = (model_id == g_model_cursor);
        const char *state = nn_model_is_installed(model_id) ? "READY" : "EMPTY";
        int page = 2 + row;

        if (selected) {
            oled_fill_page(page, 0xff);
        }
        snprintf(line, sizeof(line), "%s", nn_model_get_slot_name(model_id));
        oled_draw_text_ex(6, page, line, selected);
        oled_draw_text_right(124, page, state, selected);
    }
    snprintf(line, sizeof(line), "%u/%u",
             (unsigned)(g_model_cursor + 1),
             (unsigned)count);
    oled_draw_text(0, 7, line);
    oled_draw_text_right(124, 7, "CLICK OK", false);
    oled_flush();
}

static void oled_draw_notice_screen(void)
{
    oled_clear_fb();
    oled_draw_text_center(1, "NOTICE", false);
    oled_fill_page(3, 0xff);
    oled_draw_text_center(3, g_notice, true);
    oled_flush();
}

static void ui_render(void)
{
    switch (g_screen) {
    case UI_SCREEN_MODEL_SELECT:
        oled_draw_model_screen();
        break;
    case UI_SCREEN_NOTICE:
        oled_draw_notice_screen();
        break;
    case UI_SCREEN_PARAMS:
    default:
        oled_draw_param_screen();
        break;
    }
}

static esp_err_t oled_init(void)
{
    ESP_LOGI(TAG,
             "OLED hardware SPI pins: SCK=%d MOSI/SDA=%d RST=%d DC=%d CS=%d clk=%dHz",
             OLED_SCK_GPIO,
             OLED_MOSI_GPIO,
             OLED_RST_GPIO,
             OLED_DC_GPIO,
             OLED_CS_GPIO,
             OLED_SPI_CLOCK_HZ);

    gpio_config_t output_cfg = {
        .pin_bit_mask = (1ULL << OLED_RST_GPIO) |
                        (1ULL << OLED_DC_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&output_cfg), TAG, "OLED GPIO config failed");

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = OLED_MOSI_GPIO,
        .miso_io_num = GPIO_NUM_NC,
        .sclk_io_num = OLED_SCK_GPIO,
        .quadwp_io_num = GPIO_NUM_NC,
        .quadhd_io_num = GPIO_NUM_NC,
        .max_transfer_sz = OLED_FB_SIZE,
    };
    esp_err_t ret = spi_bus_initialize(OLED_SPI_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "OLED SPI bus init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    spi_device_interface_config_t dev_cfg = {
        .clock_speed_hz = OLED_SPI_CLOCK_HZ,
        .mode = 0,
        .spics_io_num = OLED_CS_GPIO,
        .queue_size = 4,
    };
    ESP_RETURN_ON_ERROR(spi_bus_add_device(OLED_SPI_HOST, &dev_cfg, &g_oled_spi),
                        TAG, "OLED SPI device add failed");

    gpio_set_level(OLED_RST_GPIO, 0);
    vTaskDelay(pdMS_TO_TICKS(50));
    gpio_set_level(OLED_RST_GPIO, 1);
    vTaskDelay(pdMS_TO_TICKS(50));

    const uint8_t init_commands[] = {
        0xAE,       // display off
        0xD5, 0x80, // clock divide
        0xA8, 0x3F, // multiplex ratio 1/64
        0xD3, 0x00, // display offset
        0x40,       // display start line
        0x8D, 0x14, // charge pump on
        0x20, 0x00, // horizontal addressing mode
        0xA1,       // segment remap
        0xC8,       // COM scan direction remap
        0xDA, 0x12, // COM pins
        0x81, 0xCF, // contrast
        0xD9, 0xF1, // pre-charge
        0xDB, 0x40, // VCOMH
        0xA4,       // resume RAM display
        0xA6,       // normal display
        0xAF,       // display on
    };
    for (size_t i = 0; i < sizeof(init_commands); i++) {
        ESP_RETURN_ON_ERROR(oled_write_command(init_commands[i]), TAG, "OLED init command failed");
    }

    g_oled_ready = true;
    oled_clear_fb();
    oled_flush();
    ESP_LOGI(TAG, "OLED SSD1306 hardware SPI init done");
    return ESP_OK;
}

static esp_err_t encoder_init(void)
{
    gpio_config_t input_cfg = {
        .pin_bit_mask = (1ULL << ENCODER_A_GPIO) |
                        (1ULL << ENCODER_B_GPIO) |
                        (1ULL << ENCODER_SW_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_ANYEDGE,
    };
    ESP_RETURN_ON_ERROR(gpio_config(&input_cfg), TAG, "encoder GPIO config failed");
    return ESP_OK;
}

static void IRAM_ATTR ui_gpio_isr_handler(void *arg)
{
    if (g_ui_event_queue == NULL) {
        return;
    }

    gpio_num_t gpio = (gpio_num_t)(intptr_t)arg;
    ui_gpio_event_t event = {
        .type = (gpio == ENCODER_SW_GPIO) ? UI_GPIO_EVENT_BUTTON : UI_GPIO_EVENT_ENCODER,
        .tick = xTaskGetTickCountFromISR(),
        .encoder_state = (uint8_t)(((gpio_get_level(ENCODER_A_GPIO) ? 1 : 0) << 1) |
                                   (gpio_get_level(ENCODER_B_GPIO) ? 1 : 0)),
        .button_pressed = (gpio_get_level(ENCODER_SW_GPIO) == 0),
    };

    BaseType_t high_task_woken = pdFALSE;
    (void)xQueueSendFromISR(g_ui_event_queue, &event, &high_task_woken);
    if (high_task_woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

static esp_err_t ui_gpio_interrupts_init(void)
{
    esp_err_t ret = gpio_install_isr_service(0);
    if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "GPIO ISR service init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_RETURN_ON_ERROR(gpio_isr_handler_add(ENCODER_A_GPIO, ui_gpio_isr_handler, (void *)(intptr_t)ENCODER_A_GPIO),
                        TAG, "encoder A ISR add failed");
    ESP_RETURN_ON_ERROR(gpio_isr_handler_add(ENCODER_B_GPIO, ui_gpio_isr_handler, (void *)(intptr_t)ENCODER_B_GPIO),
                        TAG, "encoder B ISR add failed");
    ESP_RETURN_ON_ERROR(gpio_isr_handler_add(ENCODER_SW_GPIO, ui_gpio_isr_handler, (void *)(intptr_t)ENCODER_SW_GPIO),
                        TAG, "encoder button ISR add failed");

    return ESP_OK;
}

static void enter_notice(const char *text, uint32_t duration_ms)
{
    snprintf(g_notice, sizeof(g_notice), "%s", text ? text : "");
    g_screen = UI_SCREEN_NOTICE;
    g_notice_until = xTaskGetTickCount() + pdMS_TO_TICKS(duration_ms);
    ui_render();
}

static void handle_encoder_delta(int delta)
{
    if (delta == 0) {
        return;
    }

    if (g_screen == UI_SCREEN_NOTICE) {
        g_screen = UI_SCREEN_PARAMS;
    }

    if (g_screen == UI_SCREEN_MODEL_SELECT) {
        int count = nn_model_get_slot_count();
        if (count <= 0) {
            ui_render();
            return;
        }
        g_model_cursor = (g_model_cursor + delta) % count;
        if (g_model_cursor < 0) {
            g_model_cursor += count;
        }
    } else {
        (void)sys_config_adjust_param(g_selected_param, delta);
    }
    ui_render();
}

static void handle_short_click(void)
{
    if (g_screen == UI_SCREEN_NOTICE) {
        g_screen = UI_SCREEN_PARAMS;
        ui_render();
        return;
    }

    if (g_screen == UI_SCREEN_MODEL_SELECT) {
        int count = nn_model_get_slot_count();
        if (count <= 0) {
            enter_notice("NO SD MODEL", 1200);
        } else if (nn_model_is_installed(g_model_cursor)) {
            if (nn_model_select(g_model_cursor) == ESP_OK) {
                g_screen = UI_SCREEN_PARAMS;
                ui_render();
            } else {
                enter_notice("MODEL ERR", 1200);
            }
        } else {
            enter_notice("MODEL EMPTY", 1200);
        }
        return;
    }

    g_selected_param = (sys_config_param_id_t)((g_selected_param + 1) % SYS_CONFIG_PARAM_COUNT);
    ui_render();
}

static void handle_long_press(void)
{
    if (nn_model_get_slot_count() <= 0) {
        nn_model_rescan();
    }
    g_screen = UI_SCREEN_MODEL_SELECT;
    g_model_cursor = nn_model_get_active_id();
    int count = nn_model_get_slot_count();
    if (count <= 0) {
        g_model_cursor = 0;
    } else if (g_model_cursor >= count) {
        g_model_cursor = count - 1;
    }
    ui_render();
}

static uint8_t encoder_read_state(void)
{
    return (uint8_t)(((gpio_get_level(ENCODER_A_GPIO) ? 1 : 0) << 1) |
                     (gpio_get_level(ENCODER_B_GPIO) ? 1 : 0));
}

static bool button_read_pressed(void)
{
    return gpio_get_level(ENCODER_SW_GPIO) == 0;
}

static int encoder_process_state(uint8_t current_state)
{
    if (!g_encoder_initialized) {
        g_encoder_last_state = current_state;
        g_encoder_initialized = true;
        return 0;
    }

    int8_t movement = g_encoder_table[(g_encoder_last_state << 2) | current_state];
    g_encoder_last_state = current_state;
    if (movement == 0) {
        return 0;
    }

    g_encoder_accumulator += movement;
    if (current_state == 0x03) {
        if (g_encoder_accumulator >= 4) {
            g_encoder_accumulator = 0;
            return 1;
        }
        if (g_encoder_accumulator <= -4) {
            g_encoder_accumulator = 0;
            return -1;
        }
        g_encoder_accumulator = 0;
    }

    return 0;
}

static void button_note_raw_state(bool raw_pressed, TickType_t tick)
{
    if (raw_pressed != g_button_raw_pressed_last) {
        g_button_raw_pressed_last = raw_pressed;
        g_button_last_raw_change_tick = tick;
    }
}

static button_event_t button_process_timers(TickType_t now)
{
    if ((now - g_button_last_raw_change_tick) >= pdMS_TO_TICKS(BUTTON_DEBOUNCE_MS) &&
        g_button_raw_pressed_last != g_button_stable_pressed) {
        g_button_stable_pressed = g_button_raw_pressed_last;
        if (g_button_stable_pressed) {
            g_button_press_start_tick = now;
            g_button_long_press_reported = false;
        } else if (!g_button_long_press_reported) {
            return BUTTON_EVENT_CLICK;
        }
    }

    if (g_button_stable_pressed && !g_button_long_press_reported &&
        (now - g_button_press_start_tick) >= pdMS_TO_TICKS(BUTTON_LONG_PRESS_MS)) {
        g_button_long_press_reported = true;
        return BUTTON_EVENT_LONG_PRESS;
    }

    return BUTTON_EVENT_NONE;
}

static bool ui_tick_due(TickType_t now, TickType_t deadline)
{
    return (TickType_t)(now - deadline) < (TickType_t)(portMAX_DELAY / 2);
}

static TickType_t ui_ticks_until(TickType_t now, TickType_t deadline)
{
    if (ui_tick_due(now, deadline)) {
        return 0;
    }
    return deadline - now;
}

static TickType_t ui_min_wait(TickType_t current, TickType_t candidate)
{
    return (candidate < current) ? candidate : current;
}

static TickType_t ui_next_wait_ticks(void)
{
    TickType_t now = xTaskGetTickCount();
    TickType_t wait_ticks = portMAX_DELAY;

    if (g_button_raw_pressed_last != g_button_stable_pressed) {
        TickType_t deadline = g_button_last_raw_change_tick + pdMS_TO_TICKS(BUTTON_DEBOUNCE_MS);
        wait_ticks = ui_min_wait(wait_ticks, ui_ticks_until(now, deadline));
    }

    if (g_button_stable_pressed && !g_button_long_press_reported) {
        TickType_t deadline = g_button_press_start_tick + pdMS_TO_TICKS(BUTTON_LONG_PRESS_MS);
        wait_ticks = ui_min_wait(wait_ticks, ui_ticks_until(now, deadline));
    }

    if (g_screen == UI_SCREEN_NOTICE) {
        wait_ticks = ui_min_wait(wait_ticks, ui_ticks_until(now, g_notice_until));
    }

    return wait_ticks;
}

static void ui_process_button_event(button_event_t button_event)
{
    if (button_event == BUTTON_EVENT_CLICK) {
        handle_short_click();
    } else if (button_event == BUTTON_EVENT_LONG_PRESS) {
        handle_long_press();
    }
}

static void ui_process_timers(void)
{
    TickType_t now = xTaskGetTickCount();
    ui_process_button_event(button_process_timers(now));

    if (g_screen == UI_SCREEN_NOTICE && ui_tick_due(now, g_notice_until)) {
        g_screen = UI_SCREEN_PARAMS;
        ui_render();
    }
}

static void ui_process_gpio_event(const ui_gpio_event_t *event)
{
    if (event->type == UI_GPIO_EVENT_ENCODER) {
        int encoder_delta = encoder_process_state(event->encoder_state);
        if (encoder_delta != 0) {
            handle_encoder_delta(encoder_delta);
        }
        return;
    }

    button_note_raw_state(event->button_pressed, event->tick);
}

static void ui_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting UI task on Core %d", xPortGetCoreID());

    ui_render();

    while (true) {
        ui_process_timers();

        ui_gpio_event_t event;
        if (xQueueReceive(g_ui_event_queue, &event, ui_next_wait_ticks()) == pdTRUE) {
            do {
                ui_process_gpio_event(&event);
            } while (xQueueReceive(g_ui_event_queue, &event, 0) == pdTRUE);
        }

        ui_process_timers();
    }
}

esp_err_t ui_controller_init(void)
{
    esp_err_t ret = oled_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "OLED init failed: %s", esp_err_to_name(ret));
    }

    ret = encoder_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Encoder init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    g_encoder_last_state = encoder_read_state();
    g_encoder_initialized = true;
    g_encoder_accumulator = 0;

    g_button_raw_pressed_last = button_read_pressed();
    g_button_stable_pressed = g_button_raw_pressed_last;
    g_button_long_press_reported = false;
    g_button_last_raw_change_tick = xTaskGetTickCount();
    g_button_press_start_tick = g_button_last_raw_change_tick;

    g_ui_event_queue = xQueueCreate(UI_EVENT_QUEUE_LEN, sizeof(ui_gpio_event_t));
    if (g_ui_event_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create UI event queue");
        return ESP_ERR_NO_MEM;
    }

    ret = ui_gpio_interrupts_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "UI GPIO interrupt init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    BaseType_t task_ret = xTaskCreatePinnedToCore(
        ui_task,
        "ui_task",
        4096,
        NULL,
        UI_TASK_PRIORITY,
        NULL,
        1);
    if (task_ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create UI task");
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}
