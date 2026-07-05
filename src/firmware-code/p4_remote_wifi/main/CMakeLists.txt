idf_component_register(SRCS  "main.c" "wifi_manager.c" "nn_model.cpp" "sys_config.c" "csi_source.c" "csi_dsp.cpp" "gesture_detector.c" "gesture_output.c" "ui_controller.c" "upload_api.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES
                        esp_driver_gpio
                        esp_driver_sdmmc
                        esp_driver_spi
                        esp_driver_uart
                        esp_event
                        esp_http_client
                        esp_netif
                        esp_timer
                        esp_wifi
                        fatfs
                        heap
                        json
                        log
                        sdmmc)
