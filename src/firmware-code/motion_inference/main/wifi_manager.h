#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize and connect Wi-Fi in Station mode
 * 
 * Automatically handles NVS flash initialization, event loops, netif,
 * and remote Wi-Fi configuration.
 */
void wifi_manager_init(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_MANAGER_H
