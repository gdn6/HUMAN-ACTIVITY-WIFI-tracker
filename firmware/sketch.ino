// ================================================================
//  WiFi CSI Pose Estimator — Genesis Mini
//  Flash to BOTH boards, changing BOARD_ROLE before each flash:
//    #define BOARD_ROLE 0  →  TRANSMITTER (Board 1, no USB needed)
//    #define BOARD_ROLE 1  →  RECEIVER    (Board 2, connect to PC)
// ================================================================
#define BOARD_ROLE 1   // 0 = TX · 1 = RX  ← change before flashing

#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"
#include <math.h>

#define WIFI_CH         1
#define TX_INTERVAL_MS  20
#define OUT_INTERVAL_MS 150

static uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static uint8_t txPayload[]     = {0xAB, 0xCD, 0xEF, 0x01};

#define MAX_SC 128
static volatile bool csiReady = false;
static float         csiAmps[MAX_SC];
static int           csiCount = 0;
static portMUX_TYPE  csiMux   = portMUX_INITIALIZER_UNLOCKED;

#if BOARD_ROLE == 1
static void csiCallback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len < 2) return;
    int n = (int)(info->len / 2);
    if (n > MAX_SC) n = MAX_SC;
    float tmp[MAX_SC];
    int8_t *b = info->buf;
    for (int i = 0; i < n; i++) {
        float re = (float)b[2*i], im = (float)b[2*i+1];
        tmp[i] = sqrtf(re*re + im*im);
    }
    portENTER_CRITICAL(&csiMux);
    memcpy(csiAmps, tmp, n * sizeof(float));
    csiCount = n; csiReady = true;
    portEXIT_CRITICAL(&csiMux);
}
#endif

void setup() {
    Serial.begin(115200);
    delay(1500);
#if BOARD_ROLE == 0
    Serial.println("BOOT:TRANSMITTER");
    WiFi.mode(WIFI_STA); WiFi.disconnect(); delay(100);
    esp_wifi_set_channel(WIFI_CH, WIFI_SECOND_CHAN_NONE);
    if (esp_now_init() != ESP_OK) { Serial.println("ERR:ESP-NOW init failed"); return; }
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, broadcastAddr, 6);
    peer.channel = WIFI_CH; peer.encrypt = false;
    esp_now_add_peer(&peer);
    Serial.printf("TX:channel=%d rate=%dHz\n", WIFI_CH, 1000/TX_INTERVAL_MS);
#else
    Serial.println("BOOT:RECEIVER");
    WiFi.mode(WIFI_STA); WiFi.disconnect(); delay(100);
    esp_wifi_set_channel(WIFI_CH, WIFI_SECOND_CHAN_NONE);
    if (esp_now_init() == ESP_OK) Serial.println("INFO:ESP-NOW ready");
    else Serial.println("WARN:ESP-NOW init failed");
    wifi_csi_config_t cfg = {};
    cfg.lltf_en=true; cfg.htltf_en=false; cfg.stbc_htltf2_en=false;
    cfg.ltf_merge_en=true; cfg.channel_filter_en=false; cfg.manu_scale=false; cfg.shift=0;
    esp_wifi_set_csi_config(&cfg);
    esp_wifi_set_csi_rx_cb(csiCallback, NULL);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_csi(true);
    Serial.printf("RX:channel=%d output=%dHz\n", WIFI_CH, 1000/OUT_INTERVAL_MS);
#endif
}

static unsigned long lastTx=0, lastOut=0;
void loop() {
#if BOARD_ROLE == 0
    unsigned long now = millis();
    if (now - lastTx >= TX_INTERVAL_MS) { lastTx=now; esp_now_send(broadcastAddr, txPayload, sizeof(txPayload)); }
#else
    { static unsigned long _hb=0; if (millis()-_hb>=2000) { _hb=millis(); Serial.println("HB:alive"); } }
    if (!csiReady) return;
    unsigned long now = millis();
    if (now - lastOut < OUT_INTERVAL_MS) return;
    lastOut = now;
    float snap[MAX_SC]; int n;
    portENTER_CRITICAL(&csiMux);
    memcpy(snap, csiAmps, csiCount*sizeof(float)); n=csiCount; csiReady=false;
    portEXIT_CRITICAL(&csiMux);
    if (n==0) return;
    float mean=0, mn=1e9f, mx=0;
    for (int i=0;i<n;i++) { mean+=snap[i]; if(snap[i]<mn)mn=snap[i]; if(snap[i]>mx)mx=snap[i]; }
    mean/=(float)n;
    float var=0;
    for (int i=0;i<n;i++) { float d=snap[i]-mean; var+=d*d; } var/=(float)n;
    Serial.print("AMP:");
    for (int i=0;i<n;i++) { Serial.print((int)snap[i]); if(i<n-1) Serial.print(','); }
    Serial.println();
    Serial.printf("FEAT:%.2f,%.2f,%.2f\n", mean, sqrtf(var), mx-mn);
#endif
}
