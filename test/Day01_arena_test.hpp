#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "StaticArena.hpp"

inline void run_day01_arena_test() {
    static const char* TAG = "MODUL1_ARENA";

    // 'static' olduğu için Stack'te DEĞİL, DRAM (.bss) bölgesinde yaşar
    static StaticArena<65536> g_arena;

    // 'constexpr' olduğu için Flash (DROM .rodata) bölgesinde yaşar (0 B DRAM)
    static constexpr float kDromWeights[4] = {0.25f, -0.50f, 0.75f, 1.00f};

    // 1. Test öncesi bellek fotoğrafı
    const size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const size_t block_before = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);

    volatile float sink = 0.0f;
    bool alloc_ok = true;
    bool alignment_ok = true;

    void* first_t1_addr = nullptr;
    void* first_t2_addr = nullptr;
    void* first_t3_addr = nullptr;

    // 2. 10.000 iterasyonluk sıfır-sızıntı stres testi
    const int64_t t_start = esp_timer_get_time();

    for (int i = 0; i < 10000; ++i) {
        float* t1 = g_arena.alloc<float>(5);      // 20 B  -> offset: 0..19
        float* t2 = g_arena.alloc<float>(128);    // 512 B -> +12 B pad -> offset: 32..543
        uint8_t* t3 = g_arena.alloc<uint8_t>(3);  // 3 B   -> +0 B pad  -> offset: 544..546

        if (!t1 || !t2 || !t3) {
            alloc_ok = false;
            break;
        }

        if (i == 0) {
            first_t1_addr = t1;
            first_t2_addr = t2;
            first_t3_addr = t3;
        }

        if ((reinterpret_cast<uintptr_t>(t1) & 15U) != 0U ||
            (reinterpret_cast<uintptr_t>(t2) & 15U) != 0U ||
            (reinterpret_cast<uintptr_t>(t3) & 15U) != 0U) {
            alignment_ok = false;
            break;
        }

        t1[0] = kDromWeights[i & 3];
        t2[0] = t1[0] * 2.0f;
        t3[0] = static_cast<uint8_t>(i & 0xFF);
        sink = t2[0];

        g_arena.reset();
    }

    const int64_t elapsed_us = esp_timer_get_time() - t_start;

    // 3. Test sonrası bellek fotoğrafı
    const size_t free_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const size_t block_after = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
    const long free_diff = static_cast<long>(free_after) - static_cast<long>(free_before);

    // 4. Seri monitör raporu
    ESP_LOGI(TAG, "=== MODUL 1: STATIC ARENA DOGRULAMA RAPORU ===");
    ESP_LOGI(TAG, "DROM Agirlik Adresi (Flash) : %p", static_cast<const void*>(kDromWeights));
    ESP_LOGI(TAG, "DRAM Arena Adresi (t1)      : %p", first_t1_addr);
    ESP_LOGI(TAG, "DRAM t2 Adresi (+32 B)      : %p", first_t2_addr);
    ESP_LOGI(TAG, "DRAM t3 Adresi (+544 B)     : %p", first_t3_addr);
    ESP_LOGI(TAG, "----------------------------------------------");
    ESP_LOGI(TAG, "Test Suresi (10.000 tur)    : %lld us (sink=%.2f)",
             static_cast<long long>(elapsed_us), sink);
    ESP_LOGI(TAG, "Tahsis Basarisi             : %s", alloc_ok ? "EVET" : "HAYIR");
    ESP_LOGI(TAG, "16-Byte Hizalama            : %s", alignment_ok ? "DOGRULANDI" : "HATA");
    ESP_LOGI(TAG, "Arena Zirve Kullanim        : %zu / %zu B",
             g_arena.peak_bytes(), g_arena.capacity());
    ESP_LOGI(TAG, "Reset Sonrasi Kullanim      : %zu B", g_arena.used_bytes());
    ESP_LOGI(TAG, "Bos DRAM (Once -> Sonra)    : %zu -> %zu B (Fark: %ld B)",
             free_before, free_after, free_diff);
    ESP_LOGI(TAG, "En Buyuk Blok (Once->Sonra) : %zu -> %zu B", block_before, block_after);
}