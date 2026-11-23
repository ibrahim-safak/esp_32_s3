#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "RingBuffer.hpp"
#include "StaticArena.hpp"
#include "TensorView.hpp"

inline void run_day02_slice_test() {
    static const char* TAG = "MODUL2_SLICE";
    static constexpr size_t kBufLen = 512;
    static constexpr uint16_t kWinLen = 64;
    static constexpr uint32_t kHopStride = 32;
    static constexpr float kPi = 3.14159265358979323846f;

    // 1. Gün'den gelen statik arena (Stack'te DEĞİL, DRAM .bss bölgesinde)
    static StaticArena<65536> g_arena;
    g_arena.reset();

    const size_t free_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    // Ana sensör tamponunu StaticArena'dan tek parça ayır (512 * 4 B = 2048 B)
    float* raw_buf = g_arena.alloc<float>(kBufLen);
    if (!raw_buf) {
        ESP_LOGE(TAG, "Arena tahsisi basarisiz!");
        return;
    }

    RingBuffer<kBufLen> rb(raw_buf);

    // 1. Aşama: 512 örneklik sentetik sinüs dalgası bas (4 tam periyot)
    // i=16 -> sin(pi/4) = 0.7071 | i=32 -> sin(pi/2) = 1.0000
    for (size_t i = 0; i < kBufLen; ++i) {
        const float angle = 2.0f * kPi * 4.0f * static_cast<float>(i) /
                            static_cast<float>(kBufLen);
        rb.push(std::sin(angle));
    }

    // 2. Aşama: Zero-Copy Kayan Pencere Dilimleme (15 x 64)
    const int64_t t_start = esp_timer_get_time();

    TensorView signal = rb.raw_buffer_view();
    TensorView windows = signal.sliding_windows(kWinLen, kHopStride);

    const uint8_t* buf_start = reinterpret_cast<const uint8_t*>(raw_buf);
    const uint8_t* buf_end = reinterpret_cast<const uint8_t*>(raw_buf + kBufLen);

    bool all_zero_copy = windows.valid() && windows.rows == 15 && windows.cols == 64;
    bool stride_bytes_ok = true;

    for (uint16_t w = 0; w < windows.rows; ++w) {
        TensorView win = windows.row(w);
        const uint8_t* win_start = reinterpret_cast<const uint8_t*>(win.data_ptr);
        const uint8_t* win_end =
            reinterpret_cast<const uint8_t*>(win.data_ptr + win.cols);

        // Her pencere mutlaka ana tamponun fiziksel adres sınırları içinde kalmalı
        if (win_start < buf_start || win_end > buf_end) {
            all_zero_copy = false;
        }

        // Her pencere başlangıcı tam olarak w * 128 Byte ileride olmalı
        const ptrdiff_t expected_offset =
            static_cast<ptrdiff_t>(w) * static_cast<ptrdiff_t>(kHopStride * sizeof(float));
        if ((win_start - buf_start) != expected_offset) {
            stride_bytes_ok = false;
        }
    }

    // Çakışma (Overlap) kanıtı: W0[32] ve W1[0] aynı fiziksel adresi mi gösteriyor?
    TensorView w0 = windows.row(0);
    TensorView w1 = windows.row(1);
    TensorView w14 = windows.row(14);
    const bool overlap_same_addr = (&w0.at_1d(32) == &w1.at_1d(0));

    // 3. Aşama: 1D Seyreltme — slice(start=10, length=8, step=2)
    TensorView sub = signal.slice(10, 8, 2);
    // sub[3] -> raw_buf[10 + 3*2] = raw_buf[16] -> sin(pi/4) = 0.7071
    const bool slice_ok = sub.valid() &&
                          (sub.data_ptr == raw_buf + 10) &&
                          (&sub.at_1d(3) == &raw_buf[16]);

    const int64_t slice_us = esp_timer_get_time() - t_start;

    // 4. Aşama: RingBuffer Sarma (Wrap-Around) Kanıtı
    // Tampon doluyken (write_idx=0) 10 işaretli örnek daha bas (100.0 .. 109.0)
    for (int k = 0; k < 10; ++k) {
        rb.push(100.0f + static_cast<float>(k));
    }

    const bool wrap_ok = (rb.write_index() == 10) &&
                         (rb.count() == kBufLen) &&
                         (rb.at_oldest(511) == 109.0f);

    const size_t free_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    const long free_diff = static_cast<long>(free_after) - static_cast<long>(free_before);

    // 5. Seri Monitör Raporu
    ESP_LOGI(TAG, "=== MODUL 2: RING BUFFER & ZERO-COPY DOGRULAMA ===");
    ESP_LOGI(TAG, "sizeof(TensorView)          : %zu B (Hedef: 16 B)", sizeof(TensorView));
    ESP_LOGI(TAG, "Ana Tampon Araligi          : %p .. %p (2048 B)",
             static_cast<const void*>(buf_start), static_cast<const void*>(buf_end));
    ESP_LOGI(TAG, "W0  Baslangic (+0 B)        : %p", static_cast<const void*>(w0.data_ptr));
    ESP_LOGI(TAG, "W1  Baslangic (+128 B)      : %p", static_cast<const void*>(w1.data_ptr));
    ESP_LOGI(TAG, "W14 Baslangic (+1792 B)     : %p", static_cast<const void*>(w14.data_ptr));
    ESP_LOGI(TAG, "--------------------------------------------------");
    ESP_LOGI(TAG, "W0[32] Adres / Deger        : %p / %.4f",
             static_cast<const void*>(&w0.at_1d(32)), w0.at_1d(32));
    ESP_LOGI(TAG, "W1[0]  Adres / Deger        : %p / %.4f",
             static_cast<const void*>(&w1.at_1d(0)), w1.at_1d(0));
    ESP_LOGI(TAG, "Cakisma Ayni Adres mi?      : %s", overlap_same_addr ? "EVET" : "HAYIR");
    ESP_LOGI(TAG, "slice(10,8,2)[3] -> buf[16] : %s (Deger: %.4f)",
             slice_ok ? "DOGRULANDI" : "HATA", sub.at_1d(3));
    ESP_LOGI(TAG, "Tum 15 Pencere Zero-Copy    : %s (Adim +128B: %s)",
             all_zero_copy ? "DOGRULANDI" : "HATA", stride_bytes_ok ? "EVET" : "HAYIR");
    ESP_LOGI(TAG, "RingBuffer Wrap (512+10)    : %s (write_idx=%zu, en_yeni=%.1f)",
             wrap_ok ? "DOGRULANDI" : "HATA", rb.write_index(), rb.at_oldest(511));
    ESP_LOGI(TAG, "Dilimleme Suresi            : %lld us", static_cast<long long>(slice_us));
    ESP_LOGI(TAG, "Arena Zirve (Kopya Yok!)    : %zu / %zu B",
             g_arena.peak_bytes(), g_arena.capacity());
    ESP_LOGI(TAG, "Bos DRAM Fark               : %ld B", free_diff);
}