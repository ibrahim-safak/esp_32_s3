#pragma once

#include <cstddef>
#include <cstdint>

// Veriyi kopyalamayan, yalnızca DRAM'deki bir bölgeye bakan 16-Byte'lık gözlük.
// ESP32-S3 (32-bit): 4 (ptr) + 2 (rows) + 2 (cols) + 4 (row_stride) + 4 (col_stride) = 16 B
struct TensorView {
    float* data_ptr = nullptr;
    uint16_t rows = 0;
    uint16_t cols = 0;
    uint32_t row_stride = 0;
    uint32_t col_stride = 0;

    // 1D ham dizi üzerine Zero-Copy görünüm oluştur
    static TensorView make_1d(float* ptr, uint16_t length) {
        if (!ptr || length == 0) {
            return {};
        }
        return TensorView{ptr, 1, length, length, 1};
    }

    // 2D Row-Major (satır öncelikli) görünüm oluştur
    static TensorView make_2d(float* ptr, uint16_t num_rows, uint16_t num_cols) {
        if (!ptr || num_rows == 0 || num_cols == 0) {
            return {};
        }
        return TensorView{ptr, num_rows, num_cols, num_cols, 1};
    }

    bool valid() const {
        return data_ptr != nullptr && rows > 0 && cols > 0;
    }

    size_t size() const {
        return static_cast<size_t>(rows) * static_cast<size_t>(cols);
    }

    // Evrensel 2D eleman erişimi: data_ptr[r * row_stride + c * col_stride]
    float& at(uint16_t r, uint16_t c) {
        return data_ptr[static_cast<uint32_t>(r) * row_stride +
                        static_cast<uint32_t>(c) * col_stride];
    }

    const float& at(uint16_t r, uint16_t c) const {
        return data_ptr[static_cast<uint32_t>(r) * row_stride +
                        static_cast<uint32_t>(c) * col_stride];
    }

    // 1D görünüm için kısa yol erişimi
    float& at_1d(uint16_t i) {
        return data_ptr[static_cast<uint32_t>(i) * col_stride];
    }

    const float& at_1d(uint16_t i) const {
        return data_ptr[static_cast<uint32_t>(i) * col_stride];
    }

    // O(1) Zero-Copy 1D dilimleme: slice(start, length, step)
    TensorView slice(uint16_t start, uint16_t length, uint32_t step = 1) const {
        if (!valid() || rows != 1 || length == 0 || step == 0 || start >= cols) {
            return {};
        }

        // Taşma korumalı üst sınır kontrolü (uint32_t aritmetiği)
        const uint32_t last_idx =
            static_cast<uint32_t>(start) + static_cast<uint32_t>(length - 1) * step;
        if (last_idx >= cols) {
            return {};
        }

        float* new_ptr = data_ptr + (static_cast<uint32_t>(start) * col_stride);
        const uint32_t new_col_stride = col_stride * step;
        const uint32_t new_row_stride = static_cast<uint32_t>(length) * new_col_stride;

        return TensorView{new_ptr, 1, length, new_row_stride, new_col_stride};
    }

    // O(1) Zero-Copy kayan pencere: 1D sinyali 2D (pencere_sayisi x window_len) yapar
    TensorView sliding_windows(uint16_t window_len, uint32_t hop_stride) const {
        if (!valid() || rows != 1 || window_len == 0 || hop_stride == 0 ||
            window_len > cols) {
            return {};
        }

        const uint32_t num_windows =
            ((static_cast<uint32_t>(cols) - window_len) / hop_stride) + 1U;
        if (num_windows > 65535U) {
            return {};
        }

        const uint32_t new_row_stride = hop_stride * col_stride;
        return TensorView{data_ptr,
                          static_cast<uint16_t>(num_windows),
                          window_len,
                          new_row_stride,
                          col_stride};
    }

    // 2D görünümden tek bir satırı (pencereyi) kopyalamadan 1D görünüm olarak çek
    TensorView row(uint16_t r) const {
        if (!valid() || r >= rows) {
            return {};
        }
        float* row_ptr = data_ptr + (static_cast<uint32_t>(r) * row_stride);
        return TensorView{row_ptr, 1, cols, row_stride, col_stride};
    }

    // O(1) Zero-Copy transpoze: sadece şekil ve adım yer değiştirir
    TensorView transpose() const {
        if (!valid()) {
            return {};
        }
        return TensorView{data_ptr, cols, rows, col_stride, row_stride};
    }
};

// 32-bit ESP32-S3 üzerinde TensorView'un tam 16 Byte olduğunu derleme anında kilitle
static_assert(sizeof(void*) != 4 || sizeof(TensorView) == 16,
              "TensorView 32-bit hedefte tam 16 byte olmalidir.");