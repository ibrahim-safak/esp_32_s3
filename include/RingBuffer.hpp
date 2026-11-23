#pragma once

#include <cstddef>
#include <cstdint>

#include "TensorView.hpp"

template <size_t Capacity>
class RingBuffer {
public:
    // Bit-maske sarma (& kMask) için kapasite mutlaka 2'nin kuvveti olmalı (örn. 512)
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "RingBuffer kapasitesi 2'nin kuvveti olmalidir (64, 128, 256, 512...).");

    static constexpr size_t kMask = Capacity - 1;

    // Bellek StaticArena'dan dışarıdan enjekte edilir (sıfır malloc / sıfır kopya)
    explicit RingBuffer(float* external_storage) : storage_(external_storage) {}

    // Kopyalamayı yasakla (işaretçi ve yazma durumu tek bir sahibin kontrolünde kalmalı)
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    // Yeni sensör örneğini O(1) sürede yaz ve gerekirse başa sar
    void push(float sample) {
        if (!storage_) {
            return;
        }
        storage_[write_idx_] = sample;
        write_idx_ = (write_idx_ + 1U) & kMask;
        if (count_ < Capacity) {
            ++count_;
        }
    }

    // En eski örnekten başlayarak mantıksal indeksle okuma (0 = en eski, count-1 = en yeni)
    float at_oldest(size_t logical_idx) const {
        if (!storage_ || count_ == 0 || logical_idx >= count_) {
            return 0.0f;
        }
        const size_t oldest_idx = (count_ < Capacity) ? 0U : write_idx_;
        const size_t physical_idx = (oldest_idx + logical_idx) & kMask;
        return storage_[physical_idx];
    }

    // Ham bitişik tamponu doğrudan Zero-Copy TensorView olarak dışarı aç
    TensorView raw_buffer_view() {
        if (!storage_ || count_ == 0) {
            return {};
        }
        return TensorView::make_1d(storage_, static_cast<uint16_t>(count_));
    }

    void reset() {
        write_idx_ = 0;
        count_ = 0;
    }

    float* data() { return storage_; }
    const float* data() const { return storage_; }
    size_t count() const { return count_; }
    size_t write_index() const { return write_idx_; }
    bool is_full() const { return count_ == Capacity; }
    static constexpr size_t capacity() { return Capacity; }

private:
    float* storage_ = nullptr;
    size_t write_idx_ = 0;
    size_t count_ = 0;
};