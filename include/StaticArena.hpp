#pragma once

#include <cstddef>
#include <cstdint>

template <size_t CapacityBytes>
class StaticArena {
public:
    static constexpr size_t kAlignment = 16;

    // 1. Aşama: Derleme zamanı güvenlik garantileri
    static_assert(CapacityBytes > 0, "StaticArena kapasitesi 0 byte olamaz.");
    static_assert((kAlignment & (kAlignment - 1)) == 0, "Hizalama degeri 2'nin kuvveti olmalidir.");

    StaticArena() = default;

    // 2. Aşama: 64 KB bloğun yanlışlıkla Stack'e kopyalanmasını engelle
    StaticArena(const StaticArena&) = delete;
    StaticArena& operator=(const StaticArena&) = delete;

    // 3. Aşama: 16-byte hizalı ham byte tahsisi (Bump Allocation)
    void* alloc_bytes(size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }

        // Alt 4 biti maskeleyerek bir üst 16'nın katına yuvarla
        const size_t aligned_offset =
            (offset_ + (kAlignment - 1)) & ~(kAlignment - 1);

        // Tam sayı taşmasına (overflow) karşı güvenli sınır kontrolü
        if (aligned_offset > CapacityBytes ||
            bytes > (CapacityBytes - aligned_offset)) {
            return nullptr;
        }

        void* ptr = &buffer_[aligned_offset];
        offset_ = aligned_offset + bytes;

        // Zirve kullanımı kaydet (telemetri için)
        if (offset_ > peak_bytes_) {
            peak_bytes_ = offset_;
        }

        return ptr;
    }

    // 4. Aşama: Tip güvenli tensör tahsis sarmalayıcısı
    template <typename T>
    T* alloc(size_t count) {
        static_assert(alignof(T) <= kAlignment,
                      "Tipin hizalama gereksinimi 16 byte sinirini asiyor.");

        // Çarpma taşmasını (count * sizeof(T)) önceden yakala
        if (count == 0 || count > (CapacityBytes / sizeof(T))) {
            return nullptr;
        }

        return static_cast<T*>(alloc_bytes(count * sizeof(T)));
    }

    // 5. Aşama: O(1) sürede tüm arenayı serbest bırak (imleci başa çek)
    void reset() {
        offset_ = 0;
    }

    // Telemetri ve durum sorguları
    size_t used_bytes() const { return offset_; }
    size_t peak_bytes() const { return peak_bytes_; }
    size_t remaining_bytes() const { return CapacityBytes - offset_; }
    static constexpr size_t capacity() { return CapacityBytes; }

private:
    // Başlangıç adresini 16'nın tam katına sabitle (DRAM .bss bölgesinde yaşar)
    alignas(kAlignment) uint8_t buffer_[CapacityBytes]{};
    size_t offset_ = 0;
    size_t peak_bytes_ = 0;
};