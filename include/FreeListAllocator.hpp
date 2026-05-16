#ifndef MLPINFER_FREELISTALLOCATOR_H
#define MLPINFER_FREELISTALLOCATOR_H

#pragma once

#include "FreeListMemoryPool.hpp"

template <typename T>
class FreeListAllocator {
public:
    using value_type = T;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    using is_always_equal = std::false_type;

    FreeListMemoryPool* pool = nullptr;

    FreeListAllocator() noexcept = default;

    explicit FreeListAllocator(FreeListMemoryPool& memory_pool) noexcept
        : pool(&memory_pool) {}

    template <typename U>
    FreeListAllocator(const FreeListAllocator<U>& other) noexcept
        : pool(other.pool) {}

    T* allocate(std::size_t n) {
        if (pool == nullptr) {
            throw std::bad_alloc{};
        }

        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_alloc{};
        }

        void* memory = pool->allocate(n * sizeof(T), alignof(T));
        return static_cast<T*>(memory);
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        pool->deallocate(ptr);
    }

#ifdef ALLOC_DISABLE_INIT
    // Called by vector when it wants to default-construct an element.
    // For float/int/etc., this starts object lifetime but does not zero memory.
    template <typename U>
    void construct(U* ptr) {
        static_assert(
            std::is_trivially_default_constructible_v<U>,
            "No-init default construction is only safe for trivial types"
        );

        ::new (static_cast<void*>(ptr)) U;
    }

    // Called for push_back, emplace_back, moves, copies, etc.
    template <typename U, typename... Args>
        requires (sizeof...(Args) > 0)
    void construct(U* ptr, Args&&... args) {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }
#endif

    template <typename U>
    bool operator==(const FreeListAllocator<U>& other) const noexcept {
        return pool == other.pool;
    }

    template <typename U>
    bool operator!=(const FreeListAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

    template <typename U>
    friend class FreeListAllocator;
};

#endif //MLPINFER_FREELISTALLOCATOR_H