#ifndef LOGGING_ALLOCATOR_HPP
#define LOGGING_ALLOCATOR_HPP

#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

template <typename T>
struct LoggingAllocator {
    using value_type = T;

    LoggingAllocator() noexcept = default;

    template <typename U>
    LoggingAllocator(const LoggingAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::cout << "[allocate] "
                  << n << " object(s), "
                  << n * sizeof(T) << " bytes, "
                  << "type = " << typeid(T).name()
                  << '\n';

        return static_cast<T*>(::operator new(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::cout << "[deallocate] "
                  << n << " object(s), "
                  << n * sizeof(T) << " bytes, "
                  << "type = " << typeid(T).name()
                  << '\n';

        ::operator delete(p);
    }

    template <typename U>
    bool operator==(const LoggingAllocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const LoggingAllocator<U>&) const noexcept {
        return false;
    }
};

#endif  // LOGGING_ALLOCATOR_HPP
