#ifndef MLPINFER_FREELISTMEMORYPOOL_H
#define MLPINFER_FREELISTMEMORYPOOL_H

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

class FreeListMemoryPool {
private:
    struct FreeBlock {
        std::size_t size;
        FreeBlock* next;
    };

    struct AllocationHeader {
        std::size_t block_size;
        std::size_t padding;
    };

    std::byte* memory_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t alignment_ = alignof(std::max_align_t);
    FreeBlock* free_list_ = nullptr;

private:
    static bool is_power_of_two(std::size_t x) {
        return x != 0 && (x & (x - 1)) == 0;
    }

    static std::uintptr_t align_up(std::uintptr_t value, std::size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static std::size_t align_up_size(std::size_t value, std::size_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

public:
    explicit FreeListMemoryPool(
        std::size_t capacity,
        std::size_t alignment = alignof(std::max_align_t)
    )
        : alignment_(alignment) {
#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] constructor: requested capacity = "
                  << capacity
                  << ", requested alignment = "
                  << alignment_
                  << '\n';
#endif

        if (!is_power_of_two(alignment_)) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] constructor: invalid alignment, "
                      << "not a power of two: "
                      << alignment_
                      << '\n';
#endif
            throw std::invalid_argument("Pool alignment must be a power of two");
        }

        capacity_ = align_up_size(capacity, alignment_);

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] constructor: aligned capacity = "
                  << capacity_
                  << '\n';
#endif

        memory_ = static_cast<std::byte*>(
            ::operator new(capacity_, std::align_val_t{alignment_})
        );

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] constructor: memory = "
                  << static_cast<void*>(memory_)
                  << '\n';
#endif

        free_list_ = reinterpret_cast<FreeBlock*>(memory_);
        free_list_->size = capacity_;
        free_list_->next = nullptr;

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] constructor: initial free block, "
                  << "offset = 0, size = "
                  << free_list_->size
                  << '\n';
#endif
    }

    FreeListMemoryPool(const FreeListMemoryPool&) = delete;
    FreeListMemoryPool& operator=(const FreeListMemoryPool&) = delete;

    ~FreeListMemoryPool() {
#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] destructor: memory = "
                  << static_cast<void*>(memory_)
                  << ", capacity = "
                  << capacity_
                  << '\n';
#endif

        ::operator delete(memory_, std::align_val_t{alignment_});
    }

    void* allocate(std::size_t bytes, std::size_t requested_alignment) {
#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] allocate: bytes = "
                  << bytes
                  << ", requested alignment = "
                  << requested_alignment
                  << '\n';
#endif

        if (bytes == 0) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] allocate: zero-byte allocation, "
                      << "returning nullptr"
                      << '\n';
#endif
            return nullptr;
        }

        if (requested_alignment > alignment_) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] allocate: requested alignment "
                      << requested_alignment
                      << " exceeds pool alignment "
                      << alignment_
                      << '\n';
#endif
            throw std::bad_alloc{};
        }

        if (!is_power_of_two(requested_alignment)) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] allocate: requested alignment "
                      << requested_alignment
                      << " is not a power of two"
                      << '\n';
#endif
            throw std::bad_alloc{};
        }

        FreeBlock* previous = nullptr;
        FreeBlock* current = free_list_;

        while (current != nullptr) {
            std::byte* block_begin = reinterpret_cast<std::byte*>(current);
            std::uintptr_t raw_address =
                reinterpret_cast<std::uintptr_t>(block_begin);

            std::uintptr_t user_address = align_up(
                raw_address + sizeof(AllocationHeader),
                requested_alignment
            );

            std::uintptr_t header_address =
                user_address - sizeof(AllocationHeader);

            std::size_t padding =
                static_cast<std::size_t>(header_address - raw_address);

            std::size_t required_size =
                padding + sizeof(AllocationHeader) + bytes;

            required_size = std::max(required_size, sizeof(FreeBlock));
            required_size = align_up_size(required_size, alignment_);

#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] allocate: inspecting block, "
                      << "offset = "
                      << (block_begin - memory_)
                      << ", block size = "
                      << current->size
                      << ", padding = "
                      << padding
                      << ", required size = "
                      << required_size
                      << '\n';
#endif

            if (required_size <= current->size) {
                std::size_t remaining_size = current->size - required_size;

#if defined(FREE_LIST_DEBUG)
                std::cerr << "[FreeListMemoryPool] allocate: selected block, "
                          << "offset = "
                          << (block_begin - memory_)
                          << ", block size = "
                          << current->size
                          << ", remaining size = "
                          << remaining_size
                          << '\n';
#endif

                if (remaining_size >= sizeof(FreeBlock)) {
                    auto* next_block = reinterpret_cast<FreeBlock*>(
                        block_begin + required_size
                    );

                    next_block->size = remaining_size;
                    next_block->next = current->next;

                    if (previous != nullptr) {
                        previous->next = next_block;
                    } else {
                        free_list_ = next_block;
                    }

#if defined(FREE_LIST_DEBUG)
                    std::cerr << "[FreeListMemoryPool] allocate: split block, "
                              << "allocated size = "
                              << required_size
                              << ", new free block offset = "
                              << (reinterpret_cast<std::byte*>(next_block) - memory_)
                              << ", new free block size = "
                              << next_block->size
                              << '\n';
#endif
                } else {
                    required_size = current->size;

                    if (previous != nullptr) {
                        previous->next = current->next;
                    } else {
                        free_list_ = current->next;
                    }

#if defined(FREE_LIST_DEBUG)
                    std::cerr << "[FreeListMemoryPool] allocate: consumed whole block, "
                              << "allocated size = "
                              << required_size
                              << '\n';
#endif
                }

                auto* header =
                    reinterpret_cast<AllocationHeader*>(header_address);

                header->block_size = required_size;
                header->padding = padding;

#if defined(FREE_LIST_DEBUG)
                std::cerr << "[FreeListMemoryPool] allocate: returning ptr = "
                          << reinterpret_cast<void*>(user_address)
                          << ", user offset = "
                          << (reinterpret_cast<std::byte*>(user_address) - memory_)
                          << ", header offset = "
                          << (reinterpret_cast<std::byte*>(header_address) - memory_)
                          << ", block size = "
                          << header->block_size
                          << ", padding = "
                          << header->padding
                          << '\n';
#endif

                return reinterpret_cast<void*>(user_address);
            }

            previous = current;
            current = current->next;
        }

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] allocate: out of memory for bytes = "
                  << bytes
                  << ", requested alignment = "
                  << requested_alignment
                  << '\n';
#endif

        throw std::bad_alloc{};
    }

    void deallocate(void* ptr) noexcept {
#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] deallocate: ptr = "
                  << ptr
                  << '\n';
#endif

        if (ptr == nullptr) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] deallocate: nullptr, ignoring"
                      << '\n';
#endif
            return;
        }

        auto* user_address = static_cast<std::byte*>(ptr);

        auto* header = reinterpret_cast<AllocationHeader*>(
            user_address - sizeof(AllocationHeader)
        );

        std::byte* block_begin =
            user_address - sizeof(AllocationHeader) - header->padding;

        std::size_t block_size = header->block_size;

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] deallocate: user offset = "
                  << (user_address - memory_)
                  << ", block offset = "
                  << (block_begin - memory_)
                  << ", block size = "
                  << block_size
                  << ", padding = "
                  << header->padding
                  << '\n';
#endif

        auto* new_block = reinterpret_cast<FreeBlock*>(block_begin);
        new_block->size = block_size;
        new_block->next = nullptr;

        insert_and_coalesce(new_block);
    }

    void dump() const {
        std::cout << "Free list:\n";

        const FreeBlock* current = free_list_;

        while (current != nullptr) {
            const auto* block_address =
                reinterpret_cast<const std::byte*>(current);

            std::ptrdiff_t offset = block_address - memory_;

            std::cout << "  offset = " << offset
                      << ", size = " << current->size
                      << '\n';

            current = current->next;
        }
    }

private:
    void insert_and_coalesce(FreeBlock* block) noexcept {
#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] insert_and_coalesce: block offset = "
                  << (reinterpret_cast<std::byte*>(block) - memory_)
                  << ", block size = "
                  << block->size
                  << '\n';
#endif

        if (free_list_ == nullptr || block < free_list_) {
            block->next = free_list_;
            free_list_ = block;

#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] insert_and_coalesce: inserted at head"
                      << '\n';
#endif

            coalesce_with_next(block);
            return;
        }

        FreeBlock* previous = free_list_;
        FreeBlock* current = free_list_->next;

        while (current != nullptr && current < block) {
            previous = current;
            current = current->next;
        }

        block->next = current;
        previous->next = block;

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] insert_and_coalesce: inserted after offset = "
                  << (reinterpret_cast<std::byte*>(previous) - memory_);

        if (current != nullptr) {
            std::cerr << ", before offset = "
                      << (reinterpret_cast<std::byte*>(current) - memory_);
        } else {
            std::cerr << ", before nullptr";
        }

        std::cerr << '\n';
#endif

        coalesce_with_next(block);
        coalesce_with_next(previous);
    }

    static void coalesce_with_next(FreeBlock* block) noexcept {
        if (block == nullptr || block->next == nullptr) {
#if defined(FREE_LIST_DEBUG)
            if (block == nullptr) {
                std::cerr << "[FreeListMemoryPool] coalesce_with_next: block is nullptr"
                          << '\n';
            } else {
                std::cerr << "[FreeListMemoryPool] coalesce_with_next: no next block"
                          << '\n';
            }
#endif
            return;
        }

        auto* block_end =
            reinterpret_cast<std::byte*>(block) + block->size;

        auto* next_begin =
            reinterpret_cast<std::byte*>(block->next);

#if defined(FREE_LIST_DEBUG)
        std::cerr << "[FreeListMemoryPool] coalesce_with_next: block = "
                  << static_cast<void*>(reinterpret_cast<std::byte*>(block))
                  << ", block size = "
                  << block->size
                  << ", next = "
                  << static_cast<void*>(next_begin)
                  << ", next size = "
                  << block->next->size
                  << '\n';
#endif

        if (block_end == next_begin) {
#if defined(FREE_LIST_DEBUG)
            std::cerr << "[FreeListMemoryPool] coalesce_with_next: coalescing blocks, "
                      << "new size = "
                      << (block->size + block->next->size)
                      << '\n';
#endif

            block->size += block->next->size;
            block->next = block->next->next;
        }
#if defined(FREE_LIST_DEBUG)
        else {
            std::cerr << "[FreeListMemoryPool] coalesce_with_next: blocks are not adjacent"
                      << '\n';
        }
#endif
    }
};

#endif //MLPINFER_FREELISTMEMORYPOOL_H