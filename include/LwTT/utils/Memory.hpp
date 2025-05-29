/**
 * @file Memory.hpp
 * @brief Memory management utilities for LwTT
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_UTILS_MEMORY_HPP
#define LWTT_UTILS_MEMORY_HPP

#include <memory>
#include <vector>
#include <cstddef>
#include <mutex>

namespace crllwtt {
namespace Utils {

/**
 * @brief Memory pool for efficient memory allocation
 */
class MemoryPool {
public:
    /**
     * @brief Constructor
     * @param pool_size_mb Pool size in megabytes
     */
    explicit MemoryPool(size_t pool_size_mb = 256);

    /**
     * @brief Destructor
     */
    ~MemoryPool();

    /**
     * @brief Allocate memory from pool
     * @param size Size in bytes
     * @param alignment Memory alignment (default: 32 for SIMD)
     * @return Pointer to allocated memory
     */
    void* Allocate(size_t size, size_t alignment = 32);

    /**
     * @brief Deallocate memory (return to pool)
     * @param ptr Pointer to memory
     */
    void Deallocate(void* ptr);

    /**
     * @brief Get total pool size
     * @return Pool size in bytes
     */
    size_t GetPoolSize() const { return pool_size_; }

    /**
     * @brief Get used memory size
     * @return Used memory in bytes
     */
    size_t GetUsedMemory() const { return used_memory_; }

    /**
     * @brief Reset pool (deallocate all)
     */
    void Reset();

private:
    size_t pool_size_;
    size_t used_memory_;
    std::vector<char> pool_buffer_;
    std::vector<std::pair<void*, size_t>> free_blocks_;
    mutable std::mutex mutex_;
    
    // Internal methods
    void* AllocateFromPool(size_t size, size_t alignment);
    void MergeAdjacentBlocks();
};

/**
 * @brief RAII memory allocator using memory pool
 */
template<typename T>
class PoolAllocator {
public:
    using value_type = T;

    /**
     * @brief Constructor
     * @param pool Memory pool reference
     */
    explicit PoolAllocator(MemoryPool& pool) : pool_(pool) {}

    /**
     * @brief Copy constructor
     */
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}

    /**
     * @brief Allocate memory for n objects
     * @param n Number of objects
     * @return Pointer to allocated memory
     */
    T* allocate(size_t n) {
        return static_cast<T*>(pool_.Allocate(n * sizeof(T), alignof(T)));
    }

    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory
     * @param n Number of objects (unused)
     */
    void deallocate(T* ptr, size_t n) {
        (void)n; // Suppress unused parameter warning
        pool_.Deallocate(ptr);
    }

    /**
     * @brief Equality comparison
     */
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return &pool_ == &other.pool_;
    }

    /**
     * @brief Inequality comparison
     */
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

private:
    MemoryPool& pool_;
    
    template<typename U>
    friend class PoolAllocator;
};

/**
 * @brief Aligned memory allocator
 */
class AlignedAllocator {
public:
    /**
     * @brief Allocate aligned memory
     * @param size Size in bytes
     * @param alignment Memory alignment
     * @return Pointer to aligned memory
     */
    static void* Allocate(size_t size, size_t alignment = 32);

    /**
     * @brief Deallocate aligned memory
     * @param ptr Pointer to memory
     */
    static void Deallocate(void* ptr);

    /**
     * @brief Check if pointer is aligned
     * @param ptr Pointer to check
     * @param alignment Required alignment
     * @return true if aligned
     */
    static bool IsAligned(const void* ptr, size_t alignment);
};

/**
 * @brief Memory statistics and monitoring
 */
class MemoryMonitor {
public:
    /**
     * @brief Get current memory usage
     * @return Memory usage in bytes
     */
    static size_t GetCurrentUsage();

    /**
     * @brief Get peak memory usage
     * @return Peak memory usage in bytes
     */
    static size_t GetPeakUsage();

    /**
     * @brief Reset peak usage counter
     */
    static void ResetPeakUsage();

    /**
     * @brief Enable/disable memory tracking
     * @param enable Enable tracking
     */
    static void EnableTracking(bool enable);

private:
    static size_t current_usage_;
    static size_t peak_usage_;
    static bool tracking_enabled_;
    static std::mutex monitor_mutex_;
};

/**
 * @brief RAII wrapper for automatic memory management
 */
template<typename T>
class UniquePtr {
public:
    /**
     * @brief Constructor
     * @param ptr Raw pointer
     * @param pool Memory pool for deallocation
     */
    UniquePtr(T* ptr, MemoryPool& pool) : ptr_(ptr), pool_(pool) {}

    /**
     * @brief Destructor
     */
    ~UniquePtr() {
        if (ptr_) {
            ptr_->~T();
            pool_.Deallocate(ptr_);
        }
    }

    /**
     * @brief Move constructor
     */
    UniquePtr(UniquePtr&& other) noexcept : ptr_(other.ptr_), pool_(other.pool_) {
        other.ptr_ = nullptr;
    }

    /**
     * @brief Move assignment
     */
    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                ptr_->~T();
                pool_.Deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Dereference operator
     */
    T& operator*() const { return *ptr_; }

    /**
     * @brief Arrow operator
     */
    T* operator->() const { return ptr_; }

    /**
     * @brief Get raw pointer
     */
    T* get() const { return ptr_; }

    /**
     * @brief Release ownership
     */
    T* release() {
        T* result = ptr_;
        ptr_ = nullptr;
        return result;
    }

    /**
     * @brief Check if valid
     */
    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_;
    MemoryPool& pool_;

    // Disable copy
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;
};

/**
 * @brief Utility functions for memory operations
 */
namespace MemoryUtils {

    /**
     * @brief Copy memory with SIMD optimization
     * @param dst Destination pointer
     * @param src Source pointer
     * @param size Size in bytes
     */
    void FastMemcpy(void* dst, const void* src, size_t size);

    /**
     * @brief Set memory with SIMD optimization
     * @param ptr Pointer to memory
     * @param value Value to set
     * @param size Size in bytes
     */
    void FastMemset(void* ptr, int value, size_t size);

    /**
     * @brief Compare memory with SIMD optimization
     * @param ptr1 First pointer
     * @param ptr2 Second pointer
     * @param size Size in bytes
     * @return 0 if equal, non-zero otherwise
     */
    int FastMemcmp(const void* ptr1, const void* ptr2, size_t size);

    /**
     * @brief Get system memory information
     * @return Available system memory in bytes
     */
    size_t GetAvailableMemory();

    /**
     * @brief Prefetch memory for cache optimization
     * @param ptr Pointer to memory
     * @param size Size in bytes
     */
    void Prefetch(const void* ptr, size_t size);

} // namespace MemoryUtils

} // namespace Utils
} // namespace LwTT

#endif // LWTT_UTILS_MEMORY_HPP
