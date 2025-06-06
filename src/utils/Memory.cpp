/**
 * @file Memory.cpp
 * @brief Memory management utilities implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "../../include/LwTT/utils/Memory.hpp"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace crllwtt {
namespace Utils {

// MemoryPool implementation
MemoryPool::MemoryPool(size_t pool_size_mb) 
    : pool_size_(pool_size_mb * 1024 * 1024), used_memory_(0) {
    
    // Allocate the memory pool
    pool_buffer_.resize(pool_size_);
    
    // Initialize with one large free block
    free_blocks_.emplace_back(pool_buffer_.data(), pool_size_);
}

MemoryPool::~MemoryPool() {
    // Destructor - memory is automatically freed when vector is destroyed
}

void* MemoryPool::Allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Align size to the specified boundary
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);
    
    // Find a suitable free block
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
        if (it->second >= aligned_size) {
            void* ptr = it->first;
            
            // If the block is larger than needed, split it
            if (it->second > aligned_size) {
                void* new_free_ptr = static_cast<char*>(ptr) + aligned_size;
                size_t new_free_size = it->second - aligned_size;
                
                // Update the current block or create a new one
                it->first = new_free_ptr;
                it->second = new_free_size;
            } else {
                // Remove the block entirely
                free_blocks_.erase(it);
            }
            
            used_memory_ += aligned_size;
            return ptr;
        }
    }
    
    // No suitable block found
    throw std::bad_alloc();
}

void MemoryPool::Deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!ptr) return;
    
    // For simplicity, we don't track individual allocation sizes
    // In a real implementation, we would need to store allocation metadata
    // For now, just add it back as a free block with arbitrary size
    // This is a simplified implementation
    
    // Note: This is not a complete implementation
    // A production memory pool would need to track allocation sizes
}

void MemoryPool::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Reset all memory as one large free block
    free_blocks_.clear();
    free_blocks_.emplace_back(pool_buffer_.data(), pool_size_);
    used_memory_ = 0;
}

// AlignedAllocator implementation
void* AlignedAllocator::Allocate(size_t size, size_t alignment) {
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return ptr;
}

void AlignedAllocator::Deallocate(void* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

bool AlignedAllocator::IsAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

// MemoryMonitor implementation
size_t MemoryMonitor::current_usage_ = 0;
size_t MemoryMonitor::peak_usage_ = 0;
bool MemoryMonitor::tracking_enabled_ = false;
std::mutex MemoryMonitor::monitor_mutex_;

size_t MemoryMonitor::GetCurrentUsage() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return current_usage_;
}

size_t MemoryMonitor::GetPeakUsage() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    return peak_usage_;
}

void MemoryMonitor::ResetPeakUsage() {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    peak_usage_ = current_usage_;
}

void MemoryMonitor::EnableTracking(bool enable) {
    std::lock_guard<std::mutex> lock(monitor_mutex_);
    tracking_enabled_ = enable;
}

// MemoryUtils implementation
namespace MemoryUtils {

void FastMemcpy(void* dst, const void* src, size_t size) {
    // For now, use standard memcpy
    // In a real implementation, we could use SIMD optimizations
    std::memcpy(dst, src, size);
}

void FastMemset(void* ptr, int value, size_t size) {
    // For now, use standard memset
    // In a real implementation, we could use SIMD optimizations
    std::memset(ptr, value, size);
}

int FastMemcmp(const void* ptr1, const void* ptr2, size_t size) {
    // For now, use standard memcmp
    // In a real implementation, we could use SIMD optimizations
    return std::memcmp(ptr1, ptr2, size);
}

size_t GetAvailableMemory() {
    // This is platform-specific and simplified
    // In a real implementation, we would query system memory
    return 1024 * 1024 * 1024; // Return 1GB as placeholder
}

void Prefetch(const void* ptr, size_t size) {
    // Platform-specific prefetch instructions
    // For now, this is a no-op
    (void)ptr;
    (void)size;
}

} // namespace MemoryUtils

} // namespace Utils
} // namespace LwTT
