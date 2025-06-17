/**
 * @file PreallocatedBuffers.hpp
 * @brief Pre-allocated buffer management for real-time applications
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_UTILS_PREALLOCATED_BUFFERS_HPP
#define LWTT_UTILS_PREALLOCATED_BUFFERS_HPP

#include "Memory.hpp"
#include "../core/Tensor.hpp"
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>

namespace crllwtt {
namespace Utils {

/**
 * @brief Configuration for pre-allocated buffers
 */
struct BufferPoolConfig {
    size_t max_tensor_size = 1024 * 1024;  // 1MB per tensor
    size_t tensor_pool_size = 16;          // Number of tensors in pool
    size_t gradient_buffer_size = 512;     // Number of gradient buffers
    size_t attention_buffer_size = 256;    // Number of attention buffers
    bool enable_simd_alignment = true;     // Align for SIMD operations
    size_t alignment = 32;                 // Memory alignment in bytes
};

/**
 * @brief Pre-allocated tensor pool for zero-allocation inference
 */
class TensorPool {
public:
    explicit TensorPool(const BufferPoolConfig& config);
    ~TensorPool();

    /**
     * @brief Get a tensor from the pool
     * @param shape Required tensor shape
     * @return Tensor from pool or new tensor if pool empty
     */
    Core::Tensor GetTensor(const std::vector<int>& shape);

    /**
     * @brief Return tensor to pool
     * @param tensor Tensor to return
     */
    void ReturnTensor(Core::Tensor&& tensor);

    /**
     * @brief Get pool statistics
     */
    struct PoolStats {
        size_t total_tensors;
        size_t available_tensors;
        size_t peak_usage;
        size_t total_memory_mb;
    };
    
    PoolStats GetStats() const;

    /**
     * @brief Reset pool and clear all tensors
     */
    void Reset();

private:
    BufferPoolConfig config_;
    std::queue<Core::Tensor> available_tensors_;
    std::atomic<size_t> peak_usage_{0};
    mutable std::mutex mutex_;
    std::unique_ptr<MemoryPool> memory_pool_;
};

/**
 * @brief Circular buffer for gradient storage with real-time constraints
 */
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity) 
        : buffer_(capacity), capacity_(capacity), head_(0), tail_(0), full_(false) {}

    /**
     * @brief Add element to buffer (overwrites oldest if full)
     * @param item Item to add
     */
    void Push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_[head_] = item;
        
        if (full_) {
            tail_ = (tail_ + 1) % capacity_;
        }
        
        head_ = (head_ + 1) % capacity_;
        full_ = head_ == tail_;
    }

    /**
     * @brief Get most recent element
     * @return Most recent element or default if empty
     */
    T GetLatest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (Empty()) return T{};
        
        size_t latest_idx = (head_ + capacity_ - 1) % capacity_;
        return buffer_[latest_idx];
    }

    /**
     * @brief Get element at offset from most recent
     * @param offset Offset from latest (0 = latest, 1 = previous, etc.)
     * @return Element or default if offset too large
     */
    T GetAtOffset(size_t offset) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (Empty() || offset >= Size()) return T{};
        
        size_t idx = (head_ + capacity_ - 1 - offset) % capacity_;
        return buffer_[idx];
    }

    /**
     * @brief Check if buffer is empty
     */
    bool Empty() const {
        return !full_ && (head_ == tail_);
    }

    /**
     * @brief Get current size
     */
    size_t Size() const {
        if (full_) return capacity_;
        return (head_ >= tail_) ? (head_ - tail_) : (capacity_ + head_ - tail_);
    }

    /**
     * @brief Clear buffer
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = tail_ = 0;
        full_ = false;
    }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_;
    size_t tail_;
    bool full_;
    mutable std::mutex mutex_;
};

/**
 * @brief Pre-allocated buffer manager for real-time applications
 */
class PreallocatedBuffers {
public:
    explicit PreallocatedBuffers(const BufferPoolConfig& config = BufferPoolConfig{});
    ~PreallocatedBuffers();

    /**
     * @brief Get tensor for temporary computation
     * @param shape Required tensor shape
     * @return Tensor from pool
     */
    Core::Tensor GetWorkTensor(const std::vector<int>& shape);

    /**
     * @brief Return work tensor to pool
     * @param tensor Tensor to return
     */
    void ReturnWorkTensor(Core::Tensor&& tensor);

    /**
     * @brief Get attention buffer for attention computation
     * @param seq_len Sequence length
     * @param num_heads Number of attention heads
     * @return Attention buffer tensor
     */
    Core::Tensor GetAttentionBuffer(int seq_len, int num_heads);

    /**
     * @brief Store gradient in circular buffer
     * @param gradient Gradient tensor to store
     * @param param_name Parameter name for identification
     */
    void StoreGradient(const Core::Tensor& gradient, const std::string& param_name);

    /**
     * @brief Get cached gradient
     * @param param_name Parameter name
     * @param offset Offset from latest (0 = latest)
     * @return Cached gradient or empty tensor if not found
     */
    Core::Tensor GetCachedGradient(const std::string& param_name, size_t offset = 0) const;

    /**
     * @brief Pre-allocate buffers for specific model configuration
     * @param max_seq_len Maximum sequence length
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param num_layers Number of layers
     */
    void PreallocateForModel(int max_seq_len, int d_model, int num_heads, int num_layers);

    /**
     * @brief Get memory usage statistics
     */
    struct MemoryStats {
        size_t total_allocated_mb;
        size_t tensor_pool_usage_mb;
        size_t gradient_buffer_usage_mb;
        size_t attention_buffer_usage_mb;
        double memory_efficiency;  // percentage of memory actually used
    };
    
    MemoryStats GetMemoryStats() const;

    /**
     * @brief Reset all buffers and pools
     */
    void Reset();

    /**
     * @brief Enable/disable real-time mode (stricter memory constraints)
     * @param enable Enable real-time mode
     */
    void SetRealTimeMode(bool enable) { real_time_mode_ = enable; }

private:
    BufferPoolConfig config_;
    std::unique_ptr<TensorPool> tensor_pool_;
    std::unique_ptr<TensorPool> attention_pool_;
    std::unordered_map<std::string, std::unique_ptr<CircularBuffer<Core::Tensor>>> gradient_buffers_;
    std::atomic<bool> real_time_mode_{false};
    mutable std::mutex buffer_mutex_;

    // Pre-allocated buffers for common operations
    std::vector<Core::Tensor> preallocated_work_tensors_;
    std::vector<Core::Tensor> preallocated_attention_buffers_;

    // Internal methods
    void InitializeWorkTensors();
    void InitializeAttentionBuffers();
    bool IsShapeCompatible(const std::vector<int>& shape1, const std::vector<int>& shape2) const;
};

/**
 * @brief RAII helper for automatic tensor return to pool
 */
class ScopedTensor {
public:
    ScopedTensor(Core::Tensor&& tensor, PreallocatedBuffers* buffers)
        : tensor_(std::move(tensor)), buffers_(buffers) {}

    ~ScopedTensor() {
        if (buffers_) {
            buffers_->ReturnWorkTensor(std::move(tensor_));
        }
    }

    // Non-copyable but movable
    ScopedTensor(const ScopedTensor&) = delete;
    ScopedTensor& operator=(const ScopedTensor&) = delete;
    ScopedTensor(ScopedTensor&&) = default;
    ScopedTensor& operator=(ScopedTensor&&) = default;

    Core::Tensor& operator*() { return tensor_; }
    const Core::Tensor& operator*() const { return tensor_; }
    Core::Tensor* operator->() { return &tensor_; }
    const Core::Tensor* operator->() const { return &tensor_; }

private:
    Core::Tensor tensor_;
    PreallocatedBuffers* buffers_;
};

} // namespace Utils
} // namespace crllwtt

#endif // LWTT_UTILS_PREALLOCATED_BUFFERS_HPP