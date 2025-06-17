/**
 * @file PreallocatedBuffers.cpp
 * @brief Implementation of pre-allocated buffer management
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/utils/PreallocatedBuffers.hpp"
#include <algorithm>
#include <numeric>

namespace crllwtt {
namespace Utils {

// TensorPool implementation
TensorPool::TensorPool(const BufferPoolConfig& config) : config_(config) {
    // Create memory pool for tensor data
    size_t total_memory = config_.max_tensor_size * config_.tensor_pool_size;
    memory_pool_ = std::make_unique<MemoryPool>(total_memory);

    // Pre-allocate tensors
    for (size_t i = 0; i < config_.tensor_pool_size; ++i) {
        // Create empty tensor that will be resized when requested
        available_tensors_.emplace(std::vector<int>{1});
    }
}

TensorPool::~TensorPool() = default;

Core::Tensor TensorPool::GetTensor(const std::vector<int>& shape) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!available_tensors_.empty()) {
        Core::Tensor tensor = std::move(available_tensors_.front());
        available_tensors_.pop();
        
        // Create new tensor with required shape if different
        if (tensor.GetShape() != shape) {
            tensor = Core::Tensor(shape);
        }
        
        // Update peak usage
        size_t current_usage = config_.tensor_pool_size - available_tensors_.size();
        size_t expected_peak = peak_usage_.load();
        while (current_usage > expected_peak && 
               !peak_usage_.compare_exchange_weak(expected_peak, current_usage)) {
            // Retry if another thread updated peak_usage
        }
        
        return tensor;
    }
    
    // Pool is empty, create new tensor
    return Core::Tensor(shape);
}

void TensorPool::ReturnTensor(Core::Tensor&& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Only return to pool if we haven't exceeded capacity
    if (available_tensors_.size() < config_.tensor_pool_size) {
        // Clear tensor data but keep memory allocated
        tensor.Fill(0.0f);
        available_tensors_.emplace(std::move(tensor));
    }
    // If pool is full, tensor will be destroyed automatically
}

TensorPool::PoolStats TensorPool::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    PoolStats stats;
    stats.total_tensors = config_.tensor_pool_size;
    stats.available_tensors = available_tensors_.size();
    stats.peak_usage = peak_usage_.load();
    stats.total_memory_mb = (config_.max_tensor_size * config_.tensor_pool_size) / (1024 * 1024);
    
    return stats;
}

void TensorPool::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear all tensors
    while (!available_tensors_.empty()) {
        available_tensors_.pop();
    }
    
    // Reset memory pool
    memory_pool_->Reset();
    peak_usage_ = 0;
    
    // Re-initialize tensors
    for (size_t i = 0; i < config_.tensor_pool_size; ++i) {
        available_tensors_.emplace(std::vector<int>{1});
    }
}

// PreallocatedBuffers implementation
PreallocatedBuffers::PreallocatedBuffers(const BufferPoolConfig& config) : config_(config) {
    // Create tensor pools
    tensor_pool_ = std::make_unique<TensorPool>(config_);
    
    BufferPoolConfig attention_config = config_;
    attention_config.tensor_pool_size = config_.attention_buffer_size;
    attention_pool_ = std::make_unique<TensorPool>(attention_config);
    
    // Initialize pre-allocated buffers
    InitializeWorkTensors();
    InitializeAttentionBuffers();
}

PreallocatedBuffers::~PreallocatedBuffers() = default;

Core::Tensor PreallocatedBuffers::GetWorkTensor(const std::vector<int>& shape) {
    // In real-time mode, try to use pre-allocated buffers first
    if (real_time_mode_) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        // Find compatible pre-allocated tensor
        for (auto& tensor : preallocated_work_tensors_) {
            if (IsShapeCompatible(tensor.GetShape(), shape)) {
                // Create new tensor with required shape if needed
                if (tensor.GetShape() != shape) {
                    tensor = Core::Tensor(shape);
                }
                return std::move(tensor);
            }
        }
    }
    
    // Fall back to tensor pool
    return tensor_pool_->GetTensor(shape);
}

void PreallocatedBuffers::ReturnWorkTensor(Core::Tensor&& tensor) {
    if (real_time_mode_) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        // Try to return to pre-allocated buffers
        if (preallocated_work_tensors_.size() < 8) { // Limit pre-allocated buffer count
            preallocated_work_tensors_.emplace_back(std::move(tensor));
            return;
        }
    }
    
    // Return to tensor pool
    tensor_pool_->ReturnTensor(std::move(tensor));
}

Core::Tensor PreallocatedBuffers::GetAttentionBuffer(int seq_len, int num_heads) {
    std::vector<int> shape = {1, num_heads, seq_len, seq_len};
    
    // Try pre-allocated attention buffers first
    if (real_time_mode_) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        
        for (auto& buffer : preallocated_attention_buffers_) {
            if (IsShapeCompatible(buffer.GetShape(), shape)) {
                if (buffer.GetShape() != shape) {
                    buffer = Core::Tensor(shape);
                }
                return std::move(buffer);
            }
        }
    }
    
    return attention_pool_->GetTensor(shape);
}

void PreallocatedBuffers::StoreGradient(const Core::Tensor& gradient, const std::string& param_name) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Create gradient buffer if it doesn't exist
    if (gradient_buffers_.find(param_name) == gradient_buffers_.end()) {
        gradient_buffers_[param_name] = std::make_unique<CircularBuffer<Core::Tensor>>(
            config_.gradient_buffer_size);
    }
    
    // Store gradient (makes a copy)
    gradient_buffers_[param_name]->Push(gradient);
}

Core::Tensor PreallocatedBuffers::GetCachedGradient(const std::string& param_name, size_t offset) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    auto it = gradient_buffers_.find(param_name);
    if (it != gradient_buffers_.end()) {
        return it->second->GetAtOffset(offset);
    }
    
    return Core::Tensor(); // Return empty tensor if not found
}

void PreallocatedBuffers::PreallocateForModel(int max_seq_len, int d_model, int num_heads, int num_layers) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Clear existing pre-allocated buffers
    preallocated_work_tensors_.clear();
    preallocated_attention_buffers_.clear();
    
    // Pre-allocate common tensor shapes for this model
    std::vector<std::vector<int>> common_shapes = {
        {1, max_seq_len, d_model},                    // Input/output tensors
        {1, max_seq_len, d_model * 4},               // FFN intermediate
        {1, max_seq_len, d_model},                    // Attention output
        {num_heads, max_seq_len, d_model / num_heads}, // Per-head tensors
        {1, max_seq_len, max_seq_len},               // Attention scores
    };
    
    // Pre-allocate work tensors
    for (const auto& shape : common_shapes) {
        for (int i = 0; i < 2; ++i) { // 2 tensors per shape
            preallocated_work_tensors_.emplace_back(shape);
        }
    }
    
    // Pre-allocate attention buffers
    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<int> attention_shape = {1, num_heads, max_seq_len, max_seq_len};
        preallocated_attention_buffers_.emplace_back(attention_shape);
    }
}

PreallocatedBuffers::MemoryStats PreallocatedBuffers::GetMemoryStats() const {
    MemoryStats stats;
    
    // Get tensor pool stats
    auto tensor_stats = tensor_pool_->GetStats();
    auto attention_stats = attention_pool_->GetStats();
    
    stats.tensor_pool_usage_mb = tensor_stats.total_memory_mb;
    stats.attention_buffer_usage_mb = attention_stats.total_memory_mb;
    
    // Calculate gradient buffer usage
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    size_t gradient_memory = 0;
    for (const auto& pair : gradient_buffers_) {
        gradient_memory += pair.second->Size() * sizeof(Core::Tensor); // Approximate
    }
    stats.gradient_buffer_usage_mb = gradient_memory / (1024 * 1024);
    
    // Calculate pre-allocated buffer usage
    size_t preallocated_memory = 0;
    for (const auto& tensor : preallocated_work_tensors_) {
        preallocated_memory += tensor.GetSize() * sizeof(float);
    }
    for (const auto& tensor : preallocated_attention_buffers_) {
        preallocated_memory += tensor.GetSize() * sizeof(float);
    }
    
    stats.total_allocated_mb = stats.tensor_pool_usage_mb + 
                              stats.attention_buffer_usage_mb + 
                              stats.gradient_buffer_usage_mb +
                              (preallocated_memory / (1024 * 1024));
    
    // Calculate efficiency (rough estimate)
    size_t used_tensors = tensor_stats.total_tensors - tensor_stats.available_tensors;
    size_t used_attention = attention_stats.total_tensors - attention_stats.available_tensors;
    size_t total_tensors = tensor_stats.total_tensors + attention_stats.total_tensors;
    
    if (total_tensors > 0) {
        stats.memory_efficiency = (double)(used_tensors + used_attention) / total_tensors * 100.0;
    } else {
        stats.memory_efficiency = 0.0;
    }
    
    return stats;
}

void PreallocatedBuffers::Reset() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    // Reset pools
    tensor_pool_->Reset();
    attention_pool_->Reset();
    
    // Clear gradient buffers
    for (auto& pair : gradient_buffers_) {
        pair.second->Clear();
    }
    
    // Clear pre-allocated buffers
    preallocated_work_tensors_.clear();
    preallocated_attention_buffers_.clear();
    
    // Re-initialize
    InitializeWorkTensors();
    InitializeAttentionBuffers();
}

void PreallocatedBuffers::InitializeWorkTensors() {
    // Pre-allocate some common tensor sizes
    std::vector<std::vector<int>> common_shapes = {
        {1, 128, 256},    // Small model
        {1, 256, 512},    // Medium model
        {1, 512, 1024},   // Large model
    };
    
    for (const auto& shape : common_shapes) {
        preallocated_work_tensors_.emplace_back(shape);
    }
}

void PreallocatedBuffers::InitializeAttentionBuffers() {
    // Pre-allocate common attention buffer sizes
    std::vector<std::vector<int>> attention_shapes = {
        {1, 8, 128, 128},   // 8 heads, 128 seq_len
        {1, 8, 256, 256},   // 8 heads, 256 seq_len
        {1, 16, 128, 128},  // 16 heads, 128 seq_len
    };
    
    for (const auto& shape : attention_shapes) {
        preallocated_attention_buffers_.emplace_back(shape);
    }
}

bool PreallocatedBuffers::IsShapeCompatible(const std::vector<int>& existing_shape, 
                                           const std::vector<int>& required_shape) const {
    // Check if existing tensor can be resized to required shape
    if (existing_shape.size() != required_shape.size()) {
        return false;
    }
    
    // Calculate total elements
    auto existing_size = std::accumulate(existing_shape.begin(), existing_shape.end(), 1, std::multiplies<int>());
    auto required_size = std::accumulate(required_shape.begin(), required_shape.end(), 1, std::multiplies<int>());
    
    // Allow reuse if existing tensor is same size or larger (with some tolerance)
    return existing_size >= required_size && existing_size <= required_size * 2;
}

} // namespace Utils
} // namespace crllwtt