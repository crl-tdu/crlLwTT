/**
 * @file Tensor.cpp
 * @brief Implementation of Tensor class
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/core/Tensor.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace LwTT {
namespace Core {

Tensor::Tensor() : data_(nullptr), size_(0) {
    // Create empty tensor
}

Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    // Calculate total size
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    // Allocate memory
    data_ = new float[size_];
    
    // Initialize with zeros
    std::fill(data_, data_ + size_, 0.0f);
}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_), size_(other.size_) {
    // Copy constructor
    data_ = new float[size_];
    std::memcpy(data_, other.data_, size_ * sizeof(float));
}

Tensor::Tensor(Tensor&& other) noexcept : data_(nullptr), shape_(), size_(0) {
    // Move constructor
    std::swap(data_, other.data_);
    std::swap(shape_, other.shape_);
    std::swap(size_, other.size_);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Clean up existing data
        delete[] data_;
        
        // Copy shape and size
        shape_ = other.shape_;
        size_ = other.size_;
        
        // Allocate and copy data
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Clean up existing data
        delete[] data_;
        
        // Move data
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        
        // Reset other object
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Tensor::~Tensor() {
    delete[] data_;
}

float Tensor::GetValue(const std::vector<int>& indices) const {
    int index = FlattenIndices(indices);
    if (index < 0 || index >= size_) {
        throw std::out_of_range("Tensor indices out of range");
    }
    return data_[index];
}

void Tensor::SetValue(const std::vector<int>& indices, float value) {
    int index = FlattenIndices(indices);
    if (index < 0 || index >= size_) {
        throw std::out_of_range("Tensor indices out of range");
    }
    data_[index] = value;
}

int Tensor::FlattenIndices(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }
    
    int flat_index = 0;
    int stride = 1;
    
    // Convert multi-dimensional indices to flat index (C-style row-major order)
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            return -1; // Invalid index
        }
        flat_index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return flat_index;
}

Tensor Tensor::Add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes do not match for addition");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Tensor Tensor::Multiply(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes do not match for element-wise multiplication");
    }
    
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

Tensor Tensor::MultiplyScalar(float scalar) const {
    Tensor result(shape_);
    for (int i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    
    return result;
}

void Tensor::Fill(float value) {
    std::fill(data_, data_ + size_, value);
}

void Tensor::RandomNormal(float mean, float std) {
    // Simple normal distribution using Box-Muller transform
    for (int i = 0; i < size_; i += 2) {
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        
        float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
        float z1 = std::sqrt(-2.0f * std::log(u1)) * std::sin(2.0f * M_PI * u2);
        
        data_[i] = mean + std * z0;
        if (i + 1 < size_) {
            data_[i + 1] = mean + std * z1;
        }
    }
}

void Tensor::Multiply(float scalar) {
    for (int i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
}

Tensor Tensor::Slice(int dim, int start, int end) const {
    // Simple implementation for 3D tensors (most common case)
    if (shape_.size() != 3 || dim != 1) {
        throw std::invalid_argument("Slice operation only supports dimension 1 for 3D tensors");
    }
    
    if (start < 0) start = shape_[dim] + start;
    if (end < 0) end = shape_[dim] + end;
    
    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;
    
    Tensor result(new_shape);
    
    // Copy data
    int batch_size = shape_[0];
    int seq_len = shape_[1];
    int feature_dim = shape_[2];
    
    for (int b = 0; b < batch_size; ++b) {
        for (int t = start; t < end; ++t) {
            for (int f = 0; f < feature_dim; ++f) {
                std::vector<int> src_idx = {b, t, f};
                std::vector<int> dst_idx = {b, t - start, f};
                result.Set(dst_idx, Get(src_idx));
            }
        }
    }
    
    return result;
}

void Tensor::SetSlice(int dim, int start, int end, const Tensor& values) {
    // Simple implementation for 3D tensors
    if (shape_.size() != 3 || dim != 1) {
        throw std::invalid_argument("SetSlice operation only supports dimension 1 for 3D tensors");
    }
    
    if (start < 0) start = shape_[dim] + start;
    if (end < 0) end = shape_[dim] + end;
    
    auto val_shape = values.Shape();
    int batch_size = shape_[0];
    int feature_dim = shape_[2];
    
    for (int b = 0; b < batch_size && b < val_shape[0]; ++b) {
        for (int t = start; t < end && (t - start) < val_shape[1]; ++t) {
            for (int f = 0; f < feature_dim && f < val_shape[2]; ++f) {
                std::vector<int> dst_idx = {b, t, f};
                std::vector<int> src_idx = {b, t - start, f};
                Set(dst_idx, values.Get(src_idx));
            }
        }
    }
}

std::string Tensor::ShapeString() const {
    std::string result = "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        result += std::to_string(shape_[i]);
        if (i < shape_.size() - 1) {
            result += ", ";
        }
    }
    result += "]";
    return result;
}

} // namespace Core
} // namespace LwTT
