/**
 * @file TransformerBlock.cpp
 * @brief Implementation of Transformer Block
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/layers/TransformerBlock.hpp"
#include <stdexcept>
#include <cmath>

namespace crllwtt {
namespace Layers {

// Forward declarations for internal components
class TransformerBlock::MultiHeadAttention {
public:
    explicit MultiHeadAttention(int d_model, int n_heads, float dropout_rate)
        : d_model_(d_model), n_heads_(n_heads), dropout_rate_(dropout_rate),
          d_k_(d_model / n_heads) {
        if (d_model % n_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by n_heads");
        }
    }
    
    Core::Tensor Forward(const Core::Tensor& input, const Core::Tensor* mask = nullptr) {
        // Simplified multi-head attention implementation
        auto input_shape = input.Shape();
        if (input_shape.size() != 3) {
            throw std::invalid_argument("Input must be 3D tensor [batch, seq_len, d_model]");
        }
        
        int batch_size = input_shape[0];
        int seq_len = input_shape[1];
        int d_model = input_shape[2];
        
        // Create output tensor with same shape as input
        Core::Tensor output({batch_size, seq_len, d_model});
        
        // For now, implement a simple identity operation
        // In a full implementation, this would include:
        // 1. Linear projections for Q, K, V
        // 2. Scaled dot-product attention
        // 3. Multi-head concatenation and final projection
        
        const float* input_data = input.GetData();
        float* output_data = output.GetData();
        
        for (int i = 0; i < input.GetSize(); ++i) {
            output_data[i] = input_data[i];
        }
        
        // Store attention weights (simplified - just identity for now)
        attention_weights_ = Core::Tensor({batch_size, n_heads_, seq_len, seq_len});
        attention_weights_.Fill(0.0f);
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < n_heads_; ++h) {
                for (int i = 0; i < seq_len; ++i) {
                    // Set diagonal to 1 (attending to self)
                    attention_weights_.SetValue({b, h, i, i}, 1.0f);
                }
            }
        }
        
        return output;
    }
    
    Core::Tensor GetAttentionWeights() const {
        return attention_weights_;
    }
    
private:
    int d_model_;
    int n_heads_;
    float dropout_rate_;
    int d_k_;
    mutable Core::Tensor attention_weights_;
};

class TransformerBlock::FeedForward {
public:
    explicit FeedForward(int d_model, int d_ff, float dropout_rate)
        : d_model_(d_model), d_ff_(d_ff), dropout_rate_(dropout_rate) {}
    
    Core::Tensor Forward(const Core::Tensor& input) {
        // Simplified feed-forward implementation
        // In a full implementation, this would be:
        // output = ReLU(input * W1 + b1) * W2 + b2
        
        auto input_shape = input.Shape();
        Core::Tensor output(input_shape);
        
        const float* input_data = input.GetData();
        float* output_data = output.GetData();
        
        // Apply a simple non-linear transformation
        for (int i = 0; i < input.GetSize(); ++i) {
            float x = input_data[i];
            // Simplified GELU-like activation: x * sigmoid(1.702 * x)
            float sigmoid_val = 1.0f / (1.0f + std::exp(-1.702f * x));
            output_data[i] = x * sigmoid_val;
        }
        
        return output;
    }
    
private:
    int d_model_;
    int d_ff_;
    float dropout_rate_;
};

class TransformerBlock::LayerNorm {
public:
    explicit LayerNorm(int d_model, float eps = 1e-6f)
        : d_model_(d_model), eps_(eps) {}
    
    Core::Tensor Forward(const Core::Tensor& input) {
        auto input_shape = input.Shape();
        Core::Tensor output(input_shape);
        
        if (input_shape.size() != 3) {
            // If not 3D, just copy input
            const float* input_data = input.GetData();
            float* output_data = output.GetData();
            for (int i = 0; i < input.GetSize(); ++i) {
                output_data[i] = input_data[i];
            }
            return output;
        }
        
        int batch_size = input_shape[0];
        int seq_len = input_shape[1];
        int d_model = input_shape[2];
        
        const float* input_data = input.GetData();
        float* output_data = output.GetData();
        
        // Apply layer normalization along the last dimension
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                int offset = (b * seq_len + s) * d_model;
                
                // Compute mean
                float mean = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    mean += input_data[offset + d];
                }
                mean /= d_model;
                
                // Compute variance
                float variance = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    float diff = input_data[offset + d] - mean;
                    variance += diff * diff;
                }
                variance /= d_model;
                
                // Normalize
                float inv_std = 1.0f / std::sqrt(variance + eps_);
                for (int d = 0; d < d_model; ++d) {
                    output_data[offset + d] = (input_data[offset + d] - mean) * inv_std;
                }
            }
        }
        
        return output;
    }
    
private:
    int d_model_;
    float eps_;
};

// TransformerBlock implementation
TransformerBlock::TransformerBlock(const TransformerBlockConfig& config)
    : config_(config), training_(false) {
    
    // Initialize components
    attention_ = std::make_unique<MultiHeadAttention>(
        config_.d_model, config_.n_heads, config_.dropout_rate);
    
    feed_forward_ = std::make_unique<FeedForward>(
        config_.d_model, config_.d_ff, config_.dropout_rate);
    
    norm1_ = std::make_unique<LayerNorm>(config_.d_model, config_.layer_norm_eps);
    norm2_ = std::make_unique<LayerNorm>(config_.d_model, config_.layer_norm_eps);
}

TransformerBlock::~TransformerBlock() = default;

Core::Tensor TransformerBlock::Forward(const Core::Tensor& input,
                                      const Core::Tensor* mask,
                                      const Core::TimeInfo* time_info) {
    
    if (config_.use_pre_norm) {
        // Pre-normalization: Norm -> Attention -> Residual -> Norm -> FFN -> Residual
        
        // Self-attention with pre-normalization
        Core::Tensor norm1_out = norm1_->Forward(input);
        Core::Tensor attn_out = attention_->Forward(norm1_out, mask);
        Core::Tensor residual1 = input.Add(attn_out);
        
        // Feed-forward with pre-normalization
        Core::Tensor norm2_out = norm2_->Forward(residual1);
        Core::Tensor ff_out = feed_forward_->Forward(norm2_out);
        Core::Tensor residual2 = residual1.Add(ff_out);
        
        return residual2;
    } else {
        // Post-normalization: Attention -> Residual -> Norm -> FFN -> Residual -> Norm
        
        // Self-attention with post-normalization
        Core::Tensor attn_out = attention_->Forward(input, mask);
        Core::Tensor residual1 = input.Add(attn_out);
        Core::Tensor norm1_out = norm1_->Forward(residual1);
        
        // Feed-forward with post-normalization
        Core::Tensor ff_out = feed_forward_->Forward(norm1_out);
        Core::Tensor residual2 = norm1_out.Add(ff_out);
        Core::Tensor norm2_out = norm2_->Forward(residual2);
        
        return norm2_out;
    }
}

Core::Tensor TransformerBlock::GetAttentionWeights() const {
    if (attention_) {
        return attention_->GetAttentionWeights();
    }
    return Core::Tensor({1}); // Return dummy tensor if no attention weights
}

} // namespace Layers
} // namespace LwTT
