/**
 * @file KernelFusion.cpp
 * @brief Implementation of kernel fusion optimizations for efficient computation
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/optimization/KernelFusion.hpp"
#include "LwTT/utils/SIMD.hpp"
#include <algorithm>
#include <cmath>

#ifdef LWTT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace crllwtt {
namespace Optimization {

// KernelFusion implementation
KernelFusion::KernelFusion(const KernelFusionConfig& config) : config_(config) {
    // Initialize internal state
}

Core::Tensor KernelFusion::FusedLinearReLU(const Core::Tensor& input, 
                                          const Core::Tensor& weight, 
                                          const Core::Tensor& bias) {
    const auto& input_shape = input.GetShape();
    const auto& weight_shape = weight.GetShape();
    
    // Validate shapes: input [batch, in_features], weight [out_features, in_features]
    if (input_shape.size() != 2 || weight_shape.size() != 2) {
        throw std::invalid_argument("Input and weight must be 2D tensors");
    }
    
    int batch_size = input_shape[0];
    int in_features = input_shape[1];
    int out_features = weight_shape[0];
    
    if (weight_shape[1] != in_features) {
        throw std::invalid_argument("Input and weight dimensions don't match");
    }
    
    Core::Tensor output({batch_size, out_features});
    
    const float* input_data = input.GetData();
    const float* weight_data = weight.GetData();
    const float* bias_data = bias.GetSize() > 0 ? bias.GetData() : nullptr;
    float* output_data = output.GetData();
    
    // Fused linear + ReLU computation
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            float sum = 0.0f;
            
            // Linear transformation using SIMD
            sum = Utils::SIMDUtils::DotProduct(
                &input_data[b * in_features],
                &weight_data[out_idx * in_features],
                in_features
            );
            
            // Add bias
            if (bias_data) {
                sum += bias_data[out_idx];
            }
            
            // Apply ReLU activation
            output_data[b * out_features + out_idx] = std::max(0.0f, sum);
        }
    }
    
    return output;
}

Core::Tensor KernelFusion::FusedAttentionSoftmax(const Core::Tensor& query,
                                                 const Core::Tensor& key,
                                                 const Core::Tensor& value,
                                                 const Core::Tensor* mask,
                                                 float scale) {
    const auto& q_shape = query.GetShape();
    const auto& k_shape = key.GetShape();
    const auto& v_shape = value.GetShape();
    
    // Validate input dimensions [batch, seq_len, d_model]
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::invalid_argument("Query, Key, Value must be 3D tensors");
    }
    
    int batch_size = q_shape[0];
    int seq_len = q_shape[1];
    int d_model = q_shape[2];
    
    Core::Tensor output({batch_size, seq_len, d_model});
    
    const float* q_data = query.GetData();
    const float* k_data = key.GetData();
    const float* v_data = value.GetData();
    const float* mask_data = mask ? mask->GetData() : nullptr;
    float* output_data = output.GetData();
    
    // Pre-allocate attention weights
    std::vector<float> attention_scores(seq_len * seq_len);
    std::vector<float> attention_weights(seq_len * seq_len);
    
    #pragma omp parallel for if(batch_size > 1) firstprivate(attention_scores, attention_weights)
    for (int b = 0; b < batch_size; ++b) {
        // Compute attention scores: Q * K^T
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float score = Utils::SIMDUtils::DotProduct(
                    &q_data[b * seq_len * d_model + i * d_model],
                    &k_data[b * seq_len * d_model + j * d_model],
                    d_model
                );
                
                // Apply scaling
                score *= scale;
                
                // Apply mask if provided
                if (mask_data) {
                    int mask_idx = b * seq_len * seq_len + i * seq_len + j;
                    if (mask_data[mask_idx] < 0.5f) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                }
                
                attention_scores[i * seq_len + j] = score;
            }
        }
        
        // Apply softmax row-wise and compute attention output in one pass
        for (int i = 0; i < seq_len; ++i) {
            // Find maximum for numerical stability
            float max_score = attention_scores[i * seq_len];
            for (int j = 1; j < seq_len; ++j) {
                max_score = std::max(max_score, attention_scores[i * seq_len + j]);
            }
            
            // Compute softmax weights
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float weight = std::exp(attention_scores[i * seq_len + j] - max_score);
                attention_weights[i * seq_len + j] = weight;
                sum += weight;
            }
            
            // Normalize weights and compute output simultaneously
            if (sum > 0.0f) {
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < seq_len; ++j) {
                    attention_weights[i * seq_len + j] *= inv_sum;
                }
            }
            
            // Compute weighted sum of values
            for (int d = 0; d < d_model; ++d) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    weighted_sum += attention_weights[i * seq_len + j] * 
                                   v_data[b * seq_len * d_model + j * d_model + d];
                }
                output_data[b * seq_len * d_model + i * d_model + d] = weighted_sum;
            }
        }
    }
    
    return output;
}

Core::Tensor KernelFusion::FusedLayerNormReLU(const Core::Tensor& input,
                                             const Core::Tensor& gamma,
                                             const Core::Tensor& beta,
                                             float epsilon) {
    const auto& input_shape = input.GetShape();
    Core::Tensor output(input_shape);
    
    if (input_shape.size() < 2) {
        throw std::invalid_argument("Input must have at least 2 dimensions");
    }
    
    int batch_size = input_shape[0];
    int feature_size = input_shape[input_shape.size() - 1];
    int sequence_length = 1;
    for (size_t i = 1; i < input_shape.size() - 1; ++i) {
        sequence_length *= input_shape[i];
    }
    
    const float* input_data = input.GetData();
    const float* gamma_data = gamma.GetData();
    const float* beta_data = beta.GetData();
    float* output_data = output.GetData();
    
    #pragma omp parallel for if(batch_size * sequence_length > 100)
    for (int batch_seq = 0; batch_seq < batch_size * sequence_length; ++batch_seq) {
        int offset = batch_seq * feature_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < feature_size; ++i) {
            mean += input_data[offset + i];
        }
        mean /= feature_size;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < feature_size; ++i) {
            float diff = input_data[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;
        
        // Normalize and apply ReLU
        float inv_std = 1.0f / std::sqrt(variance + epsilon);
        for (int i = 0; i < feature_size; ++i) {
            float normalized = (input_data[offset + i] - mean) * inv_std;
            float scaled = normalized * gamma_data[i] + beta_data[i];
            output_data[offset + i] = std::max(0.0f, scaled); // Apply ReLU
        }
    }
    
    return output;
}

Core::Tensor KernelFusion::FusedGELUDropout(const Core::Tensor& input, 
                                           float dropout_rate,
                                           bool training) {
    const auto& input_shape = input.GetShape();
    Core::Tensor output(input_shape);
    
    const float* input_data = input.GetData();
    float* output_data = output.GetData();
    int total_size = input.GetSize();
    
    // GELU approximation constants
    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
    
    if (training && dropout_rate > 0.0f) {
        // Apply dropout during training
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        float scale = 1.0f / (1.0f - dropout_rate);
        
        #pragma omp parallel for
        for (int i = 0; i < total_size; ++i) {
            float x = input_data[i];
            
            // GELU activation
            float gelu_x = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
            
            // Apply dropout
            if (dist(gen) < dropout_rate) {
                output_data[i] = 0.0f;
            } else {
                output_data[i] = gelu_x * scale;
            }
        }
    } else {
        // No dropout during inference or when dropout_rate is 0
        #pragma omp parallel for
        for (int i = 0; i < total_size; ++i) {
            float x = input_data[i];
            output_data[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
        }
    }
    
    return output;
}

void KernelFusion::OptimizeComputationGraph(const std::vector<Operation>& operations) {
    // Simple optimization: identify fusion opportunities
    fused_operations_.clear();
    
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& current_op = operations[i];
        bool fused = false;
        
        // Look for common fusion patterns
        if (i + 1 < operations.size()) {
            const auto& next_op = operations[i + 1];
            
            // Linear + ReLU fusion
            if (current_op.type == OperationType::Linear && next_op.type == OperationType::ReLU) {
                FusedOperation fused_op;
                fused_op.type = FusedOperationType::LinearReLU;
                fused_op.input_indices = current_op.input_indices;
                fused_op.output_index = next_op.output_index;
                fused_operations_.push_back(fused_op);
                ++i; // Skip the next operation as it's been fused
                fused = true;
            }
            // LayerNorm + ReLU fusion
            else if (current_op.type == OperationType::LayerNorm && next_op.type == OperationType::ReLU) {
                FusedOperation fused_op;
                fused_op.type = FusedOperationType::LayerNormReLU;
                fused_op.input_indices = current_op.input_indices;
                fused_op.output_index = next_op.output_index;
                fused_operations_.push_back(fused_op);
                ++i; // Skip the next operation
                fused = true;
            }
        }
        
        // Look for GELU + Dropout fusion
        if (!fused && i + 1 < operations.size()) {
            const auto& next_op = operations[i + 1];
            if (current_op.type == OperationType::GELU && next_op.type == OperationType::Dropout) {
                FusedOperation fused_op;
                fused_op.type = FusedOperationType::GELUDropout;
                fused_op.input_indices = current_op.input_indices;
                fused_op.output_index = next_op.output_index;
                fused_operations_.push_back(fused_op);
                ++i; // Skip the next operation
                fused = true;
            }
        }
        
        // If no fusion opportunity found, keep original operation
        if (!fused) {
            FusedOperation single_op;
            single_op.type = FusedOperationType::Single;
            single_op.input_indices = current_op.input_indices;
            single_op.output_index = current_op.output_index;
            single_op.original_operation = current_op;
            fused_operations_.push_back(single_op);
        }
    }
}

KernelFusion::OptimizationStats KernelFusion::GetOptimizationStats() const {
    OptimizationStats stats;
    
    for (const auto& op : fused_operations_) {
        stats.total_operations++;
        
        switch (op.type) {
            case FusedOperationType::LinearReLU:
            case FusedOperationType::LayerNormReLU:
            case FusedOperationType::GELUDropout:
            case FusedOperationType::AttentionSoftmax:
                stats.fused_operations++;
                break;
            case FusedOperationType::Single:
                stats.single_operations++;
                break;
        }
    }
    
    if (stats.total_operations > 0) {
        stats.fusion_ratio = static_cast<float>(stats.fused_operations) / stats.total_operations;
    }
    
    return stats;
}

// Static utility methods for common fusions
Core::Tensor KernelFusion::MatMulBiasActivation(const Core::Tensor& input,
                                               const Core::Tensor& weight,
                                               const Core::Tensor& bias,
                                               ActivationType activation) {
    const auto& input_shape = input.GetShape();
    const auto& weight_shape = weight.GetShape();
    
    if (input_shape.size() != 2 || weight_shape.size() != 2) {
        throw std::invalid_argument("Input and weight must be 2D tensors");
    }
    
    int batch_size = input_shape[0];
    int in_features = input_shape[1];
    int out_features = weight_shape[0];
    
    Core::Tensor output({batch_size, out_features});
    
    const float* input_data = input.GetData();
    const float* weight_data = weight.GetData();
    const float* bias_data = bias.GetSize() > 0 ? bias.GetData() : nullptr;
    float* output_data = output.GetData();
    
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int out_idx = 0; out_idx < out_features; ++out_idx) {
            // Compute matrix multiplication
            float sum = Utils::SIMDUtils::DotProduct(
                &input_data[b * in_features],
                &weight_data[out_idx * in_features],
                in_features
            );
            
            // Add bias
            if (bias_data) {
                sum += bias_data[out_idx];
            }
            
            // Apply activation
            switch (activation) {
                case ActivationType::ReLU:
                    sum = std::max(0.0f, sum);
                    break;
                case ActivationType::GELU: {
                    const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
                    sum = 0.5f * sum * (1.0f + std::tanh(sqrt_2_pi * (sum + 0.044715f * sum * sum * sum)));
                    break;
                }
                case ActivationType::Tanh:
                    sum = std::tanh(sum);
                    break;
                case ActivationType::Sigmoid:
                    sum = 1.0f / (1.0f + std::exp(-sum));
                    break;
                case ActivationType::None:
                default:
                    // No activation
                    break;
            }
            
            output_data[b * out_features + out_idx] = sum;
        }
    }
    
    return output;
}

Core::Tensor KernelFusion::FusedTransformerBlock(const Core::Tensor& input,
                                                const TransformerBlockParams& params) {
    // Implement a simplified fused transformer block
    // This would normally include attention, layer norm, and feed-forward in one kernel
    
    // For now, implement basic operations sequentially
    // In a full implementation, this would be a single optimized kernel
    
    Core::Tensor output = input;
    
    // Self-attention (simplified)
    Core::Tensor attention_output = FusedAttentionSoftmax(
        input, input, input, nullptr, 1.0f / std::sqrt(input.GetShape()[2])
    );
    
    // Add residual connection
    const float* input_data = input.GetData();
    const float* attention_data = attention_output.GetData();
    float* output_data = output.GetData();
    int total_size = input.GetSize();
    
    #pragma omp parallel for
    for (int i = 0; i < total_size; ++i) {
        output_data[i] = input_data[i] + attention_data[i];
    }
    
    // Layer normalization would go here
    // Feed-forward network would go here
    
    return output;
}

} // namespace Optimization
} // namespace crllwtt