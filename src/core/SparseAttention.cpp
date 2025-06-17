/**
 * @file SparseAttention.cpp
 * @brief Implementation of Sparse Attention mechanism for efficient Transformer
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/core/SparseAttention.hpp"
#include "LwTT/utils/Memory.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>

#ifdef LWTT_ENABLE_OPENMP
#include <omp.h>
#endif

namespace crllwtt {
namespace Core {

SparseAttention::SparseAttention(const SparseAttentionConfig& config) 
    : config_(config) {
    // Initialize any required internal state
}

Tensor SparseAttention::Forward(const Tensor& query,
                               const Tensor& key,
                               const Tensor& value,
                               const Tensor* mask) {
    const auto& q_shape = query.GetShape();
    const auto& k_shape = key.GetShape();
    const auto& v_shape = value.GetShape();
    
    // Validate input dimensions
    if (q_shape.size() != 3 || k_shape.size() != 3 || v_shape.size() != 3) {
        throw std::invalid_argument("Query, Key, Value must be 3D tensors [batch, seq_len, d_model]");
    }
    
    int batch_size = q_shape[0];
    int seq_len = q_shape[1];
    int d_model = q_shape[2];
    
    // Compute attention scores: Q * K^T
    Tensor attention_scores = ComputeAttentionScores(query, key);
    
    // Create or use sparse mask
    Tensor sparse_mask;
    if (mask != nullptr) {
        sparse_mask = *mask;
    } else {
        // Create adaptive sparse mask based on attention scores
        if (adaptive_sparsity_) {
            sparse_mask = ComputeAdaptiveMask(attention_scores);
        } else if (config_.use_local_attention) {
            sparse_mask = CreateLocalWindowMask(seq_len);
        } else {
            sparse_mask = CreateSparseMask(seq_len, SparsePatternType::Random);
        }
    }
    
    // Apply sparsity mask to attention scores
    Tensor masked_scores = ApplySparsityMask(attention_scores, sparse_mask);
    
    // Apply softmax to get attention weights
    // Scale by sqrt(d_model) for numerical stability
    float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    
    // Scale and apply softmax row-wise
    Tensor attention_weights({batch_size, seq_len, seq_len});
    const float* scores_data = masked_scores.GetData();
    float* weights_data = attention_weights.GetData();
    
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            // Find maximum for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < seq_len; ++j) {
                int idx = b * seq_len * seq_len + i * seq_len + j;
                max_val = std::max(max_val, scores_data[idx] * scale);
            }
            
            // Compute softmax
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                int idx = b * seq_len * seq_len + i * seq_len + j;
                weights_data[idx] = std::exp(scores_data[idx] * scale - max_val);
                sum += weights_data[idx];
            }
            
            // Normalize
            if (sum > 0.0f) {
                for (int j = 0; j < seq_len; ++j) {
                    int idx = b * seq_len * seq_len + i * seq_len + j;
                    weights_data[idx] /= sum;
                }
            }
        }
    }
    
    // Apply attention weights to values: Attention * V
    Tensor output({batch_size, seq_len, d_model});
    const float* value_data = value.GetData();
    float* output_data = output.GetData();
    
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < d_model; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    int weight_idx = b * seq_len * seq_len + i * seq_len + j;
                    int value_idx = b * seq_len * d_model + j * d_model + d;
                    sum += weights_data[weight_idx] * value_data[value_idx];
                }
                int output_idx = b * seq_len * d_model + i * d_model + d;
                output_data[output_idx] = sum;
            }
        }
    }
    
    return output;
}

Tensor SparseAttention::ComputeAttentionScores(const Tensor& query, const Tensor& key) {
    const auto& q_shape = query.GetShape();
    int batch_size = q_shape[0];
    int seq_len = q_shape[1];
    int d_model = q_shape[2];
    
    Tensor scores({batch_size, seq_len, seq_len});
    const float* q_data = query.GetData();
    const float* k_data = key.GetData();
    float* scores_data = scores.GetData();
    
    // Compute Q * K^T efficiently
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot_product = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    int q_idx = b * seq_len * d_model + i * d_model + d;
                    int k_idx = b * seq_len * d_model + j * d_model + d;
                    dot_product += q_data[q_idx] * k_data[k_idx];
                }
                int scores_idx = b * seq_len * seq_len + i * seq_len + j;
                scores_data[scores_idx] = dot_product;
            }
        }
    }
    
    return scores;
}

Tensor SparseAttention::ApplySparsityMask(const Tensor& attention_scores, const Tensor& mask) {
    const auto& scores_shape = attention_scores.GetShape();
    Tensor masked_scores(scores_shape);
    
    const float* scores_data = attention_scores.GetData();
    const float* mask_data = mask.GetData();
    float* masked_data = masked_scores.GetData();
    
    int total_elements = scores_shape[0] * scores_shape[1] * scores_shape[2];
    
    #pragma omp parallel for
    for (int i = 0; i < total_elements; ++i) {
        masked_data[i] = mask_data[i] > 0.5f ? scores_data[i] : -std::numeric_limits<float>::infinity();
    }
    
    return masked_scores;
}

Tensor SparseAttention::ComputeAdaptiveMask(const Tensor& attention_scores) {
    const auto& shape = attention_scores.GetShape();
    int batch_size = shape[0];
    int seq_len = shape[1];
    
    Tensor mask({batch_size, seq_len, seq_len});
    const float* scores_data = attention_scores.GetData();
    float* mask_data = mask.GetData();
    
    // For each batch and query position, keep top-k attention scores
    int keep_count = static_cast<int>(seq_len * (1.0f - config_.sparsity_ratio));
    keep_count = std::max(1, std::min(seq_len, keep_count));
    
    #pragma omp parallel for if(batch_size > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            // Collect scores for this query position
            std::vector<std::pair<float, int>> score_indices;
            score_indices.reserve(seq_len);
            
            for (int j = 0; j < seq_len; ++j) {
                int scores_idx = b * seq_len * seq_len + i * seq_len + j;
                score_indices.emplace_back(scores_data[scores_idx], j);
            }
            
            // Sort by score (descending)
            std::partial_sort(score_indices.begin(), 
                            score_indices.begin() + keep_count,
                            score_indices.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Create mask: 1 for kept positions, 0 for pruned
            std::fill_n(mask_data + b * seq_len * seq_len + i * seq_len, seq_len, 0.0f);
            for (int k = 0; k < keep_count; ++k) {
                int j = score_indices[k].second;
                int mask_idx = b * seq_len * seq_len + i * seq_len + j;
                mask_data[mask_idx] = 1.0f;
            }
        }
    }
    
    return mask;
}

Tensor SparseAttention::CreateLocalWindowMask(int seq_len) {
    Tensor mask({1, seq_len, seq_len});
    float* mask_data = mask.GetData();
    
    // Initialize all to 0
    std::fill_n(mask_data, seq_len * seq_len, 0.0f);
    
    int window_size = config_.local_window_size;
    int half_window = window_size / 2;
    
    for (int i = 0; i < seq_len; ++i) {
        int start = std::max(0, i - half_window);
        int end = std::min(seq_len, i + half_window + 1);
        
        for (int j = start; j < end; ++j) {
            int mask_idx = i * seq_len + j;
            mask_data[mask_idx] = 1.0f;
        }
    }
    
    return mask;
}

Tensor SparseAttention::CreateBlockSparseMask(int seq_len) {
    Tensor mask({1, seq_len, seq_len});
    float* mask_data = mask.GetData();
    
    // Initialize all to 0
    std::fill_n(mask_data, seq_len * seq_len, 0.0f);
    
    int block_size = config_.block_size;
    int num_blocks = (seq_len + block_size - 1) / block_size;
    
    for (int block_i = 0; block_i < num_blocks; ++block_i) {
        for (int block_j = 0; block_j < num_blocks; ++block_j) {
            // Enable blocks on diagonal and some off-diagonal blocks
            bool enable_block = (block_i == block_j) || 
                              (std::abs(block_i - block_j) == 1) ||
                              (static_cast<float>(rand()) / RAND_MAX < config_.sparsity_ratio);
            
            if (enable_block) {
                int start_i = block_i * block_size;
                int end_i = std::min(seq_len, start_i + block_size);
                int start_j = block_j * block_size;
                int end_j = std::min(seq_len, start_j + block_size);
                
                for (int i = start_i; i < end_i; ++i) {
                    for (int j = start_j; j < end_j; ++j) {
                        int mask_idx = i * seq_len + j;
                        mask_data[mask_idx] = 1.0f;
                    }
                }
            }
        }
    }
    
    return mask;
}

Tensor SparseAttention::CreateSparseMask(int seq_len, SparsePatternType pattern_type) {
    switch (pattern_type) {
        case SparsePatternType::LocalWindow:
            return CreateLocalWindowMask(seq_len);
            
        case SparsePatternType::BlockSparse:
            return CreateBlockSparseMask(seq_len);
            
        case SparsePatternType::Random: {
            Tensor mask({1, seq_len, seq_len});
            float* mask_data = mask.GetData();
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            
            float keep_prob = 1.0f - config_.sparsity_ratio;
            for (int i = 0; i < seq_len * seq_len; ++i) {
                mask_data[i] = (dis(gen) < keep_prob) ? 1.0f : 0.0f;
            }
            return mask;
        }
        
        case SparsePatternType::Structured: {
            return SparseAttentionUtils::CreateStridedMask(seq_len, 2); // stride of 2
        }
        
        case SparsePatternType::Adaptive:
        default:
            // Will be computed dynamically based on attention scores
            Tensor mask({1, seq_len, seq_len});
            float* mask_data = mask.GetData();
            std::fill_n(mask_data, seq_len * seq_len, 1.0f); // Default to dense
            return mask;
    }
}

// Utility functions implementation
namespace SparseAttentionUtils {

float ComputeSparsity(const Tensor& attention_weights) {
    const float* data = attention_weights.GetData();
    int total_elements = attention_weights.GetSize();
    
    int zero_count = 0;
    const float epsilon = 1e-8f;
    
    for (int i = 0; i < total_elements; ++i) {
        if (std::abs(data[i]) < epsilon) {
            ++zero_count;
        }
    }
    
    return static_cast<float>(zero_count) / static_cast<float>(total_elements);
}

Tensor CreateStridedMask(int seq_len, int stride) {
    Tensor mask({1, seq_len, seq_len});
    float* mask_data = mask.GetData();
    
    // Initialize all to 0
    std::fill_n(mask_data, seq_len * seq_len, 0.0f);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; j += stride) {
            int mask_idx = i * seq_len + j;
            mask_data[mask_idx] = 1.0f;
        }
    }
    
    return mask;
}

Tensor OptimizeSparsityPattern(const Tensor& query, const Tensor& key, float target_sparsity) {
    // Compute attention scores
    const auto& q_shape = query.GetShape();
    int batch_size = q_shape[0];
    int seq_len = q_shape[1];
    int d_model = q_shape[2];
    
    // Simple implementation: compute Q*K^T and keep top-k based on magnitude
    Tensor scores({batch_size, seq_len, seq_len});
    const float* q_data = query.GetData();
    const float* k_data = key.GetData();
    float* scores_data = scores.GetData();
    
    // Compute attention scores
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot_product = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    int q_idx = b * seq_len * d_model + i * d_model + d;
                    int k_idx = b * seq_len * d_model + j * d_model + d;
                    dot_product += q_data[q_idx] * k_data[k_idx];
                }
                int scores_idx = b * seq_len * seq_len + i * seq_len + j;
                scores_data[scores_idx] = std::abs(dot_product);
            }
        }
    }
    
    // Create mask by keeping top-(1-target_sparsity) elements
    Tensor mask({batch_size, seq_len, seq_len});
    float* mask_data = mask.GetData();
    
    int keep_count = static_cast<int>(seq_len * seq_len * (1.0f - target_sparsity));
    keep_count = std::max(1, std::min(seq_len * seq_len, keep_count));
    
    for (int b = 0; b < batch_size; ++b) {
        std::vector<std::pair<float, int>> score_indices;
        score_indices.reserve(seq_len * seq_len);
        
        for (int idx = 0; idx < seq_len * seq_len; ++idx) {
            int scores_idx = b * seq_len * seq_len + idx;
            score_indices.emplace_back(scores_data[scores_idx], idx);
        }
        
        // Sort by score (descending)
        std::partial_sort(score_indices.begin(),
                        score_indices.begin() + keep_count,
                        score_indices.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Create mask
        std::fill_n(mask_data + b * seq_len * seq_len, seq_len * seq_len, 0.0f);
        for (int k = 0; k < keep_count; ++k) {
            int idx = score_indices[k].second;
            int mask_idx = b * seq_len * seq_len + idx;
            mask_data[mask_idx] = 1.0f;
        }
    }
    
    return mask;
}

} // namespace SparseAttentionUtils

} // namespace Core
} // namespace crllwtt