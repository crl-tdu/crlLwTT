/**
 * @file SparseAttention.hpp
 * @brief Sparse Attention mechanism for efficient Transformer implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_CORE_SPARSE_ATTENTION_HPP
#define LWTT_CORE_SPARSE_ATTENTION_HPP

#include "Tensor.hpp"
#include <vector>
#include <memory>

namespace LwTT {
namespace Core {

/**
 * @brief Configuration for sparse attention
 */
struct SparseAttentionConfig {
    float sparsity_ratio = 0.1f;        ///< Ratio of attention weights to keep
    int block_size = 64;                ///< Block size for block-sparse attention
    bool use_random_sparsity = false;   ///< Use random sparsity pattern
    bool use_local_attention = true;    ///< Use local attention window
    int local_window_size = 128;        ///< Local attention window size
};

/**
 * @brief Sparse attention pattern types
 */
enum class SparsePatternType {
    Random,         ///< Random sparsity pattern
    Structured,     ///< Structured sparsity (e.g., strided)
    LocalWindow,    ///< Local attention window
    BlockSparse,    ///< Block-sparse attention
    Adaptive        ///< Adaptive sparsity based on attention scores
};

/**
 * @brief Sparse Attention implementation
 */
class SparseAttention {
public:
    /**
     * @brief Constructor
     * @param config Sparse attention configuration
     */
    explicit SparseAttention(const SparseAttentionConfig& config = SparseAttentionConfig{});

    /**
     * @brief Destructor
     */
    ~SparseAttention() = default;

    /**
     * @brief Apply sparse attention
     * @param query Query tensor [batch_size, seq_len, d_model]
     * @param key Key tensor [batch_size, seq_len, d_model]
     * @param value Value tensor [batch_size, seq_len, d_model]
     * @param mask Optional attention mask
     * @return Attention output tensor
     */
    Tensor Forward(const Tensor& query,
                   const Tensor& key,
                   const Tensor& value,
                   const Tensor* mask = nullptr);

    /**
     * @brief Create sparse attention mask
     * @param seq_len Sequence length
     * @param pattern_type Sparsity pattern type
     * @return Sparse attention mask
     */
    Tensor CreateSparseMask(int seq_len, SparsePatternType pattern_type = SparsePatternType::Adaptive);

    /**
     * @brief Set sparsity ratio
     * @param ratio Sparsity ratio (0.0 to 1.0)
     */
    void SetSparsityRatio(float ratio) { config_.sparsity_ratio = ratio; }

    /**
     * @brief Get current sparsity ratio
     * @return Current sparsity ratio
     */
    float GetSparsityRatio() const { return config_.sparsity_ratio; }

    /**
     * @brief Enable/disable adaptive sparsity
     * @param enable Enable adaptive sparsity
     */
    void SetAdaptiveSparsity(bool enable) { adaptive_sparsity_ = enable; }

    /**
     * @brief Get configuration
     * @return Configuration
     */
    const SparseAttentionConfig& GetConfig() const { return config_; }

private:
    SparseAttentionConfig config_;
    bool adaptive_sparsity_ = true;
    
    // Internal methods
    Tensor ComputeAttentionScores(const Tensor& query, const Tensor& key);
    Tensor ApplySparsityMask(const Tensor& attention_scores, const Tensor& mask);
    Tensor ComputeAdaptiveMask(const Tensor& attention_scores);
    Tensor CreateLocalWindowMask(int seq_len);
    Tensor CreateBlockSparseMask(int seq_len);
};

/**
 * @brief Utility functions for sparse attention
 */
namespace SparseAttentionUtils {
    
    /**
     * @brief Compute attention sparsity statistics
     * @param attention_weights Attention weights tensor
     * @return Sparsity ratio (fraction of zero weights)
     */
    float ComputeSparsity(const Tensor& attention_weights);

    /**
     * @brief Create strided sparsity pattern
     * @param seq_len Sequence length
     * @param stride Stride value
     * @return Sparse mask tensor
     */
    Tensor CreateStridedMask(int seq_len, int stride);

    /**
     * @brief Optimize sparse attention pattern based on input
     * @param query Query tensor
     * @param key Key tensor
     * @param target_sparsity Target sparsity ratio
     * @return Optimized sparse mask
     */
    Tensor OptimizeSparsityPattern(const Tensor& query, const Tensor& key, float target_sparsity);

} // namespace SparseAttentionUtils

} // namespace Core
} // namespace LwTT

#endif // LWTT_CORE_SPARSE_ATTENTION_HPP
