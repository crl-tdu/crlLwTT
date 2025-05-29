/**
 * @file TransformerBlock.hpp
 * @brief Transformer Block implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_LAYERS_TRANSFORMER_BLOCK_HPP
#define LWTT_LAYERS_TRANSFORMER_BLOCK_HPP

#include "../core/Tensor.hpp"
#include "../core/TimeEncoding.hpp"
#include <memory>

namespace LwTT {
namespace Layers {

/**
 * @brief Configuration for Transformer Block
 */
struct TransformerBlockConfig {
    int d_model = 256;           ///< Model dimension
    int n_heads = 8;             ///< Number of attention heads
    int d_ff = 1024;             ///< Feed-forward dimension
    float dropout_rate = 0.1f;   ///< Dropout rate
    float layer_norm_eps = 1e-6f; ///< Layer normalization epsilon
    bool use_pre_norm = true;    ///< Use pre-normalization
};

/**
 * @brief Transformer Block layer
 */
class TransformerBlock {
public:
    /**
     * @brief Constructor
     * @param config Block configuration
     */
    explicit TransformerBlock(const TransformerBlockConfig& config);

    /**
     * @brief Destructor
     */
    ~TransformerBlock();

    /**
     * @brief Forward pass
     * @param input Input tensor [batch_size, seq_len, d_model]
     * @param mask Attention mask (optional)
     * @param time_info Time information (optional)
     * @return Output tensor [batch_size, seq_len, d_model]
     */
    Core::Tensor Forward(const Core::Tensor& input,
                        const Core::Tensor* mask = nullptr,
                        const Core::TimeInfo* time_info = nullptr);

    /**
     * @brief Get attention weights from the last forward pass
     * @return Attention weights tensor
     */
    Core::Tensor GetAttentionWeights() const;

    /**
     * @brief Set training mode
     * @param training Training mode flag
     */
    void SetTraining(bool training) { training_ = training; }

    /**
     * @brief Check if in training mode
     * @return true if in training mode
     */
    bool IsTraining() const { return training_; }

    /**
     * @brief Get configuration
     * @return Configuration
     */
    const TransformerBlockConfig& GetConfig() const { return config_; }

private:
    TransformerBlockConfig config_;
    bool training_ = false;
    
    // Components (forward declarations)
    class MultiHeadAttention;
    class FeedForward;
    class LayerNorm;
    
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForward> feed_forward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
    
    mutable Core::Tensor attention_weights_;
};

} // namespace Layers
} // namespace LwTT

#endif // LWTT_LAYERS_TRANSFORMER_BLOCK_HPP
