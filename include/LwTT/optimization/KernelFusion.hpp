/**
 * @file KernelFusion.hpp
 * @brief Kernel fusion optimizations for efficient computation
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_OPTIMIZATION_KERNEL_FUSION_HPP
#define LWTT_OPTIMIZATION_KERNEL_FUSION_HPP

#include "../core/Tensor.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <random>

namespace crllwtt {
namespace Optimization {

/**
 * @brief Configuration for kernel fusion optimizations
 */
struct KernelFusionConfig {
    bool enable_linear_relu_fusion = true;      // Fuse linear layers with ReLU
    bool enable_attention_fusion = true;        // Fuse attention computation
    bool enable_layernorm_fusion = true;        // Fuse layer normalization with activations
    bool enable_gelu_dropout_fusion = true;     // Fuse GELU with dropout
    bool enable_transformer_block_fusion = true; // Fuse entire transformer blocks
    float fusion_threshold = 0.1f;              // Minimum benefit threshold for fusion
};

/**
 * @brief Types of operations that can be fused
 */
enum class OperationType {
    Linear,
    ReLU,
    GELU,
    Tanh,
    Sigmoid,
    LayerNorm,
    Attention,
    Dropout,
    Softmax
};

/**
 * @brief Types of fused operations
 */
enum class FusedOperationType {
    LinearReLU,
    LayerNormReLU,
    GELUDropout,
    AttentionSoftmax,
    TransformerBlock,
    Single  // Not fused
};

/**
 * @brief Activation function types for fusion
 */
enum class ActivationType {
    None,
    ReLU,
    GELU,
    Tanh,
    Sigmoid
};

/**
 * @brief Operation description for fusion optimization
 */
struct Operation {
    OperationType type;
    std::vector<int> input_indices;
    int output_index;
    std::unordered_map<std::string, float> parameters;
};

/**
 * @brief Fused operation description
 */
struct FusedOperation {
    FusedOperationType type;
    std::vector<int> input_indices;
    int output_index;
    Operation original_operation;  // For single operations
    std::unordered_map<std::string, float> parameters;
};

/**
 * @brief Parameters for transformer block fusion
 */
struct TransformerBlockParams {
    int d_model;
    int num_heads;
    int ff_dim;
    float dropout_rate;
    bool use_gelu;
};

/**
 * @brief Kernel fusion optimization engine
 */
class KernelFusion {
public:
    explicit KernelFusion(const KernelFusionConfig& config = KernelFusionConfig{});
    ~KernelFusion() = default;

    /**
     * @brief Fused linear transformation with ReLU activation
     * @param input Input tensor [batch_size, in_features]
     * @param weight Weight matrix [out_features, in_features]
     * @param bias Bias vector [out_features]
     * @return Output tensor [batch_size, out_features]
     */
    Core::Tensor FusedLinearReLU(const Core::Tensor& input, 
                                 const Core::Tensor& weight, 
                                 const Core::Tensor& bias);

    /**
     * @brief Fused attention computation with softmax
     * @param query Query tensor [batch, seq_len, d_model]
     * @param key Key tensor [batch, seq_len, d_model]
     * @param value Value tensor [batch, seq_len, d_model]
     * @param mask Optional attention mask
     * @param scale Scaling factor (typically 1/sqrt(d_model))
     * @return Attention output [batch, seq_len, d_model]
     */
    Core::Tensor FusedAttentionSoftmax(const Core::Tensor& query,
                                      const Core::Tensor& key,
                                      const Core::Tensor& value,
                                      const Core::Tensor* mask = nullptr,
                                      float scale = 1.0f);

    /**
     * @brief Fused layer normalization with ReLU
     * @param input Input tensor
     * @param gamma Scale parameters
     * @param beta Shift parameters
     * @param epsilon Small constant for numerical stability
     * @return Normalized and activated output
     */
    Core::Tensor FusedLayerNormReLU(const Core::Tensor& input,
                                   const Core::Tensor& gamma,
                                   const Core::Tensor& beta,
                                   float epsilon = 1e-5f);

    /**
     * @brief Fused GELU activation with dropout
     * @param input Input tensor
     * @param dropout_rate Dropout probability
     * @param training Whether in training mode
     * @return GELU-activated output with dropout applied
     */
    Core::Tensor FusedGELUDropout(const Core::Tensor& input, 
                                 float dropout_rate = 0.1f,
                                 bool training = true);

    /**
     * @brief Optimize computation graph for fusion opportunities
     * @param operations List of operations to optimize
     */
    void OptimizeComputationGraph(const std::vector<Operation>& operations);

    /**
     * @brief Get optimization statistics
     */
    struct OptimizationStats {
        int total_operations = 0;
        int fused_operations = 0;
        int single_operations = 0;
        float fusion_ratio = 0.0f;
        float estimated_speedup = 1.0f;
    };
    
    OptimizationStats GetOptimizationStats() const;

    /**
     * @brief Get the list of fused operations after optimization
     * @return Vector of fused operations
     */
    const std::vector<FusedOperation>& GetFusedOperations() const {
        return fused_operations_;
    }

    /**
     * @brief Static utility: Matrix multiplication with bias and activation
     * @param input Input tensor
     * @param weight Weight matrix
     * @param bias Bias vector
     * @param activation Activation function to apply
     * @return Output tensor
     */
    static Core::Tensor MatMulBiasActivation(const Core::Tensor& input,
                                            const Core::Tensor& weight,
                                            const Core::Tensor& bias,
                                            ActivationType activation = ActivationType::None);

    /**
     * @brief Fused transformer block computation
     * @param input Input tensor [batch, seq_len, d_model]
     * @param params Transformer block parameters
     * @return Transformer block output
     */
    Core::Tensor FusedTransformerBlock(const Core::Tensor& input,
                                      const TransformerBlockParams& params);

private:
    KernelFusionConfig config_;
    std::vector<FusedOperation> fused_operations_;

    // Internal methods for fusion analysis
    bool CanFuseOperations(const Operation& op1, const Operation& op2) const;
    float EstimateFusionBenefit(const Operation& op1, const Operation& op2) const;
    FusedOperationType DetermineFusionType(const Operation& op1, const Operation& op2) const;
};

/**
 * @brief High-level fusion patterns for common operations
 */
namespace FusionPatterns {

    /**
     * @brief Fused multi-head attention (complete computation in one kernel)
     * @param input Input tensor [batch, seq_len, d_model]
     * @param wq Query weight matrix
     * @param wk Key weight matrix  
     * @param wv Value weight matrix
     * @param wo Output projection weight
     * @param num_heads Number of attention heads
     * @param mask Optional attention mask
     * @return Multi-head attention output
     */
    Core::Tensor FusedMultiHeadAttention(
        const Core::Tensor& input,
        const Core::Tensor& wq,
        const Core::Tensor& wk, 
        const Core::Tensor& wv,
        const Core::Tensor& wo,
        int num_heads,
        const Core::Tensor* mask = nullptr);

    /**
     * @brief Fused feed-forward network (Linear + GELU + Linear)
     * @param input Input tensor
     * @param w1 First linear layer weight
     * @param b1 First linear layer bias
     * @param w2 Second linear layer weight
     * @param b2 Second linear layer bias
     * @param dropout_rate Dropout rate
     * @param training Whether in training mode
     * @return Feed-forward network output
     */
    Core::Tensor FusedFeedForward(
        const Core::Tensor& input,
        const Core::Tensor& w1,
        const Core::Tensor& b1,
        const Core::Tensor& w2,
        const Core::Tensor& b2,
        float dropout_rate = 0.1f,
        bool training = true);

    /**
     * @brief Fused residual connection with layer norm
     * @param input Main input tensor
     * @param residual Residual tensor to add
     * @param gamma Layer norm scale
     * @param beta Layer norm bias
     * @param epsilon Numerical stability constant
     * @return Normalized residual output
     */
    Core::Tensor FusedResidualLayerNorm(
        const Core::Tensor& input,
        const Core::Tensor& residual,
        const Core::Tensor& gamma,
        const Core::Tensor& beta,
        float epsilon = 1e-5f);

} // namespace FusionPatterns

} // namespace Optimization
} // namespace crllwtt

#endif // LWTT_OPTIMIZATION_KERNEL_FUSION_HPP