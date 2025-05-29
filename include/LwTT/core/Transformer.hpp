/**
 * @file Transformer.hpp
 * @brief Lightweight Time-aware Transformer Core Implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_CORE_TRANSFORMER_HPP
#define LWTT_CORE_TRANSFORMER_HPP

#include "Tensor.hpp"
#include "TimeEncoding.hpp"
#include "SparseAttention.hpp"
#include "../layers/TransformerBlock.hpp"
#include "../utils/Memory.hpp"
#include "../utils/Threading.hpp"
#include <vector>
#include <memory>
#include <fstream>
#include <iosfwd>

namespace LwTT {
namespace Core {

/**
 * @brief Configuration for the Transformer model
 */
struct TransformerConfig {
    // Model architecture
    int d_model = 256;           ///< Model dimension
    int n_heads = 8;             ///< Number of attention heads
    int n_layers = 4;            ///< Number of transformer layers
    int d_ff = 1024;             ///< Feed-forward dimension
    int max_seq_len = 512;       ///< Maximum sequence length
    int vocab_size = 10000;      ///< Vocabulary size (if using embeddings)

    // Time-aware features
    bool enable_time_encoding = true;    ///< Enable time-aware positional encoding
    float time_scale = 1.0f;             ///< Time scaling factor
    int personal_embed_dim = 32;         ///< Personal embedding dimension

    // Efficiency optimizations
    bool use_sparse_attention = true;    ///< Use sparse attention mechanism
    float sparsity_ratio = 0.1f;         ///< Sparsity ratio for attention
    bool enable_kernel_fusion = true;    ///< Enable kernel fusion

    // Regularization
    float dropout_rate = 0.1f;           ///< Dropout rate
    float layer_norm_eps = 1e-6f;        ///< Layer normalization epsilon

    // Memory optimization
    bool gradient_checkpointing = false; ///< Enable gradient checkpointing
    int memory_pool_size_mb = 256;       ///< Memory pool size

    // Quantization
    bool enable_quantization = false;    ///< Enable quantization
    int quantization_bits = 8;           ///< Quantization bits

    // Threading
    int num_threads = 0;                 ///< Number of threads (0 = auto)

    // Uncertainty estimation
    bool enable_uncertainty = true;      ///< Enable uncertainty estimation
    int mc_samples = 10;                 ///< Number of Monte Carlo samples for uncertainty
    float uncertainty_threshold = 0.1f;  ///< Uncertainty threshold for warnings
};

/**
 * @brief Lightweight Time-aware Transformer model
 */
class Transformer {
public:
    /**
     * @brief Constructor
     * @param config Model configuration
     */
    explicit Transformer(const TransformerConfig& config);

    /**
     * @brief Destructor
     */
    ~Transformer();

    /**
     * @brief Forward pass
     * @param input Input tensor [batch_size, seq_len, d_model]
     * @param mask Attention mask (optional)
     * @param time_info Time information for time-aware encoding (optional)
     * @param personal_id Personal ID for personalization (optional)
     * @return Output tensor [batch_size, seq_len, d_model]
     */
    Tensor Forward(const Tensor& input,
                   const Tensor* mask = nullptr,
                   const TimeInfo* time_info = nullptr,
                   int personal_id = -1);

    /**
     * @brief Forward pass with uncertainty estimation
     * @param input Input tensor
     * @param mask Attention mask (optional)
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Pair of (output, uncertainty)
     */
    std::pair<Tensor, Tensor> ForwardWithUncertainty(
        const Tensor& input,
        const Tensor* mask = nullptr,
        const TimeInfo* time_info = nullptr,
        int personal_id = -1);

    /**
     * @brief Multi-step prediction
     * @param input Input tensor
     * @param n_steps Number of prediction steps
     * @param mask Attention mask (optional)
     * @param time_info Time information (optional)
     * @return Multi-step predictions [batch_size, n_steps, d_model]
     */
    Tensor PredictMultiStep(const Tensor& input,
                           int n_steps,
                           const Tensor* mask = nullptr,
                           const TimeInfo* time_info = nullptr);

    /**
     * @brief Get attention weights for interpretability
     * @param layer_idx Layer index (-1 for all layers)
     * @return Attention weights
     */
    std::vector<Tensor> GetAttentionWeights(int layer_idx = -1) const;

    /**
     * @brief Enable/disable training mode
     * @param training Training mode flag
     */
    void SetTraining(bool training);

    /**
     * @brief Check if model is in training mode
     * @return true if in training mode
     */
    bool IsTraining() const { return training_; }

    /**
     * @brief Get model configuration
     * @return Configuration
     */
    const TransformerConfig& GetConfig() const { return config_; }

    /**
     * @brief Get number of parameters
     * @return Parameter count
     */
    size_t GetParameterCount() const;

    /**
     * @brief Get memory usage in bytes
     * @return Memory usage
     */
    size_t GetMemoryUsage() const;

    /**
     * @brief Optimize model for inference
     * @param optimization_level Optimization level (0-3)
     */
    void OptimizeForInference(int optimization_level = 2);

    /**
     * @brief Save model to file
     * @param filepath File path
     * @return true if successful
     */
    bool SaveModel(const std::string& filepath) const;

    /**
     * @brief Load model from file
     * @param filepath File path
     * @return true if successful
     */
    bool LoadModel(const std::string& filepath);

    /**
     * @brief Reset internal state
     */
    void Reset();

    /**
     * @brief Enable profiling
     * @param enable Enable profiling
     */
    void EnableProfiling(bool enable) { profiling_enabled_ = enable; }

    /**
     * @brief Get profiling results
     * @return Profiling data
     */
    std::string GetProfilingResults() const;

private:
    // Configuration
    TransformerConfig config_;

    // Model components
    std::unique_ptr<TimeEncoding> time_encoding_;
    std::vector<std::unique_ptr<Layers::TransformerBlock>> layers_;
    std::unique_ptr<Utils::MemoryPool> memory_pool_;
    std::unique_ptr<Utils::ThreadPool> thread_pool_;

    // State
    bool training_ = false;
    bool profiling_enabled_ = false;
    mutable std::vector<Tensor> attention_weights_;

    // Internal buffers
    mutable Tensor hidden_buffer_;
    mutable Tensor attention_buffer_;
    mutable Tensor temp_buffer_;

    // Performance counters
    mutable size_t forward_count_ = 0;
    mutable double total_forward_time_ = 0.0;
    mutable size_t peak_memory_usage_ = 0;

    // Private methods
    void InitializeComponents();
    void AllocateBuffers();
    Tensor ApplyTimeEncoding(const Tensor& input,
                           const TimeInfo* time_info,
                           int personal_id) const;
    void UpdateProfilingStats(double elapsed_time, size_t memory_usage) const;
    Tensor CreateAttentionMask(const Tensor& input, const Tensor* external_mask) const;
    void ValidateInput(const Tensor& input) const;

    // Model serialization helper methods
    bool WriteModelHeader(std::ofstream& file) const;
    bool WriteModelConfig(std::ofstream& file) const;
    bool WriteModelParameters(std::ofstream& file) const;
    bool WriteTensorData(std::ofstream& file, const Tensor& tensor) const;
    
    bool ReadModelHeader(std::ifstream& file) const;
    bool ReadModelConfig(std::ifstream& file, TransformerConfig& config) const;
    bool ReadModelParameters(std::ifstream& file) const;
    bool ReadTensorData(std::ifstream& file, Tensor& tensor) const;
    
    bool ValidateConfigCompatibility(const TransformerConfig& loaded_config) const;
    uint32_t CalculateChecksum() const;
};

/**
 * @brief Time-aware Transformer Builder for easy model construction
 */
class TransformerBuilder {
public:
    TransformerBuilder() = default;

    TransformerBuilder& SetModelDimension(int d_model) {
        config_.d_model = d_model;
        return *this;
    }

    TransformerBuilder& SetNumHeads(int n_heads) {
        config_.n_heads = n_heads;
        return *this;
    }

    TransformerBuilder& SetNumLayers(int n_layers) {
        config_.n_layers = n_layers;
        return *this;
    }

    TransformerBuilder& SetFeedForwardDim(int d_ff) {
        config_.d_ff = d_ff;
        return *this;
    }

    TransformerBuilder& SetMaxSequenceLength(int max_seq_len) {
        config_.max_seq_len = max_seq_len;
        return *this;
    }

    TransformerBuilder& EnableTimeAwareness(bool enable = true, float time_scale = 1.0f) {
        config_.enable_time_encoding = enable;
        config_.time_scale = time_scale;
        return *this;
    }

    TransformerBuilder& EnableSparseAttention(bool enable = true, float sparsity_ratio = 0.1f) {
        config_.use_sparse_attention = enable;
        config_.sparsity_ratio = sparsity_ratio;
        return *this;
    }

    TransformerBuilder& SetDropoutRate(float dropout_rate) {
        config_.dropout_rate = dropout_rate;
        return *this;
    }

    TransformerBuilder& EnableQuantization(bool enable = true, int bits = 8) {
        config_.enable_quantization = enable;
        config_.quantization_bits = bits;
        return *this;
    }

    TransformerBuilder& SetNumThreads(int num_threads) {
        config_.num_threads = num_threads;
        return *this;
    }

    TransformerBuilder& SetMemoryPoolSize(int size_mb) {
        config_.memory_pool_size_mb = size_mb;
        return *this;
    }

    TransformerBuilder& EnableUncertainty(bool enable = true, int mc_samples = 10) {
        config_.enable_uncertainty = enable;
        config_.mc_samples = mc_samples;
        return *this;
    }

    TransformerBuilder& SetUncertaintyThreshold(float threshold) {
        config_.uncertainty_threshold = threshold;
        return *this;
    }

    std::unique_ptr<Transformer> Build() {
        return std::make_unique<Transformer>(config_);
    }

    const TransformerConfig& GetConfig() const { return config_; }

private:
    TransformerConfig config_;
};

/**
 * @brief Utility functions for Transformer operations
 */
namespace TransformerUtils {

    /**
     * @brief Create causal mask for autoregressive generation
     * @param seq_len Sequence length
     * @return Causal mask tensor
     */
    Tensor CreateCausalMask(int seq_len);

    /**
     * @brief Create padding mask from sequence lengths
     * @param seq_lengths Sequence lengths [batch_size]
     * @param max_len Maximum sequence length
     * @return Padding mask tensor
     */
    Tensor CreatePaddingMask(const std::vector<int>& seq_lengths, int max_len);

    /**
     * @brief Generate positional encoding
     * @param seq_len Sequence length
     * @param d_model Model dimension
     * @return Positional encoding tensor
     */
    Tensor GeneratePositionalEncoding(int seq_len, int d_model);

    /**
     * @brief Apply temperature scaling to attention weights
     * @param attention Attention weights
     * @param temperature Temperature parameter
     * @return Scaled attention weights
     */
    Tensor ApplyTemperatureScaling(const Tensor& attention, float temperature);

    /**
     * @brief Compute attention entropy for uncertainty estimation
     * @param attention Attention weights
     * @return Attention entropy
     */
    Tensor ComputeAttentionEntropy(const Tensor& attention);

    /**
     * @brief Beam search for sequence generation
     * @param model Transformer model
     * @param initial_input Initial input
     * @param beam_size Beam size
     * @param max_length Maximum generation length
     * @return Generated sequences
     */
    std::vector<std::vector<int>> BeamSearch(
        const Transformer& model,
        const Tensor& initial_input,
        int beam_size,
        int max_length
    );

} // namespace TransformerUtils

} // namespace Core
} // namespace LwTT

#endif // LWTT_CORE_TRANSFORMER_HPP