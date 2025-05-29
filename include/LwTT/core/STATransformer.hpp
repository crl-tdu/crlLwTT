/**
 * @file STATransformer.hpp
 * @brief STA (Sense The Ambience) Architecture Implementation with Lightweight Time-aware Transformer
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_CORE_STA_TRANSFORMER_HPP
#define LWTT_CORE_STA_TRANSFORMER_HPP

#include "Transformer.hpp"
#include "TimeEncoding.hpp"
#include "Tensor.hpp"
#include <functional>
#include <memory>
#include <unordered_map>

namespace crllwtt {
namespace Core {

/**
 * @brief Configuration for STA Transformer
 */
struct STAConfig {
    // Base transformer configuration
    TransformerConfig transformer_config;
    
    // STA-specific parameters
    int observable_state_dim = 32;           ///< Dimension of observable human state x
    int controllable_input_dim = 16;         ///< Dimension of controllable environmental input u
    int predicted_state_dim = 8;             ///< Dimension of predicted future state s
    
    // Learning parameters
    float learning_rate = 0.001f;            ///< Learning rate for real-time adaptation
    float momentum = 0.9f;                   ///< Momentum for gradient updates
    float sensitivity_threshold = 0.01f;     ///< Threshold for sensitivity-based control
    float control_gain = 0.1f;               ///< Control gain (eta_u)
    
    // Uncertainty estimation
    bool enable_uncertainty = true;          ///< Enable uncertainty estimation
    int ensemble_size = 5;                   ///< Number of models in ensemble
    float uncertainty_weight = 0.1f;         ///< Weight for uncertainty in control
    
    // Personal adaptation
    bool enable_personal_adaptation = true;  ///< Enable personal adaptation
    int max_persons = 100;                   ///< Maximum number of persons
    float personal_learning_rate = 0.0001f;  ///< Learning rate for personal embeddings
    
    // Real-time constraints
    float max_inference_time_ms = 1.0f;      ///< Maximum inference time in milliseconds
    int buffer_size = 1000;                  ///< Size of experience buffer
    bool enable_streaming = true;            ///< Enable streaming processing
};

/**
 * @brief Meta-evaluation function interface
 */
class MetaEvaluationFunction {
public:
    virtual ~MetaEvaluationFunction() = default;
    
    /**
     * @brief Evaluate predicted state
     * @param predicted_state Predicted future state
     * @param uncertainty Uncertainty estimate (optional)
     * @return Evaluation score (higher is better)
     */
    virtual float Evaluate(const Tensor& predicted_state, 
                          const Tensor* uncertainty = nullptr) const = 0;
    
    /**
     * @brief Compute gradient of evaluation function
     * @param predicted_state Predicted future state
     * @return Gradient w.r.t. predicted state
     */
    virtual Tensor ComputeGradient(const Tensor& predicted_state) const = 0;
    
    /**
     * @brief Update evaluation function based on feedback
     * @param predicted_state Predicted state
     * @param actual_outcome Actual outcome
     * @param feedback Implicit feedback (e.g., user actions)
     */
    virtual void UpdateFromFeedback(const Tensor& /*predicted_state*/,
                                   const Tensor& /*actual_outcome*/,
                                   const Tensor& /*feedback*/) {}
};

/**
 * @brief Default meta-evaluation function (distance to target state)
 */
class TargetStateEvaluator : public MetaEvaluationFunction {
public:
    explicit TargetStateEvaluator(const Tensor& target_state, float weight = 1.0f)
        : target_state_(target_state), weight_(weight) {}
    
    float Evaluate(const Tensor& predicted_state, 
                   const Tensor* uncertainty = nullptr) const override;
    
    Tensor ComputeGradient(const Tensor& predicted_state) const override;
    
    void SetTargetState(const Tensor& target_state) { target_state_ = target_state; }
    
private:
    Tensor target_state_;
    float weight_;
};

/**
 * @brief Experience buffer for storing interaction history
 */
struct ExperienceEntry {
    Tensor observable_state;      ///< x[k-1]
    Tensor controllable_input;    ///< u[k-1]
    Tensor predicted_state;       ///< predicted s[k]
    Tensor actual_state;          ///< actual s[k]
    TimeInfo time_info;           ///< Time information
    int personal_id;              ///< Personal identifier
    float reward;                 ///< Reward/feedback
    double timestamp;             ///< Timestamp
};

/**
 * @brief STA (Sense The Ambience) Transformer implementation
 */
class STATransformer {
public:
    /**
     * @brief Constructor
     * @param config STA configuration
     */
    explicit STATransformer(const STAConfig& config);
    
    /**
     * @brief Destructor
     */
    ~STATransformer();
    
    /**
     * @brief Predict future human state
     * @param observable_state Observable human state x[k-1]
     * @param controllable_input Environmental input u[k-1]
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Predicted future state s[k]
     */
    Tensor PredictState(const Tensor& observable_state,
                       const Tensor& controllable_input,
                       const TimeInfo* time_info = nullptr,
                       int personal_id = -1);
    
    /**
     * @brief Predict with uncertainty estimation
     * @param observable_state Observable human state
     * @param controllable_input Environmental input
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Pair of (predicted_state, uncertainty)
     */
    std::pair<Tensor, Tensor> PredictWithUncertainty(
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo* time_info = nullptr,
        int personal_id = -1);
    
    /**
     * @brief Compute sensitivity of predicted state to controllable input
     * @param observable_state Observable human state
     * @param controllable_input Environmental input
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Sensitivity matrix ∂s/∂u
     */
    Tensor ComputeSensitivity(const Tensor& observable_state,
                             const Tensor& controllable_input,
                             const TimeInfo* time_info = nullptr,
                             int personal_id = -1);
    
    /**
     * @brief Compute optimal control input based on sensitivity and meta-evaluation
     * @param observable_state Current observable state
     * @param current_input Current controllable input
     * @param meta_eval Meta-evaluation function
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Optimal control input
     */
    Tensor ComputeOptimalControl(const Tensor& observable_state,
                                const Tensor& current_input,
                                const MetaEvaluationFunction& meta_eval,
                                const TimeInfo* time_info = nullptr,
                                int personal_id = -1);
    
    /**
     * @brief Update model with new observation (real-time learning)
     * @param observable_state Observable state x[k-1]
     * @param controllable_input Control input u[k-1]
     * @param actual_state Actual observed state s[k]
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     */
    void UpdateModel(const Tensor& observable_state,
                    const Tensor& controllable_input,
                    const Tensor& actual_state,
                    const TimeInfo* time_info = nullptr,
                    int personal_id = -1);
    
    /**
     * @brief Perform one step of STA interaction
     * @param observable_state Current observable state
     * @param current_input Current control input
     * @param meta_eval Meta-evaluation function
     * @param time_info Time information (optional)
     * @param personal_id Personal ID (optional)
     * @return Next control input
     */
    Tensor STAStep(const Tensor& observable_state,
                  const Tensor& current_input,
                  const MetaEvaluationFunction& meta_eval,
                  const TimeInfo* time_info = nullptr,
                  int personal_id = -1);
    
    /**
     * @brief Add experience to buffer
     * @param experience Experience entry
     */
    void AddExperience(const ExperienceEntry& experience);
    
    /**
     * @brief Get prediction confidence
     * @param observable_state Observable state
     * @param controllable_input Control input
     * @return Confidence score [0, 1]
     */
    float GetPredictionConfidence(const Tensor& observable_state,
                                 const Tensor& controllable_input);
    
    /**
     * @brief Enable/disable streaming mode
     * @param enable Enable streaming
     */
    void SetStreamingMode(bool enable) { streaming_mode_ = enable; }
    
    /**
     * @brief Get model performance statistics
     * @return Performance statistics as string
     */
    std::string GetPerformanceStats() const;
    
    /**
     * @brief Reset model state
     */
    void Reset();
    
    /**
     * @brief Save model state
     * @param filepath File path
     * @return Success flag
     */
    bool SaveModel(const std::string& filepath) const;
    
    /**
     * @brief Load model state
     * @param filepath File path
     * @return Success flag
     */
    bool LoadModel(const std::string& filepath);
    
    /**
     * @brief Get configuration
     * @return Configuration
     */
    const STAConfig& GetConfig() const { return config_; }

private:
    STAConfig config_;
    
    // Core prediction model
    std::unique_ptr<Transformer> prediction_model_;
    
    // Ensemble for uncertainty estimation
    std::vector<std::unique_ptr<Transformer>> ensemble_models_;
    
    // Personal embeddings
    std::unordered_map<int, Tensor> personal_embeddings_;
    
    // Experience buffer
    std::vector<ExperienceEntry> experience_buffer_;
    size_t buffer_index_ = 0;
    
    // Performance tracking
    size_t prediction_count_ = 0;
    double total_prediction_time_ = 0.0;
    double total_update_time_ = 0.0;
    float cumulative_loss_ = 0.0f;
    
    // State
    bool streaming_mode_ = false;
    mutable std::vector<Tensor> gradient_cache_;
    
    // Private methods
    void InitializeModels();
    void InitializePersonalEmbeddings();
    Tensor CombineInputs(const Tensor& observable_state,
                        const Tensor& controllable_input,
                        int personal_id = -1) const;
    Tensor ExtractPredictedState(const Tensor& model_output) const;
    void UpdatePersonalEmbedding(int personal_id, const Tensor& gradient);
    Tensor ComputeNumericalGradient(const Tensor& observable_state,
                                   const Tensor& controllable_input,
                                   const TimeInfo* time_info,
                                   int personal_id) const;
    void BackpropagateGradients(const Tensor& prediction_error);
    float ComputePredictionLoss(const Tensor& predicted, const Tensor& actual) const;
    void UpdatePerformanceStats(double prediction_time, double update_time, float loss);
    bool ValidateInputDimensions(const Tensor& observable_state,
                               const Tensor& controllable_input) const;
};

/**
 * @brief STA Transformer Builder for easy construction
 */
class STABuilder {
public:
    STABuilder() = default;
    
    STABuilder& SetObservableStateDim(int dim) {
        config_.observable_state_dim = dim;
        return *this;
    }
    
    STABuilder& SetControllableInputDim(int dim) {
        config_.controllable_input_dim = dim;
        return *this;
    }
    
    STABuilder& SetPredictedStateDim(int dim) {
        config_.predicted_state_dim = dim;
        return *this;
    }
    
    STABuilder& SetLearningRate(float lr) {
        config_.learning_rate = lr;
        return *this;
    }
    
    STABuilder& SetControlGain(float gain) {
        config_.control_gain = gain;
        return *this;
    }
    
    STABuilder& EnableUncertainty(bool enable = true, int ensemble_size = 5) {
        config_.enable_uncertainty = enable;
        config_.ensemble_size = ensemble_size;
        return *this;
    }
    
    STABuilder& EnablePersonalAdaptation(bool enable = true, int max_persons = 100) {
        config_.enable_personal_adaptation = enable;
        config_.max_persons = max_persons;
        return *this;
    }
    
    STABuilder& SetTransformerConfig(const TransformerConfig& transformer_config) {
        config_.transformer_config = transformer_config;
        return *this;
    }
    
    STABuilder& SetMaxInferenceTime(float time_ms) {
        config_.max_inference_time_ms = time_ms;
        return *this;
    }
    
    STABuilder& SetBufferSize(int size) {
        config_.buffer_size = size;
        return *this;
    }
    
    std::unique_ptr<STATransformer> Build() {
        return std::make_unique<STATransformer>(config_);
    }
    
private:
    STAConfig config_;
};

} // namespace Core
} // namespace LwTT

#endif // LWTT_CORE_STA_TRANSFORMER_HPP
