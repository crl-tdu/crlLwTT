/**
 * @file ChainRuleSensitivity.hpp
 * @brief Chain rule implementation for complete sensitivity analysis
 * @version 1.0.0
 * @date 2025-06-02
 */

#ifndef LWTT_CORE_CHAIN_RULE_SENSITIVITY_HPP
#define LWTT_CORE_CHAIN_RULE_SENSITIVITY_HPP

#include "Tensor.hpp"
#include "AutogradSensitivity.hpp"
#include "ObjectiveFunction.hpp"
#include "STATransformer.hpp"
#include <functional>
#include <memory>
#include <chrono>

namespace crllwtt {
namespace Core {

/**
 * @brief Configuration for chain rule sensitivity analysis
 */
struct ChainRuleConfig {
    bool enable_caching = true;                ///< Enable gradient caching for performance
    bool enable_parallel_computation = false; ///< Enable parallel gradient computation
    float numerical_epsilon = 1e-6f;          ///< Epsilon for numerical differentiation fallback
    bool enable_gradient_verification = false; ///< Enable gradient verification against numerical
    int max_cache_size = 1000;               ///< Maximum number of cached gradients
    bool enable_performance_monitoring = true; ///< Enable performance monitoring
    float gradient_magnitude_threshold = 1e-8f; ///< Threshold for gradient magnitude check
};

/**
 * @brief Complete chain rule sensitivity analyzer
 * 
 * This class implements the complete chain rule computation:
 * ∂M(ŝ)/∂u = (∂M/∂ŝ) · (∂ŝ/∂u)
 * 
 * Where:
 * - M(ŝ) is the objective function
 * - ŝ is the predicted state from STATransformer
 * - u is the controllable input
 */
class ChainRuleSensitivity {
public:
    /**
     * @brief Constructor
     * @param autograd_sensitivity Autograd sensitivity analyzer
     * @param config Configuration
     */
    explicit ChainRuleSensitivity(
        std::shared_ptr<AutogradSensitivity> autograd_sensitivity,
        const ChainRuleConfig& config = ChainRuleConfig{}
    );

    /**
     * @brief Destructor
     */
    ~ChainRuleSensitivity();

    /**
     * @brief Compute complete sensitivity ∂M(ŝ)/∂u
     * @param objective_function Objective function M(ŝ)
     * @param sta_transformer STA transformer for prediction ŝ = f(x, u)
     * @param observable_state Current observable state x
     * @param controllable_input Current controllable input u
     * @param time_info Time information for prediction
     * @param uncertainty Optional uncertainty estimate
     * @return Complete sensitivity tensor ∂M(ŝ)/∂u
     */
    Tensor ComputeCompleteSensitivity(
        const ObjectiveFunction& objective_function,
        STATransformer& sta_transformer,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo& time_info,
        const Tensor* uncertainty = nullptr
    );

    /**
     * @brief Compute objective gradient ∂M/∂ŝ
     * @param objective_function Objective function
     * @param predicted_state Predicted state ŝ
     * @param uncertainty Optional uncertainty
     * @return Objective gradient ∂M/∂ŝ
     */
    Tensor ComputeObjectiveGradient(
        const ObjectiveFunction& objective_function,
        const Tensor& predicted_state,
        const Tensor* uncertainty = nullptr
    );

    /**
     * @brief Compute prediction gradient ∂ŝ/∂u using autograd
     * @param sta_transformer STA transformer
     * @param observable_state Observable state x
     * @param controllable_input Controllable input u
     * @param time_info Time information
     * @return Prediction gradient ∂ŝ/∂u
     */
    Tensor ComputePredictionGradient(
        STATransformer& sta_transformer,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo& time_info
    );

    /**
     * @brief Compute batch sensitivities for multiple inputs
     * @param objective_function Objective function
     * @param sta_transformer STA transformer
     * @param observable_states Vector of observable states
     * @param controllable_inputs Vector of controllable inputs
     * @param time_infos Vector of time information
     * @return Vector of sensitivity tensors
     */
    std::vector<Tensor> ComputeBatchSensitivities(
        const ObjectiveFunction& objective_function,
        STATransformer& sta_transformer,
        const std::vector<Tensor>& observable_states,
        const std::vector<Tensor>& controllable_inputs,
        const std::vector<TimeInfo>& time_infos
    );

    /**
     * @brief Verify gradient computation using numerical differentiation
     * @param objective_function Objective function
     * @param sta_transformer STA transformer
     * @param observable_state Observable state
     * @param controllable_input Controllable input
     * @param time_info Time information
     * @param epsilon Finite difference epsilon
     * @return True if analytical and numerical gradients match
     */
    bool VerifyGradients(
        const ObjectiveFunction& objective_function,
        STATransformer& sta_transformer,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo& time_info,
        float epsilon = 1e-5f
    );

    /**
     * @brief Get performance statistics
     * @return Performance statistics as string
     */
    std::string GetPerformanceStatistics() const;

    /**
     * @brief Clear gradient cache
     */
    void ClearCache();

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const ChainRuleConfig& GetConfig() const { return config_; }

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void UpdateConfig(const ChainRuleConfig& config);

private:
    ChainRuleConfig config_;
    std::shared_ptr<AutogradSensitivity> autograd_sensitivity_;
    
    // Performance monitoring
    mutable size_t total_computations_;
    mutable double total_computation_time_;
    mutable double total_objective_grad_time_;
    mutable double total_prediction_grad_time_;
    mutable double total_chain_rule_time_;
    
    // Caching system
    struct GradientCacheEntry {
        Tensor observable_state;
        Tensor controllable_input;
        TimeInfo time_info;
        Tensor objective_gradient;
        Tensor prediction_gradient;
        Tensor complete_sensitivity;
        std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    };
    
    mutable std::vector<GradientCacheEntry> gradient_cache_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
    
    // Private helper methods
    void UpdatePerformanceStats(double computation_time,
                               double objective_grad_time,
                               double prediction_grad_time,
                               double chain_rule_time) const;
    
    bool FindCachedGradient(const Tensor& observable_state,
                           const Tensor& controllable_input,
                           const TimeInfo& time_info,
                           GradientCacheEntry& entry) const;
    
    void CacheGradient(const Tensor& observable_state,
                      const Tensor& controllable_input,
                      const TimeInfo& time_info,
                      const Tensor& objective_gradient,
                      const Tensor& prediction_gradient,
                      const Tensor& complete_sensitivity) const;
    
    Tensor MultiplyGradients(const Tensor& objective_grad,
                            const Tensor& prediction_grad) const;
    
    bool TensorsEqual(const Tensor& a, const Tensor& b, float tolerance = 1e-6f) const;
    
    Tensor ComputeNumericalSensitivity(
        const ObjectiveFunction& objective_function,
        STATransformer& sta_transformer,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo& time_info,
        float epsilon
    ) const;
};

/**
 * @brief Configuration for real-time optimization
 */
struct RealtimeConfig {
    float target_execution_time_ms = 0.8f;  ///< Target execution time in milliseconds
    bool enable_approximate_gradients = true; ///< Enable gradient approximation for speed
    int max_approximation_order = 2;        ///< Maximum order for gradient approximation
    bool enable_adaptive_precision = true;   ///< Enable adaptive precision control
    float min_precision = 1e-4f;            ///< Minimum computational precision
    float max_precision = 1e-6f;            ///< Maximum computational precision
    bool enable_early_termination = true;    ///< Enable early termination for real-time
    int max_iterations = 5;                 ///< Maximum iterations for iterative methods
};

/**
 * @brief Real-time sensitivity optimizer
 * 
 * Optimizes sensitivity computation for real-time applications
 * with target execution time < 1ms.
 */
class RealtimeSensitivityOptimizer {
public:
    /**
     * @brief Constructor
     * @param chain_rule_sensitivity Chain rule sensitivity analyzer
     * @param config Real-time configuration
     */
    explicit RealtimeSensitivityOptimizer(
        std::shared_ptr<ChainRuleSensitivity> chain_rule_sensitivity,
        const RealtimeConfig& config = {}
    );

    /**
     * @brief Compute optimized sensitivity for real-time applications
     * @param objective_function Objective function
     * @param sta_transformer STA transformer
     * @param observable_state Observable state
     * @param controllable_input Controllable input
     * @param time_info Time information
     * @return Optimized sensitivity tensor
     */
    Tensor ComputeOptimizedSensitivity(
        const ObjectiveFunction& objective_function,
        STATransformer& sta_transformer,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const TimeInfo& time_info
    );

    /**
     * @brief Get optimization statistics
     * @return Statistics including average execution time
     */
    std::string GetOptimizationStatistics() const;

    /**
     * @brief Check if real-time constraints are met
     * @return True if execution time is within target
     */
    bool IsRealtimeCompliant() const;

private:
    std::shared_ptr<ChainRuleSensitivity> chain_rule_sensitivity_;
    RealtimeConfig config_;
    
    mutable double average_execution_time_;
    mutable size_t total_computations_;
    mutable size_t successful_realtime_computations_;
    
    void UpdateExecutionStats(double execution_time) const;
    float AdaptPrecision(double current_execution_time) const;
};

/**
 * @brief Utility functions for chain rule sensitivity analysis
 */
namespace ChainRuleUtils {

    /**
     * @brief Create optimized chain rule sensitivity analyzer
     * @param state_dim State dimension
     * @param input_dim Input dimension
     * @return Configured chain rule sensitivity analyzer
     */
    std::shared_ptr<ChainRuleSensitivity> CreateOptimizedAnalyzer(
        [[maybe_unused]] int state_dim, [[maybe_unused]] int input_dim);

    /**
     * @brief Benchmark chain rule computation performance
     * @param analyzer Chain rule analyzer
     * @param num_tests Number of benchmark tests
     * @return Performance benchmark results
     */
    std::string BenchmarkChainRule(
        [[maybe_unused]] ChainRuleSensitivity& analyzer,
        [[maybe_unused]] int num_tests = 100);

    /**
     * @brief Validate chain rule implementation
     * @param analyzer Chain rule analyzer
     * @param tolerance Validation tolerance
     * @return True if implementation is correct
     */
    bool ValidateChainRuleImplementation(
        [[maybe_unused]] ChainRuleSensitivity& analyzer,
        [[maybe_unused]] float tolerance = 1e-4f);

    /**
     * @brief Profile memory usage of chain rule computation
     * @param analyzer Chain rule analyzer
     * @return Memory usage statistics
     */
    std::string ProfileMemoryUsage(
        [[maybe_unused]] const ChainRuleSensitivity& analyzer);

} // namespace ChainRuleUtils

} // namespace Core
} // namespace crllwtt

#endif // LWTT_CORE_CHAIN_RULE_SENSITIVITY_HPP
