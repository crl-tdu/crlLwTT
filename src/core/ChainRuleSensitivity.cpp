/**
 * @file ChainRuleSensitivity.cpp
 * @brief Implementation of chain rule sensitivity analysis
 * @version 1.0.0
 * @date 2025-06-02
 */

#include "../../include/LwTT/core/ChainRuleSensitivity.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace crllwtt {
namespace Core {

ChainRuleSensitivity::ChainRuleSensitivity(
    std::shared_ptr<AutogradSensitivity> autograd_sensitivity,
    const ChainRuleConfig& config)
    : config_(config), autograd_sensitivity_(autograd_sensitivity),
      total_computations_(0), total_computation_time_(0.0),
      total_objective_grad_time_(0.0), total_prediction_grad_time_(0.0),
      total_chain_rule_time_(0.0), cache_hits_(0), cache_misses_(0) {
    
    if (!autograd_sensitivity_) {
        throw std::invalid_argument("AutogradSensitivity cannot be null");
    }
}

ChainRuleSensitivity::~ChainRuleSensitivity() {
    ClearCache();
}

Tensor ChainRuleSensitivity::ComputeCompleteSensitivity(
    const ObjectiveFunction& objective_function,
    STATransformer& sta_transformer,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo& time_info,
    const Tensor* uncertainty) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Check cache first
    if (config_.enable_caching) {
        GradientCacheEntry cached_entry;
        if (FindCachedGradient(observable_state, controllable_input, time_info, cached_entry)) {
            cache_hits_++;
            return cached_entry.complete_sensitivity;
        }
        cache_misses_++;
    }
    
    // Step 1: Forward pass to get predicted state ŝ
    Tensor predicted_state = sta_transformer.PredictState(observable_state, controllable_input, &time_info);
    
    // Step 2: Compute objective gradient ∂M/∂ŝ
    auto obj_grad_start = std::chrono::high_resolution_clock::now();
    Tensor objective_gradient = ComputeObjectiveGradient(objective_function, predicted_state, uncertainty);
    auto obj_grad_end = std::chrono::high_resolution_clock::now();
    
    // Step 3: Compute prediction gradient ∂ŝ/∂u
    auto pred_grad_start = std::chrono::high_resolution_clock::now();
    Tensor prediction_gradient = ComputePredictionGradient(sta_transformer, observable_state, controllable_input, time_info);
    auto pred_grad_end = std::chrono::high_resolution_clock::now();
    
    // Step 4: Apply chain rule: ∂M(ŝ)/∂u = (∂M/∂ŝ) · (∂ŝ/∂u)
    auto chain_rule_start = std::chrono::high_resolution_clock::now();
    Tensor complete_sensitivity = MultiplyGradients(objective_gradient, prediction_gradient);
    auto chain_rule_end = std::chrono::high_resolution_clock::now();
    
    // Update performance statistics
    double obj_grad_time = std::chrono::duration<double, std::milli>(obj_grad_end - obj_grad_start).count();
    double pred_grad_time = std::chrono::duration<double, std::milli>(pred_grad_end - pred_grad_start).count();
    double chain_rule_time = std::chrono::duration<double, std::milli>(chain_rule_end - chain_rule_start).count();
    double total_time = std::chrono::duration<double, std::milli>(chain_rule_end - start_time).count();
    
    UpdatePerformanceStats(total_time, obj_grad_time, pred_grad_time, chain_rule_time);
    
    // Cache result
    if (config_.enable_caching) {
        CacheGradient(observable_state, controllable_input, time_info, 
                     objective_gradient, prediction_gradient, complete_sensitivity);
    }
    
    return complete_sensitivity;
}

Tensor ChainRuleSensitivity::ComputeObjectiveGradient(
    const ObjectiveFunction& objective_function,
    const Tensor& predicted_state,
    const Tensor* uncertainty) {
    
    return objective_function.ComputeGradient(predicted_state, uncertainty);
}

Tensor ChainRuleSensitivity::ComputePredictionGradient(
    STATransformer& sta_transformer,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo& time_info) {
    
    // Create prediction function for autograd
    auto prediction_function = [&sta_transformer, &time_info](const Tensor& x, const Tensor& u) -> Tensor {
        return sta_transformer.PredictState(x, u, &time_info);
    };
    
    // Use autograd to compute ∂ŝ/∂u
    return autograd_sensitivity_->ComputeSensitivity(
        prediction_function, observable_state, controllable_input);
}

std::vector<Tensor> ChainRuleSensitivity::ComputeBatchSensitivities(
    const ObjectiveFunction& objective_function,
    STATransformer& sta_transformer,
    const std::vector<Tensor>& observable_states,
    const std::vector<Tensor>& controllable_inputs,
    const std::vector<TimeInfo>& time_infos) {
    
    if (observable_states.size() != controllable_inputs.size() || 
        observable_states.size() != time_infos.size()) {
        throw std::invalid_argument("Batch input sizes must match");
    }
    
    std::vector<Tensor> sensitivities;
    sensitivities.reserve(observable_states.size());
    
    for (size_t i = 0; i < observable_states.size(); ++i) {
        Tensor sensitivity = ComputeCompleteSensitivity(
            objective_function, sta_transformer,
            observable_states[i], controllable_inputs[i], time_infos[i]);
        sensitivities.push_back(sensitivity);
    }
    
    return sensitivities;
}

bool ChainRuleSensitivity::VerifyGradients(
    const ObjectiveFunction& objective_function,
    STATransformer& sta_transformer,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo& time_info,
    float epsilon) {
    
    // Compute analytical gradient
    Tensor analytical_grad = ComputeCompleteSensitivity(
        objective_function, sta_transformer, observable_state, controllable_input, time_info);
    
    // Compute numerical gradient
    Tensor numerical_grad = ComputeNumericalSensitivity(
        objective_function, sta_transformer, observable_state, controllable_input, time_info, epsilon);
    
    // Compare gradients
    float tolerance = 1e-3f;
    for (int i = 0; i < analytical_grad.GetSize(); ++i) {
        float diff = std::abs(analytical_grad.GetData()[i] - numerical_grad.GetData()[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    
    return true;
}

std::string ChainRuleSensitivity::GetPerformanceStatistics() const {
    std::string stats = "Chain Rule Sensitivity Performance Statistics:\n";
    stats += "  Total computations: " + std::to_string(total_computations_) + "\n";
    
    if (total_computations_ > 0) {
        double avg_total = total_computation_time_ / total_computations_;
        double avg_obj_grad = total_objective_grad_time_ / total_computations_;
        double avg_pred_grad = total_prediction_grad_time_ / total_computations_;
        double avg_chain_rule = total_chain_rule_time_ / total_computations_;
        
        stats += "  Average total time: " + std::to_string(avg_total) + " ms\n";
        stats += "  Average objective gradient time: " + std::to_string(avg_obj_grad) + " ms\n";
        stats += "  Average prediction gradient time: " + std::to_string(avg_pred_grad) + " ms\n";
        stats += "  Average chain rule time: " + std::to_string(avg_chain_rule) + " ms\n";
        stats += "  Throughput: " + std::to_string(1000.0 / avg_total) + " ops/sec\n";
    }
    
    if (config_.enable_caching) {
        stats += "  Cache hits: " + std::to_string(cache_hits_) + "\n";
        stats += "  Cache misses: " + std::to_string(cache_misses_) + "\n";
        if (cache_hits_ + cache_misses_ > 0) {
            float cache_hit_rate = static_cast<float>(cache_hits_) / (cache_hits_ + cache_misses_);
            stats += "  Cache hit rate: " + std::to_string(cache_hit_rate * 100.0f) + "%\n";
        }
        stats += "  Cache size: " + std::to_string(gradient_cache_.size()) + "\n";
    }
    
    return stats;
}

void ChainRuleSensitivity::ClearCache() {
    gradient_cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

void ChainRuleSensitivity::UpdateConfig(const ChainRuleConfig& config) {
    config_ = config;
    if (!config_.enable_caching) {
        ClearCache();
    }
}

void ChainRuleSensitivity::UpdatePerformanceStats(double computation_time,
                                                 double objective_grad_time,
                                                 double prediction_grad_time,
                                                 double chain_rule_time) const {
    total_computations_++;
    total_computation_time_ += computation_time;
    total_objective_grad_time_ += objective_grad_time;
    total_prediction_grad_time_ += prediction_grad_time;
    total_chain_rule_time_ += chain_rule_time;
}

bool ChainRuleSensitivity::FindCachedGradient(const Tensor& observable_state,
                                             const Tensor& controllable_input,
                                             const TimeInfo& time_info,
                                             GradientCacheEntry& entry) const {
    for (const auto& cached : gradient_cache_) {
        if (TensorsEqual(cached.observable_state, observable_state) &&
            TensorsEqual(cached.controllable_input, controllable_input) &&
            cached.time_info.timestamps.size() == time_info.timestamps.size()) {
            
            // Compare timestamps
            bool time_match = true;
            for (size_t i = 0; i < time_info.timestamps.size(); ++i) {
                if (std::abs(cached.time_info.timestamps[i] - time_info.timestamps[i]) > 1e-6f) {
                    time_match = false;
                    break;
                }
            }
            
            if (time_match) {
                entry = cached;
                return true;
            }
        }
    }
    return false;
}

void ChainRuleSensitivity::CacheGradient(const Tensor& observable_state,
                                        const Tensor& controllable_input,
                                        const TimeInfo& time_info,
                                        const Tensor& objective_gradient,
                                        const Tensor& prediction_gradient,
                                        const Tensor& complete_sensitivity) const {
    if (gradient_cache_.size() >= static_cast<size_t>(config_.max_cache_size)) {
        // Remove oldest entry
        gradient_cache_.erase(gradient_cache_.begin());
    }
    
    GradientCacheEntry entry;
    entry.observable_state = observable_state;
    entry.controllable_input = controllable_input;
    entry.time_info = time_info;
    entry.objective_gradient = objective_gradient;
    entry.prediction_gradient = prediction_gradient;
    entry.complete_sensitivity = complete_sensitivity;
    entry.timestamp = std::chrono::high_resolution_clock::now();
    
    gradient_cache_.push_back(entry);
}

Tensor ChainRuleSensitivity::MultiplyGradients(const Tensor& objective_grad,
                                              const Tensor& prediction_grad) const {
    // For vector-vector case: element-wise multiplication
    if (objective_grad.GetSize() == prediction_grad.GetSize()) {
        Tensor result({objective_grad.GetSize()});
        for (int i = 0; i < objective_grad.GetSize(); ++i) {
            result.GetData()[i] = objective_grad.GetData()[i] * prediction_grad.GetData()[i];
        }
        return result;
    }
    
    // For matrix-vector case: matrix-vector multiplication
    auto obj_shape = objective_grad.Shape();
    auto pred_shape = prediction_grad.Shape();
    
    if (obj_shape.size() == 1 && pred_shape.size() == 2) {
        // objective_grad: [m], prediction_grad: [m x n] -> result: [n]
        int m = obj_shape[0];
        int n = pred_shape[1];
        
        if (m != pred_shape[0]) {
            throw std::invalid_argument("Gradient dimensions do not match for chain rule");
        }
        
        Tensor result({n});
        result.Fill(0.0f);
        
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                result.GetData()[j] += objective_grad.GetData()[i] * prediction_grad.GetData()[i * n + j];
            }
        }
        
        return result;
    }
    
    throw std::invalid_argument("Unsupported gradient shapes for chain rule multiplication");
}

bool ChainRuleSensitivity::TensorsEqual(const Tensor& a, const Tensor& b, float tolerance) const {
    if (a.GetSize() != b.GetSize()) {
        return false;
    }
    
    for (int i = 0; i < a.GetSize(); ++i) {
        if (std::abs(a.GetData()[i] - b.GetData()[i]) > tolerance) {
            return false;
        }
    }
    
    return true;
}

Tensor ChainRuleSensitivity::ComputeNumericalSensitivity(
    const ObjectiveFunction& objective_function,
    STATransformer& sta_transformer,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo& time_info,
    float epsilon) const {
    
    Tensor numerical_grad({controllable_input.GetSize()});
    
    // Create composite function: M(f(x, u))
    auto composite_function = [&](const Tensor& u) -> float {
        Tensor predicted = sta_transformer.PredictState(observable_state, u, &time_info);
        return objective_function.Evaluate(predicted);
    };
    
    // Compute numerical gradient using finite differences
    for (int i = 0; i < controllable_input.GetSize(); ++i) {
        Tensor u_plus = controllable_input;
        Tensor u_minus = controllable_input;
        
        u_plus.GetData()[i] += epsilon;
        u_minus.GetData()[i] -= epsilon;
        
        float value_plus = composite_function(u_plus);
        float value_minus = composite_function(u_minus);
        
        numerical_grad.GetData()[i] = (value_plus - value_minus) / (2.0f * epsilon);
    }
    
    return numerical_grad;
}

// RealtimeSensitivityOptimizer implementation
RealtimeSensitivityOptimizer::RealtimeSensitivityOptimizer(
    std::shared_ptr<ChainRuleSensitivity> chain_rule_sensitivity,
    const RealtimeConfig& config)
    : chain_rule_sensitivity_(chain_rule_sensitivity), config_(config),
      average_execution_time_(0.0), total_computations_(0),
      successful_realtime_computations_(0) {
    
    if (!chain_rule_sensitivity_) {
        throw std::invalid_argument("ChainRuleSensitivity cannot be null");
    }
}

Tensor RealtimeSensitivityOptimizer::ComputeOptimizedSensitivity(
    const ObjectiveFunction& objective_function,
    STATransformer& sta_transformer,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo& time_info) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Compute sensitivity
    Tensor sensitivity = chain_rule_sensitivity_->ComputeCompleteSensitivity(
        objective_function, sta_transformer, observable_state, controllable_input, time_info);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    UpdateExecutionStats(execution_time);
    
    if (execution_time <= config_.target_execution_time_ms) {
        successful_realtime_computations_++;
    }
    
    return sensitivity;
}

std::string RealtimeSensitivityOptimizer::GetOptimizationStatistics() const {
    std::string stats = "Real-time Sensitivity Optimizer Statistics:\n";
    stats += "  Target execution time: " + std::to_string(config_.target_execution_time_ms) + " ms\n";
    stats += "  Average execution time: " + std::to_string(average_execution_time_) + " ms\n";
    stats += "  Total computations: " + std::to_string(total_computations_) + "\n";
    stats += "  Successful real-time computations: " + std::to_string(successful_realtime_computations_) + "\n";
    
    if (total_computations_ > 0) {
        float success_rate = static_cast<float>(successful_realtime_computations_) / total_computations_;
        stats += "  Real-time success rate: " + std::to_string(success_rate * 100.0f) + "%\n";
    }
    
    return stats;
}

bool RealtimeSensitivityOptimizer::IsRealtimeCompliant() const {
    return average_execution_time_ <= config_.target_execution_time_ms;
}

void RealtimeSensitivityOptimizer::UpdateExecutionStats(double execution_time) const {
    total_computations_++;
    average_execution_time_ = (average_execution_time_ * (total_computations_ - 1) + execution_time) / total_computations_;
}

float RealtimeSensitivityOptimizer::AdaptPrecision(double current_execution_time) const {
    if (!config_.enable_adaptive_precision) {
        return config_.max_precision;
    }
    
    float time_ratio = static_cast<float>(current_execution_time / config_.target_execution_time_ms);
    
    if (time_ratio > 1.0f) {
        // Running too slow, reduce precision
        return config_.min_precision;
    } else {
        // Running fast enough, can use higher precision
        return config_.max_precision;
    }
}

// ChainRuleUtils implementation
namespace ChainRuleUtils {

std::shared_ptr<ChainRuleSensitivity> CreateOptimizedAnalyzer([[maybe_unused]] int state_dim, [[maybe_unused]] int input_dim) {
    // Create optimized autograd sensitivity
    AutogradConfig autograd_config;
    autograd_config.enable_autograd = true;
    autograd_config.gradient_clip_threshold = 1.0f;
    autograd_config.enable_gradient_checkpointing = false;
    
    auto autograd_sensitivity = std::make_shared<AutogradSensitivity>(autograd_config);
    
    // Create optimized chain rule config
    ChainRuleConfig chain_rule_config;
    chain_rule_config.enable_caching = true;
    chain_rule_config.enable_performance_monitoring = true;
    chain_rule_config.max_cache_size = 500;
    chain_rule_config.numerical_epsilon = 1e-6f;
    
    return std::make_shared<ChainRuleSensitivity>(autograd_sensitivity, chain_rule_config);
}

std::string BenchmarkChainRule([[maybe_unused]] ChainRuleSensitivity& analyzer, [[maybe_unused]] int num_tests) {
    // Implementation would include comprehensive benchmarking
    std::string result = "Chain Rule Benchmark Results:\n";
    result += "  Number of tests: " + std::to_string(num_tests) + "\n";
    result += "  Benchmark implementation needed\n";
    return result;
}

bool ValidateChainRuleImplementation([[maybe_unused]] ChainRuleSensitivity& analyzer, [[maybe_unused]] float tolerance) {
    // Implementation would include validation tests
    return true; // Placeholder
}

std::string ProfileMemoryUsage([[maybe_unused]] const ChainRuleSensitivity& analyzer) {
    std::string result = "Memory Usage Profile:\n";
    result += "  Memory profiling implementation needed\n";
    return result;
}

} // namespace ChainRuleUtils

} // namespace Core
} // namespace crllwtt
