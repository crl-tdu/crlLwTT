/**
 * @file ObjectiveFunction.cpp
 * @brief Implementation of objective function interface
 * @version 1.0.0
 * @date 2025-06-02
 */

#include "../../include/LwTT/core/ObjectiveFunction.hpp"
#include <stdexcept>
#include <cmath>
#include <chrono>

namespace crllwtt {
namespace Core {

// ObjectiveFunction base class implementation
ObjectiveFunction::ObjectiveFunction(const ObjectiveFunctionConfig& config)
    : config_(config) {
}

float ObjectiveFunction::ApplyUncertaintyPenalty(float base_value, const Tensor* uncertainty) const {
    if (!config_.enable_uncertainty_penalty || !uncertainty) {
        return base_value;
    }
    
    // Compute average uncertainty
    float avg_uncertainty = 0.0f;
    for (int i = 0; i < uncertainty->GetSize(); ++i) {
        avg_uncertainty += uncertainty->GetData()[i];
    }
    avg_uncertainty /= uncertainty->GetSize();
    
    // Apply penalty: higher uncertainty reduces objective value
    return base_value - config_.uncertainty_weight * avg_uncertainty;
}

float ObjectiveFunction::ApplySmoothnessePenalty(float base_value, 
                                               const Tensor& predicted_state,
                                               const Tensor* previous_state) const {
    if (!config_.enable_smoothness_penalty || !previous_state) {
        return base_value;
    }
    
    if (predicted_state.GetSize() != previous_state->GetSize()) {
        return base_value; // Cannot compute smoothness penalty
    }
    
    // Compute squared difference (roughness)
    float roughness = 0.0f;
    for (int i = 0; i < predicted_state.GetSize(); ++i) {
        float diff = predicted_state.GetData()[i] - previous_state->GetData()[i];
        roughness += diff * diff;
    }
    
    // Apply penalty: higher roughness reduces objective value
    return base_value - config_.smoothness_weight * roughness;
}

float ObjectiveFunction::ApplyBoundsPenalty(float base_value, const Tensor& predicted_state) const {
    if (!config_.enable_bounds_penalty) {
        return base_value;
    }
    
    float penalty = 0.0f;
    int state_size = predicted_state.GetSize();
    
    // Check lower bounds
    if (!config_.lower_bounds.empty()) {
        int bounds_size = std::min(state_size, static_cast<int>(config_.lower_bounds.size()));
        for (int i = 0; i < bounds_size; ++i) {
            float violation = config_.lower_bounds[i] - predicted_state.GetData()[i];
            if (violation > 0.0f) {
                penalty += violation * violation;
            }
        }
    }
    
    // Check upper bounds
    if (!config_.upper_bounds.empty()) {
        int bounds_size = std::min(state_size, static_cast<int>(config_.upper_bounds.size()));
        for (int i = 0; i < bounds_size; ++i) {
            float violation = predicted_state.GetData()[i] - config_.upper_bounds[i];
            if (violation > 0.0f) {
                penalty += violation * violation;
            }
        }
    }
    
    return base_value - config_.bounds_weight * penalty;
}

// TargetStateObjective implementation
TargetStateObjective::TargetStateObjective(const Tensor& target_state,
                                         const ObjectiveFunctionConfig& config)
    : ObjectiveFunction(config), target_state_(target_state) {
}

float TargetStateObjective::Evaluate(const Tensor& predicted_state,
                                    const Tensor* uncertainty,
                                    [[maybe_unused]] const void* context) const {
    if (predicted_state.GetSize() != target_state_.GetSize()) {
        throw std::invalid_argument("Predicted state and target state dimensions must match");
    }
    
    // Compute negative squared distance to target (higher is better)
    float distance_sq = 0.0f;
    for (int i = 0; i < predicted_state.GetSize(); ++i) {
        float diff = predicted_state.GetData()[i] - target_state_.GetData()[i];
        distance_sq += diff * diff;
    }
    
    float base_value = -distance_sq * config_.weight;
    
    // Apply penalties
    base_value = ApplyUncertaintyPenalty(base_value, uncertainty);
    base_value = ApplyBoundsPenalty(base_value, predicted_state);
    
    return base_value;
}

Tensor TargetStateObjective::ComputeGradient(const Tensor& predicted_state,
                                           [[maybe_unused]] const Tensor* uncertainty,
                                           [[maybe_unused]] const void* context) const {
    if (predicted_state.GetSize() != target_state_.GetSize()) {
        throw std::invalid_argument("Predicted state and target state dimensions must match");
    }
    
    // Gradient of -||s - s*||^2 w.r.t. s is -2 * weight * (s - s*)
    Tensor gradient({predicted_state.GetSize()});
    for (int i = 0; i < predicted_state.GetSize(); ++i) {
        float diff = predicted_state.GetData()[i] - target_state_.GetData()[i];
        gradient.GetData()[i] = -2.0f * config_.weight * diff;
    }
    
    // Add bounds penalty gradients
    if (config_.enable_bounds_penalty) {
        // Lower bounds penalty gradient
        if (!config_.lower_bounds.empty()) {
            int bounds_size = std::min(predicted_state.GetSize(), 
                                     static_cast<int>(config_.lower_bounds.size()));
            for (int i = 0; i < bounds_size; ++i) {
                float violation = config_.lower_bounds[i] - predicted_state.GetData()[i];
                if (violation > 0.0f) {
                    gradient.GetData()[i] -= 2.0f * config_.bounds_weight * violation;
                }
            }
        }
        
        // Upper bounds penalty gradient
        if (!config_.upper_bounds.empty()) {
            int bounds_size = std::min(predicted_state.GetSize(), 
                                     static_cast<int>(config_.upper_bounds.size()));
            for (int i = 0; i < bounds_size; ++i) {
                float violation = predicted_state.GetData()[i] - config_.upper_bounds[i];
                if (violation > 0.0f) {
                    gradient.GetData()[i] -= 2.0f * config_.bounds_weight * violation;
                }
            }
        }
    }
    
    return gradient;
}

std::string TargetStateObjective::GetDescription() const {
    return "Target State Objective: Minimize distance to target state";
}

void TargetStateObjective::SetTargetState(const Tensor& new_target) {
    target_state_ = new_target;
}

// ObjectiveUtils implementation
namespace ObjectiveUtils {

std::shared_ptr<ObjectiveFunction> CreateTargetObjective(const Tensor& target_state, float weight) {
    ObjectiveFunctionConfig config;
    config.weight = weight;
    return std::make_shared<TargetStateObjective>(target_state, config);
}

bool ValidateObjectiveGradients(const ObjectiveFunction& objective,
                               const Tensor& test_state,
                               float epsilon) {
    // Compute analytical gradient
    Tensor analytical_grad = objective.ComputeGradient(test_state);
    
    // Compute numerical gradient
    Tensor numerical_grad({test_state.GetSize()});
    
    for (int i = 0; i < test_state.GetSize(); ++i) {
        Tensor state_plus = test_state;
        Tensor state_minus = test_state;
        
        state_plus.GetData()[i] += epsilon;
        state_minus.GetData()[i] -= epsilon;
        
        float value_plus = objective.Evaluate(state_plus);
        float value_minus = objective.Evaluate(state_minus);
        
        numerical_grad.GetData()[i] = (value_plus - value_minus) / (2.0f * epsilon);
    }
    
    // Compare gradients
    float tolerance = 1e-4f;
    for (int i = 0; i < test_state.GetSize(); ++i) {
        float diff = std::abs(analytical_grad.GetData()[i] - numerical_grad.GetData()[i]);
        if (diff > tolerance) {
            return false;
        }
    }
    
    return true;
}

std::string BenchmarkObjective(const ObjectiveFunction& objective,
                              const std::vector<Tensor>& test_states) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double total_eval_time = 0.0;
    double total_grad_time = 0.0;
    
    for (const auto& state : test_states) {
        // Benchmark evaluation
        auto eval_start = std::chrono::high_resolution_clock::now();
        [[maybe_unused]] float value = objective.Evaluate(state);
        auto eval_end = std::chrono::high_resolution_clock::now();
        
        total_eval_time += std::chrono::duration<double, std::milli>(
            eval_end - eval_start).count();
        
        // Benchmark gradient computation
        auto grad_start = std::chrono::high_resolution_clock::now();
        Tensor gradient = objective.ComputeGradient(state);
        auto grad_end = std::chrono::high_resolution_clock::now();
        
        total_grad_time += std::chrono::duration<double, std::milli>(
            grad_end - grad_start).count();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    std::string result = "Objective Function Benchmark:\n";
    result += "  Test states: " + std::to_string(test_states.size()) + "\n";
    result += "  Total time: " + std::to_string(total_time) + " ms\n";
    result += "  Average evaluation time: " + std::to_string(total_eval_time / test_states.size()) + " ms\n";
    result += "  Average gradient time: " + std::to_string(total_grad_time / test_states.size()) + " ms\n";
    result += "  Throughput: " + std::to_string(1000.0 * test_states.size() / total_time) + " ops/sec\n";
    
    return result;
}

} // namespace ObjectiveUtils

} // namespace Core
} // namespace crllwtt
