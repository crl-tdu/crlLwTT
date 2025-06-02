/**
 * @file ObjectiveFunction.hpp
 * @brief Objective function interface for STA sensitivity analysis
 * @version 1.0.0
 * @date 2025-06-02
 */

#ifndef LWTT_CORE_OBJECTIVE_FUNCTION_HPP
#define LWTT_CORE_OBJECTIVE_FUNCTION_HPP

#include "Tensor.hpp"
#include "AutogradSensitivity.hpp"
#include <functional>
#include <memory>
#include <vector>
#include <string>

namespace crllwtt {
namespace Core {

/**
 * @brief Configuration for objective functions
 */
struct ObjectiveFunctionConfig {
    float weight = 1.0f;                      ///< Objective function weight
    bool enable_uncertainty_penalty = false;  ///< Enable uncertainty penalty
    float uncertainty_weight = 0.1f;          ///< Weight for uncertainty penalty
    bool enable_smoothness_penalty = false;   ///< Enable smoothness penalty
    float smoothness_weight = 0.01f;          ///< Weight for smoothness penalty
    bool enable_bounds_penalty = false;       ///< Enable bounds penalty
    float bounds_weight = 1.0f;               ///< Weight for bounds penalty
    std::vector<float> lower_bounds;          ///< Lower bounds for predicted states
    std::vector<float> upper_bounds;          ///< Upper bounds for predicted states
};

/**
 * @brief Abstract base class for objective functions M(ŝ)
 * 
 * This class provides the interface for defining objective functions
 * that evaluate the desirability of predicted future states ŝ.
 */
class ObjectiveFunction {
public:
    /**
     * @brief Constructor
     * @param config Configuration for the objective function
     */
    explicit ObjectiveFunction(const ObjectiveFunctionConfig& config = ObjectiveFunctionConfig{});

    /**
     * @brief Virtual destructor
     */
    virtual ~ObjectiveFunction() = default;

    /**
     * @brief Evaluate the objective function M(ŝ)
     * @param predicted_state Predicted future state ŝ
     * @param uncertainty Optional uncertainty estimate
     * @param context Optional context information
     * @return Objective function value (higher is better)
     */
    virtual float Evaluate(const Tensor& predicted_state,
                          const Tensor* uncertainty = nullptr,
                          const void* context = nullptr) const = 0;

    /**
     * @brief Compute gradient of objective function ∂M(ŝ)/∂ŝ
     * @param predicted_state Predicted future state ŝ
     * @param uncertainty Optional uncertainty estimate
     * @param context Optional context information
     * @return Gradient tensor ∂M(ŝ)/∂ŝ
     */
    virtual Tensor ComputeGradient(const Tensor& predicted_state,
                                  const Tensor* uncertainty = nullptr,
                                  const void* context = nullptr) const = 0;

    /**
     * @brief Update objective function based on feedback
     * @param predicted_state Predicted state that was evaluated
     * @param actual_outcome Actual observed outcome
     * @param feedback Feedback signal (reward/cost)
     */
    virtual void UpdateFromFeedback([[maybe_unused]] const Tensor& predicted_state,
                                   [[maybe_unused]] const Tensor& actual_outcome,
                                   [[maybe_unused]] float feedback) {}

    /**
     * @brief Get human-readable description of the objective
     * @return Description string
     */
    virtual std::string GetDescription() const = 0;

    /**
     * @brief Set weight for this objective function
     * @param weight New weight value
     */
    void SetWeight(float weight) { config_.weight = weight; }

    /**
     * @brief Get current weight
     * @return Current weight
     */
    float GetWeight() const { return config_.weight; }

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const ObjectiveFunctionConfig& GetConfig() const { return config_; }

protected:
    ObjectiveFunctionConfig config_;

    /**
     * @brief Apply uncertainty penalty to objective value
     * @param base_value Base objective value
     * @param uncertainty Uncertainty tensor
     * @return Penalized objective value
     */
    float ApplyUncertaintyPenalty(float base_value, const Tensor* uncertainty) const;

    /**
     * @brief Apply smoothness penalty to objective value
     * @param base_value Base objective value
     * @param predicted_state Predicted state
     * @param previous_state Previous predicted state (if available)
     * @return Penalized objective value
     */
    float ApplySmoothnessePenalty(float base_value, 
                                 const Tensor& predicted_state,
                                 const Tensor* previous_state = nullptr) const;

    /**
     * @brief Apply bounds penalty to objective value
     * @param base_value Base objective value
     * @param predicted_state Predicted state
     * @return Penalized objective value
     */
    float ApplyBoundsPenalty(float base_value, const Tensor& predicted_state) const;
};

/**
 * @brief Target state objective: minimize distance to target state
 */
class TargetStateObjective : public ObjectiveFunction {
public:
    /**
     * @brief Constructor
     * @param target_state Target state to reach
     * @param config Configuration
     */
    explicit TargetStateObjective(const Tensor& target_state,
                                 const ObjectiveFunctionConfig& config = ObjectiveFunctionConfig{});

    float Evaluate(const Tensor& predicted_state,
                   const Tensor* uncertainty = nullptr,
                   const void* context = nullptr) const override;

    Tensor ComputeGradient(const Tensor& predicted_state,
                          const Tensor* uncertainty = nullptr,
                          const void* context = nullptr) const override;

    std::string GetDescription() const override;

    /**
     * @brief Update target state
     * @param new_target New target state
     */
    void SetTargetState(const Tensor& new_target);

    /**
     * @brief Get current target state
     * @return Current target state
     */
    const Tensor& GetTargetState() const { return target_state_; }

private:
    Tensor target_state_;
};

/**
 * @brief Quadratic cost objective: minimize quadratic cost J = ŝᵀQŝ + rᵀŝ
 */
class QuadraticCostObjective : public ObjectiveFunction {
public:
    /**
     * @brief Constructor
     * @param Q_matrix Quadratic cost matrix
     * @param r_vector Linear cost vector
     * @param config Configuration
     */
    explicit QuadraticCostObjective(const Tensor& Q_matrix,
                                   const Tensor& r_vector,
                                   const ObjectiveFunctionConfig& config = ObjectiveFunctionConfig{});

    float Evaluate(const Tensor& predicted_state,
                   const Tensor* uncertainty = nullptr,
                   const void* context = nullptr) const override;

    Tensor ComputeGradient(const Tensor& predicted_state,
                          const Tensor* uncertainty = nullptr,
                          const void* context = nullptr) const override;

    std::string GetDescription() const override;

private:
    Tensor Q_matrix_;  ///< Quadratic cost matrix [n x n]
    Tensor r_vector_;  ///< Linear cost vector [n]
};

/**
 * @brief Multi-objective function: combination of multiple objectives
 */
class MultiObjectiveFunction : public ObjectiveFunction {
public:
    /**
     * @brief Constructor
     * @param config Configuration
     */
    explicit MultiObjectiveFunction(const ObjectiveFunctionConfig& config = ObjectiveFunctionConfig{});

    /**
     * @brief Add objective function with weight
     * @param objective Objective function to add
     * @param weight Weight for this objective
     */
    void AddObjective(std::shared_ptr<ObjectiveFunction> objective, float weight = 1.0f);

    /**
     * @brief Remove objective function
     * @param index Index of objective to remove
     */
    void RemoveObjective(size_t index);

    /**
     * @brief Update weight of existing objective
     * @param index Index of objective
     * @param weight New weight
     */
    void SetObjectiveWeight(size_t index, float weight);

    float Evaluate(const Tensor& predicted_state,
                   const Tensor* uncertainty = nullptr,
                   const void* context = nullptr) const override;

    Tensor ComputeGradient(const Tensor& predicted_state,
                          const Tensor* uncertainty = nullptr,
                          const void* context = nullptr) const override;

    void UpdateFromFeedback(const Tensor& predicted_state,
                           const Tensor& actual_outcome,
                           float feedback) override;

    std::string GetDescription() const override;

    /**
     * @brief Get number of objectives
     * @return Number of objectives
     */
    size_t GetNumObjectives() const { return objectives_.size(); }

private:
    struct WeightedObjective {
        std::shared_ptr<ObjectiveFunction> objective;
        float weight;
    };
    
    std::vector<WeightedObjective> objectives_;
};

/**
 * @brief Learned objective function using neural networks
 */
class LearnedObjective : public ObjectiveFunction {
public:
    /**
     * @brief Constructor
     * @param state_dim Dimension of state vector
     * @param config Configuration
     */
    explicit LearnedObjective(int state_dim,
                             const ObjectiveFunctionConfig& config = ObjectiveFunctionConfig{});

    float Evaluate(const Tensor& predicted_state,
                   const Tensor* uncertainty = nullptr,
                   const void* context = nullptr) const override;

    Tensor ComputeGradient(const Tensor& predicted_state,
                          const Tensor* uncertainty = nullptr,
                          const void* context = nullptr) const override;

    void UpdateFromFeedback(const Tensor& predicted_state,
                           const Tensor& actual_outcome,
                           float feedback) override;

    std::string GetDescription() const override;

    /**
     * @brief Train the learned objective from data
     * @param states Training states
     * @param rewards Training rewards
     * @param learning_rate Learning rate
     * @param num_epochs Number of training epochs
     */
    void Train(const std::vector<Tensor>& states,
               const std::vector<float>& rewards,
               float learning_rate = 0.001f,
               int num_epochs = 100);

private:
    int state_dim_;
    std::vector<Tensor> network_weights_;
    std::vector<Tensor> training_states_;
    std::vector<float> training_rewards_;
    
    // Simple neural network forward pass
    float NetworkForward(const Tensor& state) const;
    Tensor NetworkGradient(const Tensor& state) const;
    void UpdateNetworkWeights(const Tensor& state, float target, float learning_rate);
};

/**
 * @brief Utility functions for objective functions
 */
namespace ObjectiveUtils {

    /**
     * @brief Create a target state objective
     * @param target_state Target state to reach
     * @param weight Objective weight
     * @return Shared pointer to objective function
     */
    std::shared_ptr<ObjectiveFunction> CreateTargetObjective(
        const Tensor& target_state, float weight = 1.0f);

    /**
     * @brief Create a safety-aware objective (avoid dangerous states)
     * @param danger_threshold Threshold for dangerous states
     * @param weight Objective weight
     * @return Shared pointer to objective function
     */
    std::shared_ptr<ObjectiveFunction> CreateSafetyObjective(
        float danger_threshold, float weight = 10.0f);

    /**
     * @brief Create an energy efficiency objective
     * @param efficiency_weights Weights for each state component
     * @param weight Objective weight
     * @return Shared pointer to objective function
     */
    std::shared_ptr<ObjectiveFunction> CreateEfficiencyObjective(
        const Tensor& efficiency_weights, float weight = 1.0f);

    /**
     * @brief Create a comfort optimization objective (for human factors)
     * @param comfort_preferences User comfort preferences
     * @param weight Objective weight
     * @return Shared pointer to objective function
     */
    std::shared_ptr<ObjectiveFunction> CreateComfortObjective(
        const Tensor& comfort_preferences, float weight = 1.0f);

    /**
     * @brief Validate objective function implementation
     * @param objective Objective function to validate
     * @param test_state Test state for validation
     * @param epsilon Finite difference epsilon
     * @return True if gradients are correct
     */
    bool ValidateObjectiveGradients(
        const ObjectiveFunction& objective,
        const Tensor& test_state,
        float epsilon = 1e-5f);

    /**
     * @brief Benchmark objective function performance
     * @param objective Objective function to benchmark
     * @param test_states Vector of test states
     * @return Performance statistics as string
     */
    std::string BenchmarkObjective(
        const ObjectiveFunction& objective,
        const std::vector<Tensor>& test_states);

} // namespace ObjectiveUtils

} // namespace Core
} // namespace crllwtt

#endif // LWTT_CORE_OBJECTIVE_FUNCTION_HPP
