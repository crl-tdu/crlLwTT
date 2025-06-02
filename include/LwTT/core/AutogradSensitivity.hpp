/**
 * @file AutogradSensitivity.hpp
 * @brief Autograd-based sensitivity analysis for STA architecture
 * @version 1.0.0
 * @date 2025-06-02
 */

#ifndef LWTT_CORE_AUTOGRAD_SENSITIVITY_HPP
#define LWTT_CORE_AUTOGRAD_SENSITIVITY_HPP

#include "Tensor.hpp"
#include <torch/torch.h>
#include <functional>
#include <memory>
#include <vector>

namespace crllwtt {
namespace Core {

/**
 * @brief Configuration for autograd-based sensitivity analysis
 */
struct AutogradConfig {
    bool enable_autograd = true;                ///< Enable autograd computation
    bool retain_graph = true;                   ///< Retain computation graph for multiple backward passes
    bool create_graph = false;                  ///< Create graph for higher-order derivatives
    float gradient_clip_threshold = 1.0f;      ///< Gradient clipping threshold
    bool enable_gradient_checkpointing = false; ///< Enable gradient checkpointing for memory efficiency
    int max_backward_iterations = 10;          ///< Maximum number of backward iterations
};

/**
 * @brief Autograd-based sensitivity analyzer
 * 
 * This class provides efficient automatic differentiation for computing
 * sensitivities ∂ŝ/∂u using LibTorch's autograd system.
 */
class AutogradSensitivity {
public:
    /**
     * @brief Constructor
     * @param config Autograd configuration
     */
    explicit AutogradSensitivity(const AutogradConfig& config = AutogradConfig{});

    /**
     * @brief Destructor
     */
    ~AutogradSensitivity();

    /**
     * @brief Compute sensitivity using autograd
     * @param prediction_function Function that computes ŝ = f(x, u)
     * @param observable_state Observable state x
     * @param controllable_input Controllable input u (requires_grad=True)
     * @param gradient_outputs Optional gradient outputs for backward pass
     * @return Sensitivity tensor ∂ŝ/∂u
     */
    Tensor ComputeSensitivity(
        std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        const Tensor* gradient_outputs = nullptr
    );

    /**
     * @brief Compute higher-order derivatives
     * @param prediction_function Function that computes ŝ = f(x, u)
     * @param observable_state Observable state x
     * @param controllable_input Controllable input u (requires_grad=True)
     * @param order Derivative order (1=first, 2=second, etc.)
     * @return Higher-order derivative tensor
     */
    Tensor ComputeHigherOrderDerivative(
        std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
        const Tensor& observable_state,
        const Tensor& controllable_input,
        int order = 2
    );

    /**
     * @brief Compute Jacobian matrix using forward-mode autodiff
     * @param prediction_function Function that computes ŝ = f(x, u)
     * @param observable_state Observable state x
     * @param controllable_input Controllable input u
     * @return Jacobian matrix [predicted_state_dim x controllable_input_dim]
     */
    Tensor ComputeJacobian(
        std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
        const Tensor& observable_state,
        const Tensor& controllable_input
    );

    /**
     * @brief Compute Hessian matrix for second-order optimization
     * @param prediction_function Function that computes ŝ = f(x, u)
     * @param observable_state Observable state x
     * @param controllable_input Controllable input u
     * @return Hessian matrix [controllable_input_dim x controllable_input_dim]
     */
    Tensor ComputeHessian(
        std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
        const Tensor& observable_state,
        const Tensor& controllable_input
    );

    /**
     * @brief Convert LwTT Tensor to torch::Tensor with gradient tracking
     * @param lwtt_tensor Input LwTT tensor
     * @param requires_grad Whether to require gradients
     * @return torch::Tensor with gradient tracking
     */
    torch::Tensor ConvertToTorchTensor(const Tensor& lwtt_tensor, bool requires_grad = false);

    /**
     * @brief Convert torch::Tensor back to LwTT Tensor
     * @param torch_tensor Input torch tensor
     * @return LwTT Tensor
     */
    Tensor ConvertFromTorchTensor(const torch::Tensor& torch_tensor);

    /**
     * @brief Enable/disable gradient computation
     * @param enable Enable gradient computation
     */
    void SetGradientEnabled(bool enable);

    /**
     * @brief Check if gradients are enabled
     * @return True if gradients are enabled
     */
    bool IsGradientEnabled() const;

    /**
     * @brief Apply gradient clipping
     * @param gradients Vector of gradient tensors
     * @param threshold Clipping threshold
     */
    void ClipGradients(std::vector<torch::Tensor>& gradients, float threshold);

    /**
     * @brief Get computational graph statistics
     * @return String containing graph statistics
     */
    std::string GetGraphStatistics() const;

    /**
     * @brief Clear computational graph to free memory
     */
    void ClearGraph();

    /**
     * @brief Get configuration
     * @return Current configuration
     */
    const AutogradConfig& GetConfig() const { return config_; }

private:
    AutogradConfig config_;
    bool gradient_enabled_;
    mutable torch::Tensor last_output_;
    mutable std::vector<torch::Tensor> cached_gradients_;
    
    // Performance tracking
    mutable size_t forward_passes_;
    mutable size_t backward_passes_;
    mutable double total_forward_time_;
    mutable double total_backward_time_;

    // Private helper methods
    void ValidateInputTensors(const Tensor& observable_state, 
                             const Tensor& controllable_input) const;
    torch::Tensor PrepareGradientOutput(const Tensor& predicted_state) const;
    void UpdatePerformanceStats(double forward_time, double backward_time) const;
};

/**
 * @brief Utility functions for autograd-based sensitivity analysis
 */
namespace AutogradUtils {

    /**
     * @brief Create torch::Tensor with specified gradient requirements
     * @param shape Tensor shape
     * @param requires_grad Whether to require gradients
     * @param device Device to place tensor on
     * @return torch::Tensor with gradient tracking
     */
    torch::Tensor CreateTensorWithGrad(
        const std::vector<int>& shape,
        bool requires_grad = true,
        torch::Device device = torch::kCPU
    );

    /**
     * @brief Compute numerical gradient for verification
     * @param function Function to differentiate
     * @param input Input tensor
     * @param epsilon Finite difference epsilon
     * @return Numerical gradient
     */
    torch::Tensor ComputeNumericalGradient(
        std::function<torch::Tensor(const torch::Tensor&)> function,
        const torch::Tensor& input,
        float epsilon = 1e-5f
    );

    /**
     * @brief Verify autograd implementation against numerical gradients
     * @param autograd_grad Autograd-computed gradient
     * @param numerical_grad Numerically-computed gradient
     * @param tolerance Verification tolerance
     * @return True if gradients match within tolerance
     */
    bool VerifyGradients(
        const torch::Tensor& autograd_grad,
        const torch::Tensor& numerical_grad,
        float tolerance = 1e-4f
    );

    /**
     * @brief Profile autograd computation performance
     * @param computation Computation to profile
     * @param num_iterations Number of iterations for averaging
     * @return Performance statistics as string
     */
    std::string ProfileAutogradPerformance(
        std::function<void()> computation,
        int num_iterations = 100
    );

    /**
     * @brief Optimize computation graph for inference
     * @param model Model to optimize
     * @param sample_input Sample input for tracing
     * @return Optimized model
     */
    torch::jit::script::Module OptimizeModel(
        torch::nn::Module& model,
        const torch::Tensor& sample_input
    );

} // namespace AutogradUtils

} // namespace Core
} // namespace crllwtt

#endif // LWTT_CORE_AUTOGRAD_SENSITIVITY_HPP
