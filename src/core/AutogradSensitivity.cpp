/**
 * @file AutogradSensitivity.cpp
 * @brief Implementation of autograd-based sensitivity analysis
 * @version 1.0.0
 * @date 2025-06-02
 */

#include "../../include/LwTT/core/AutogradSensitivity.hpp"
#include <stdexcept>
#include <chrono>
#include <cmath>

namespace crllwtt {
namespace Core {

AutogradSensitivity::AutogradSensitivity(const AutogradConfig& config)
    : config_(config), gradient_enabled_(config.enable_autograd),
      forward_passes_(0), backward_passes_(0),
      total_forward_time_(0.0), total_backward_time_(0.0) {
    
    // Initialize torch global settings
    torch::manual_seed(42); // For reproducible results
    torch::set_num_threads(1); // Single-threaded for deterministic performance
}

AutogradSensitivity::~AutogradSensitivity() {
    ClearGraph();
}

Tensor AutogradSensitivity::ComputeSensitivity(
    std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const Tensor* gradient_outputs) {
    
    if (!gradient_enabled_) {
        throw std::runtime_error("Gradient computation is disabled");
    }
    
    ValidateInputTensors(observable_state, controllable_input);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert LwTT tensors to torch tensors with gradient tracking
    torch::Tensor x_torch = ConvertToTorchTensor(observable_state, false);
    torch::Tensor u_torch = ConvertToTorchTensor(controllable_input, true);
    
    // Forward pass
    Tensor predicted_lwtt = prediction_function(observable_state, controllable_input);
    torch::Tensor predicted_torch = ConvertToTorchTensor(predicted_lwtt, false);
    
    auto forward_end = std::chrono::high_resolution_clock::now();
    double forward_time = std::chrono::duration<double, std::milli>(
        forward_end - start_time).count();
    
    // Prepare gradient output for backward pass
    torch::Tensor grad_output;
    if (gradient_outputs) {
        grad_output = ConvertToTorchTensor(*gradient_outputs, false);
    } else {
        grad_output = torch::ones_like(predicted_torch);
    }
    
    // Backward pass to compute gradients
    std::vector<torch::Tensor> gradients = torch::autograd::grad(
        {predicted_torch.sum()}, // outputs (summed for scalar)
        {u_torch},               // inputs
        {grad_output.sum()},     // grad_outputs
        config_.retain_graph,    // retain_graph
        config_.create_graph     // create_graph
    );
    
    auto backward_end = std::chrono::high_resolution_clock::now();
    double backward_time = std::chrono::duration<double, std::milli>(
        backward_end - forward_end).count();
    
    UpdatePerformanceStats(forward_time, backward_time);
    
    if (gradients.empty()) {
        throw std::runtime_error("No gradients computed");
    }
    
    // Apply gradient clipping if enabled
    if (config_.gradient_clip_threshold > 0.0f) {
        ClipGradients(gradients, config_.gradient_clip_threshold);
    }
    
    // Cache gradients for potential reuse
    cached_gradients_ = gradients;
    
    // Convert back to LwTT tensor
    return ConvertFromTorchTensor(gradients[0]);
}

Tensor AutogradSensitivity::ComputeJacobian(
    std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
    const Tensor& observable_state,
    const Tensor& controllable_input) {
    
    ValidateInputTensors(observable_state, controllable_input);
    
    // Convert to torch tensors
    torch::Tensor x_torch = ConvertToTorchTensor(observable_state, false);
    torch::Tensor u_torch = ConvertToTorchTensor(controllable_input, true);
    
    // Get prediction dimensions
    Tensor predicted_sample = prediction_function(observable_state, controllable_input);
    auto pred_shape = predicted_sample.Shape();
    auto input_shape = controllable_input.Shape();
    
    if (pred_shape.size() != 1 || input_shape.size() != 1) {
        throw std::invalid_argument("Jacobian computation requires 1D tensors");
    }
    
    int output_dim = pred_shape[0];
    int input_dim = input_shape[0];
    
    // Create Jacobian matrix
    Tensor jacobian({output_dim, input_dim});
    jacobian.Fill(0.0f);
    
    // Compute Jacobian row by row
    for (int i = 0; i < output_dim; ++i) {
        // Create unit vector for output i
        Tensor unit_output({output_dim});
        unit_output.Fill(0.0f);
        unit_output.GetData()[i] = 1.0f;
        
        // Compute gradient for this output component
        Tensor row_gradient = ComputeSensitivity(
            prediction_function, observable_state, controllable_input, &unit_output);
        
        // Copy to Jacobian matrix
        auto row_shape = row_gradient.Shape();
        int copy_dim = std::min(input_dim, row_shape[0]);
        for (int j = 0; j < copy_dim; ++j) {
            jacobian.GetData()[i * input_dim + j] = row_gradient.GetData()[j];
        }
    }
    
    return jacobian;
}

Tensor AutogradSensitivity::ComputeHessian(
    std::function<Tensor(const Tensor&, const Tensor&)> prediction_function,
    const Tensor& observable_state,
    const Tensor& controllable_input) {
    
    ValidateInputTensors(observable_state, controllable_input);
    
    auto input_shape = controllable_input.Shape();
    if (input_shape.size() != 1) {
        throw std::invalid_argument("Hessian computation requires 1D input tensor");
    }
    
    int input_dim = input_shape[0];
    Tensor hessian({input_dim, input_dim});
    hessian.Fill(0.0f);
    
    // Create a scalar objective function for Hessian computation
    auto scalar_function = [&](const Tensor& x, const Tensor& u) -> Tensor {
        Tensor pred = prediction_function(x, u);
        // Sum all outputs to create scalar
        float sum = 0.0f;
        for (int i = 0; i < pred.GetSize(); ++i) {
            sum += pred.GetData()[i];
        }
        Tensor scalar_output({1});
        scalar_output.GetData()[0] = sum;
        return scalar_output;
    };
    
    // Compute Hessian using second-order derivatives
    // This is a simplified implementation - could be optimized
    for (int i = 0; i < input_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            // Compute mixed partial derivative ∂²f/∂u_i∂u_j
            // Using finite differences on first derivatives
            const float epsilon = 1e-5f;
            
            // Perturb u_j by +epsilon
            Tensor u_plus = controllable_input;
            u_plus.GetData()[j] += epsilon;
            
            Tensor grad_plus = ComputeSensitivity(
                scalar_function, observable_state, u_plus);
            
            // Perturb u_j by -epsilon
            Tensor u_minus = controllable_input;
            u_minus.GetData()[j] -= epsilon;
            
            Tensor grad_minus = ComputeSensitivity(
                scalar_function, observable_state, u_minus);
            
            // Compute second derivative
            float second_deriv = (grad_plus.GetData()[i] - grad_minus.GetData()[i]) / (2.0f * epsilon);
            hessian.GetData()[i * input_dim + j] = second_deriv;
        }
    }
    
    return hessian;
}

torch::Tensor AutogradSensitivity::ConvertToTorchTensor(const Tensor& lwtt_tensor, bool requires_grad) {
    auto shape = lwtt_tensor.Shape();
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());
    
    // Create torch tensor from data
    torch::Tensor torch_tensor = torch::from_blob(
        const_cast<float*>(lwtt_tensor.GetData()),
        torch_shape,
        torch::kFloat32
    ).clone(); // Clone to ensure proper memory management
    
    torch_tensor.set_requires_grad(requires_grad);
    return torch_tensor;
}

Tensor AutogradSensitivity::ConvertFromTorchTensor(const torch::Tensor& torch_tensor) {
    // Ensure tensor is on CPU and contiguous
    torch::Tensor cpu_tensor = torch_tensor.to(torch::kCPU).contiguous();
    
    // Get shape
    auto torch_shape = cpu_tensor.sizes();
    std::vector<int> lwtt_shape(torch_shape.begin(), torch_shape.end());
    
    // Create LwTT tensor
    Tensor lwtt_tensor(lwtt_shape);
    
    // Copy data
    float* torch_data = cpu_tensor.data_ptr<float>();
    float* lwtt_data = lwtt_tensor.GetData();
    int total_elements = lwtt_tensor.GetSize();
    
    for (int i = 0; i < total_elements; ++i) {
        lwtt_data[i] = torch_data[i];
    }
    
    return lwtt_tensor;
}

void AutogradSensitivity::SetGradientEnabled(bool enable) {
    gradient_enabled_ = enable;
    // torch::autograd::set_grad_enabled(enable); // Not available in this LibTorch version
}

bool AutogradSensitivity::IsGradientEnabled() const {
    return gradient_enabled_;
}

void AutogradSensitivity::ClipGradients(std::vector<torch::Tensor>& gradients, float threshold) {
    for (auto& grad : gradients) {
        torch::nn::utils::clip_grad_norm_(grad, threshold);
    }
}

std::string AutogradSensitivity::GetGraphStatistics() const {
    std::string stats = "Autograd Performance Statistics:\\n";
    stats += "  Forward passes: " + std::to_string(forward_passes_) + "\\n";
    stats += "  Backward passes: " + std::to_string(backward_passes_) + "\\n";
    
    if (forward_passes_ > 0) {
        double avg_forward = total_forward_time_ / forward_passes_;
        stats += "  Average forward time: " + std::to_string(avg_forward) + " ms\\n";
    }
    
    if (backward_passes_ > 0) {
        double avg_backward = total_backward_time_ / backward_passes_;
        stats += "  Average backward time: " + std::to_string(avg_backward) + " ms\\n";
    }
    
    stats += "  Gradient enabled: " + std::string(gradient_enabled_ ? "true" : "false") + "\\n";
    stats += "  Cached gradients: " + std::to_string(cached_gradients_.size()) + "\\n";
    
    return stats;
}

void AutogradSensitivity::ClearGraph() {
    cached_gradients_.clear();
    last_output_ = torch::Tensor();
}

void AutogradSensitivity::ValidateInputTensors(const Tensor& observable_state, 
                                              const Tensor& controllable_input) const {
    if (observable_state.GetSize() == 0 || controllable_input.GetSize() == 0) {
        throw std::invalid_argument("Input tensors cannot be empty");
    }
    
    // Check for NaN or infinite values
    for (int i = 0; i < observable_state.GetSize(); ++i) {
        if (!std::isfinite(observable_state.GetData()[i])) {
            throw std::invalid_argument("Observable state contains non-finite values");
        }
    }
    
    for (int i = 0; i < controllable_input.GetSize(); ++i) {
        if (!std::isfinite(controllable_input.GetData()[i])) {
            throw std::invalid_argument("Controllable input contains non-finite values");
        }
    }
}

torch::Tensor AutogradSensitivity::PrepareGradientOutput(const Tensor& predicted_state) const {
    auto shape = predicted_state.Shape();
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());
    torch::Tensor torch_tensor = torch::from_blob(
        const_cast<float*>(predicted_state.GetData()),
        torch_shape,
        torch::kFloat32
    ).clone();
    return torch::ones_like(torch_tensor);
}

void AutogradSensitivity::UpdatePerformanceStats(double forward_time, double backward_time) const {
    forward_passes_++;
    backward_passes_++;
    total_forward_time_ += forward_time;
    total_backward_time_ += backward_time;
}

// AutogradUtils implementation
namespace AutogradUtils {

torch::Tensor CreateTensorWithGrad(const std::vector<int>& shape,
                                  bool requires_grad,
                                  torch::Device device) {
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());
    torch::Tensor tensor = torch::randn(torch_shape, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device)
        .requires_grad(requires_grad));
    return tensor;
}

torch::Tensor ComputeNumericalGradient(std::function<torch::Tensor(const torch::Tensor&)> function,
                                      const torch::Tensor& input,
                                      float epsilon) {
    torch::Tensor grad = torch::zeros_like(input);
    torch::Tensor input_plus = input.clone();
    torch::Tensor input_minus = input.clone();
    
    for (int i = 0; i < input.numel(); ++i) {
        // Perturb input at position i
        input_plus.view(-1)[i] = input.view(-1)[i] + epsilon;
        input_minus.view(-1)[i] = input.view(-1)[i] - epsilon;
        
        // Compute function values
        torch::Tensor f_plus = function(input_plus);
        torch::Tensor f_minus = function(input_minus);
        
        // Compute numerical gradient
        grad.view(-1)[i] = (f_plus.sum() - f_minus.sum()) / (2.0f * epsilon);
        
        // Reset perturbations
        input_plus.view(-1)[i] = input.view(-1)[i];
        input_minus.view(-1)[i] = input.view(-1)[i];
    }
    
    return grad;
}

bool VerifyGradients(const torch::Tensor& autograd_grad,
                    const torch::Tensor& numerical_grad,
                    float tolerance) {
    torch::Tensor diff = torch::abs(autograd_grad - numerical_grad);
    torch::Tensor max_diff = torch::max(diff);
    return max_diff.item<float>() < tolerance;
}

std::string ProfileAutogradPerformance(std::function<void()> computation,
                                      int num_iterations) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        computation();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    double avg_time = total_time / num_iterations;
    
    std::string result = "Autograd Performance Profile:\\n";
    result += "  Iterations: " + std::to_string(num_iterations) + "\\n";
    result += "  Total time: " + std::to_string(total_time) + " ms\\n";
    result += "  Average time: " + std::to_string(avg_time) + " ms\\n";
    result += "  Throughput: " + std::to_string(1000.0 / avg_time) + " ops/sec\\n";
    
    return result;
}

torch::jit::script::Module OptimizeModel([[maybe_unused]] torch::nn::Module& model,
                                        [[maybe_unused]] const torch::Tensor& sample_input) {
    model.eval(); // Set to evaluation mode
    
    // Note: torch::jit::trace is not available in this LibTorch version
    // For now, just return an empty module
    // In a full implementation, alternative optimization methods would be used
    throw std::runtime_error("Model optimization not available in this LibTorch version");
}

} // namespace AutogradUtils

} // namespace Core
} // namespace crllwtt
