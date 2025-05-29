/**
 * @file STATransformer.cpp
 * @brief STA (Sense The Ambience) Architecture Implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "../../include/LwTT/core/STATransformer.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>

namespace crllwtt {
namespace Core {

// TargetStateEvaluator implementation
float TargetStateEvaluator::Evaluate(const Tensor& predicted_state, 
                                   const Tensor* uncertainty) const {
    // Compute negative squared distance to target (higher is better)
    auto diff = predicted_state.Add(target_state_.MultiplyScalar(-1.0f));
    float distance_sq = 0.0f;
    
    for (int i = 0; i < diff.GetSize(); ++i) {
        float val = diff.GetData()[i];
        distance_sq += val * val;
    }
    
    float reward = -distance_sq * weight_;
    
    // Apply uncertainty penalty if provided
    if (uncertainty) {
        float uncertainty_penalty = 0.0f;
        for (int i = 0; i < uncertainty->GetSize(); ++i) {
            uncertainty_penalty += uncertainty->GetData()[i];
        }
        reward -= 0.1f * uncertainty_penalty; // Small penalty for uncertainty
    }
    
    return reward;
}

Tensor TargetStateEvaluator::ComputeGradient(const Tensor& predicted_state) const {
    // Gradient of -||s - s*||^2 w.r.t. s is -2 * weight * (s - s*)
    auto diff = predicted_state.Add(target_state_.MultiplyScalar(-1.0f));
    return diff.MultiplyScalar(-2.0f * weight_);
}

// STATransformer implementation
STATransformer::STATransformer(const STAConfig& config) 
    : config_(config), buffer_index_(0), prediction_count_(0),
      total_prediction_time_(0.0), total_update_time_(0.0),
      cumulative_loss_(0.0f), streaming_mode_(false) {
    
    // Validate configuration
    if (config_.observable_state_dim <= 0 || 
        config_.controllable_input_dim <= 0 || 
        config_.predicted_state_dim <= 0) {
        throw std::invalid_argument("Invalid STA configuration: dimensions must be positive");
    }
    
    InitializeModels();
    InitializePersonalEmbeddings();
    
    // Initialize experience buffer
    experience_buffer_.reserve(config_.buffer_size);
}

STATransformer::~STATransformer() {
    // Destructor implementation
}

void STATransformer::InitializeModels() {
    // Configure base transformer
    TransformerConfig transformer_config = config_.transformer_config;
    
    // Ensure input dimension matches combined observable state + controllable input + personal embedding
    int total_input_dim = config_.observable_state_dim + config_.controllable_input_dim;
    if (config_.enable_personal_adaptation) {
        total_input_dim += transformer_config.personal_embed_dim;
    }
    
    // Set transformer dimensions
    transformer_config.d_model = std::max(transformer_config.d_model, total_input_dim);
    transformer_config.max_seq_len = std::max(transformer_config.max_seq_len, 50);
    
    // Create main prediction model
    prediction_model_ = std::make_unique<Transformer>(transformer_config);
    
    // Create ensemble models for uncertainty estimation
    if (config_.enable_uncertainty && config_.ensemble_size > 1) {
        ensemble_models_.reserve(config_.ensemble_size - 1);
        
        for (int i = 1; i < config_.ensemble_size; ++i) {
            auto ensemble_config = transformer_config;
            // Add slight variations to ensemble models
            ensemble_config.dropout_rate += 0.01f * i;
            ensemble_models_.emplace_back(std::make_unique<Transformer>(ensemble_config));
        }
    }
}

void STATransformer::InitializePersonalEmbeddings() {
    if (!config_.enable_personal_adaptation) return;
    
    // Pre-allocate some personal embeddings
    int embed_dim = config_.transformer_config.personal_embed_dim;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (int i = 0; i < std::min(config_.max_persons, 10); ++i) {
        Tensor embedding({embed_dim});
        for (int j = 0; j < embed_dim; ++j) {
            embedding.GetData()[j] = dist(gen);
        }
        personal_embeddings_[i] = std::move(embedding);
    }
}

Tensor STATransformer::CombineInputs(const Tensor& observable_state,
                                    const Tensor& controllable_input,
                                    int personal_id) const {
    
    int total_dim = config_.observable_state_dim + config_.controllable_input_dim;
    if (config_.enable_personal_adaptation && personal_id >= 0) {
        total_dim += config_.transformer_config.personal_embed_dim;
    }
    
    Tensor combined({1, 1, total_dim}); // [batch_size=1, seq_len=1, feature_dim]
    float* data = combined.GetData();
    
    // Copy observable state
    const float* obs_data = observable_state.GetData();
    for (int i = 0; i < config_.observable_state_dim; ++i) {
        data[i] = obs_data[i];
    }
    
    // Copy controllable input
    const float* ctrl_data = controllable_input.GetData();
    for (int i = 0; i < config_.controllable_input_dim; ++i) {
        data[config_.observable_state_dim + i] = ctrl_data[i];
    }
    
    // Add personal embedding if enabled
    if (config_.enable_personal_adaptation && personal_id >= 0) {
        auto it = personal_embeddings_.find(personal_id);
        if (it != personal_embeddings_.end()) {
            const float* embed_data = it->second.GetData();
            int embed_dim = config_.transformer_config.personal_embed_dim;
            for (int i = 0; i < embed_dim; ++i) {
                data[config_.observable_state_dim + config_.controllable_input_dim + i] = embed_data[i];
            }
        } else {
            // Initialize with zeros for unknown person
            int embed_dim = config_.transformer_config.personal_embed_dim;
            for (int i = 0; i < embed_dim; ++i) {
                data[config_.observable_state_dim + config_.controllable_input_dim + i] = 0.0f;
            }
        }
    }
    
    return combined;
}

Tensor STATransformer::ExtractPredictedState(const Tensor& model_output) const {
    // Extract the predicted state from the transformer output
    // Assume the last config_.predicted_state_dim dimensions contain the predicted state
    auto output_shape = model_output.Shape();
    if (output_shape.size() != 3) {
        throw std::runtime_error("Model output must be 3D tensor");
    }
    
    int feature_dim = output_shape[2];
    if (feature_dim < config_.predicted_state_dim) {
        throw std::runtime_error("Model output feature dimension too small");
    }
    
    // Extract last time step and last config_.predicted_state_dim features
    Tensor predicted_state({config_.predicted_state_dim});
    const float* output_data = model_output.GetData();
    
    // Get the last time step (assuming batch_size=1)
    int last_timestep_offset = (output_shape[1] - 1) * feature_dim;
    int state_offset = feature_dim - config_.predicted_state_dim;
    
    for (int i = 0; i < config_.predicted_state_dim; ++i) {
        predicted_state.GetData()[i] = output_data[last_timestep_offset + state_offset + i];
    }
    
    return predicted_state;
}

Tensor STATransformer::PredictState(const Tensor& observable_state,
                                   const Tensor& controllable_input,
                                   const TimeInfo* time_info,
                                   int personal_id) {
    
    if (!ValidateInputDimensions(observable_state, controllable_input)) {
        throw std::invalid_argument("Input dimensions do not match configuration");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Combine inputs
    Tensor combined_input = CombineInputs(observable_state, controllable_input, personal_id);
    
    // Forward pass through transformer
    Tensor model_output = prediction_model_->Forward(combined_input, nullptr, time_info, personal_id);
    
    // Extract predicted state
    Tensor predicted_state = ExtractPredictedState(model_output);
    
    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end_time - start_time);
    UpdatePerformanceStats(elapsed.count(), 0.0, 0.0f);
    
    return predicted_state;
}

std::pair<Tensor, Tensor> STATransformer::PredictWithUncertainty(
    const Tensor& observable_state,
    const Tensor& controllable_input,
    const TimeInfo* time_info,
    int personal_id) {
    
    if (!config_.enable_uncertainty || ensemble_models_.empty()) {
        // Return prediction with zero uncertainty
        Tensor prediction = PredictState(observable_state, controllable_input, time_info, personal_id);
        Tensor uncertainty({config_.predicted_state_dim});
        uncertainty.Fill(0.0f);
        return std::make_pair(prediction, uncertainty);
    }
    
    // Get predictions from all ensemble models
    std::vector<Tensor> predictions;
    predictions.reserve(config_.ensemble_size);
    
    // Main model prediction
    predictions.push_back(PredictState(observable_state, controllable_input, time_info, personal_id));
    
    // Ensemble model predictions
    Tensor combined_input = CombineInputs(observable_state, controllable_input, personal_id);
    
    for (auto& model : ensemble_models_) {
        Tensor model_output = model->Forward(combined_input, nullptr, time_info, personal_id);
        predictions.push_back(ExtractPredictedState(model_output));
    }
    
    // Compute mean and variance
    Tensor mean_prediction({config_.predicted_state_dim});
    Tensor uncertainty({config_.predicted_state_dim});
    mean_prediction.Fill(0.0f);
    uncertainty.Fill(0.0f);
    
    // Compute mean
    for (const auto& pred : predictions) {
        for (int i = 0; i < config_.predicted_state_dim; ++i) {
            mean_prediction.GetData()[i] += pred.GetData()[i] / config_.ensemble_size;
        }
    }
    
    // Compute variance (uncertainty)
    for (const auto& pred : predictions) {
        for (int i = 0; i < config_.predicted_state_dim; ++i) {
            float diff = pred.GetData()[i] - mean_prediction.GetData()[i];
            uncertainty.GetData()[i] += diff * diff / config_.ensemble_size;
        }
    }
    
    // Convert variance to standard deviation
    for (int i = 0; i < config_.predicted_state_dim; ++i) {
        uncertainty.GetData()[i] = std::sqrt(uncertainty.GetData()[i]);
    }
    
    return std::make_pair(mean_prediction, uncertainty);
}

Tensor STATransformer::ComputeSensitivity(const Tensor& observable_state,
                                         const Tensor& controllable_input,
                                         const TimeInfo* time_info,
                                         int personal_id) {
    
    // Use numerical differentiation to compute ∂ŝ/∂u
    return ComputeNumericalGradient(observable_state, controllable_input, time_info, personal_id);
}

Tensor STATransformer::ComputeNumericalGradient(const Tensor& observable_state,
                                               const Tensor& controllable_input,
                                               const TimeInfo* time_info,
                                               int personal_id) const {
    
    const float epsilon = 1e-4f;
    
    // Create sensitivity matrix [predicted_state_dim x controllable_input_dim]
    Tensor sensitivity({config_.predicted_state_dim, config_.controllable_input_dim});
    sensitivity.Fill(0.0f);
    
    // Get baseline prediction
    Tensor baseline_pred = const_cast<STATransformer*>(this)->PredictState(
        observable_state, controllable_input, time_info, personal_id);
    
    // Compute partial derivatives for each controllable input dimension
    for (int u_idx = 0; u_idx < config_.controllable_input_dim; ++u_idx) {
        // Create perturbed input
        Tensor perturbed_input = controllable_input;
        perturbed_input.GetData()[u_idx] += epsilon;
        
        // Get prediction with perturbed input
        Tensor perturbed_pred = const_cast<STATransformer*>(this)->PredictState(
            observable_state, perturbed_input, time_info, personal_id);
        
        // Compute numerical gradient
        for (int s_idx = 0; s_idx < config_.predicted_state_dim; ++s_idx) {
            float gradient = (perturbed_pred.GetData()[s_idx] - baseline_pred.GetData()[s_idx]) / epsilon;
            sensitivity.GetData()[s_idx * config_.controllable_input_dim + u_idx] = gradient;
        }
    }
    
    return sensitivity;
}

Tensor STATransformer::ComputeOptimalControl(const Tensor& observable_state,
                                            const Tensor& current_input,
                                            const MetaEvaluationFunction& meta_eval,
                                            const TimeInfo* time_info,
                                            int personal_id) {
    
    // Compute sensitivity ∂ŝ/∂u
    Tensor sensitivity = ComputeSensitivity(observable_state, current_input, time_info, personal_id);
    
    // Get current prediction
    Tensor current_prediction = PredictState(observable_state, current_input, time_info, personal_id);
    
    // Compute gradient of meta-evaluation function ∇J(ŝ)
    Tensor eval_gradient = meta_eval.ComputeGradient(current_prediction);
    
    // Compute control update using chain rule: ∇u J = (∂ŝ/∂u)^T ∇ŝ J
    Tensor control_gradient({config_.controllable_input_dim});
    control_gradient.Fill(0.0f);
    
    for (int u_idx = 0; u_idx < config_.controllable_input_dim; ++u_idx) {
        float gradient_sum = 0.0f;
        for (int s_idx = 0; s_idx < config_.predicted_state_dim; ++s_idx) {
            float sensitivity_val = sensitivity.GetData()[s_idx * config_.controllable_input_dim + u_idx];
            float eval_grad_val = eval_gradient.GetData()[s_idx];
            gradient_sum += sensitivity_val * eval_grad_val;
        }
        control_gradient.GetData()[u_idx] = gradient_sum;
    }
    
    // Apply control update: u[k] = u[k-1] + η_u * ∇u J
    Tensor optimal_control = current_input;
    for (int i = 0; i < config_.controllable_input_dim; ++i) {
        optimal_control.GetData()[i] += config_.control_gain * control_gradient.GetData()[i];
    }
    
    return optimal_control;
}

void STATransformer::UpdateModel(const Tensor& observable_state,
                                const Tensor& controllable_input,
                                const Tensor& actual_state,
                                const TimeInfo* time_info,
                                int personal_id) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get prediction
    Tensor predicted_state = PredictState(observable_state, controllable_input, time_info, personal_id);
    
    // Compute loss
    float loss = ComputePredictionLoss(predicted_state, actual_state);
    
    // Compute prediction error for backpropagation
    Tensor prediction_error = predicted_state.Add(actual_state.MultiplyScalar(-1.0f));
    
    // Perform gradient descent update
    BackpropagateGradients(prediction_error);
    
    // Update personal embedding if enabled
    if (config_.enable_personal_adaptation && personal_id >= 0) {
        UpdatePersonalEmbedding(personal_id, prediction_error);
    }
    
    // Store experience
    ExperienceEntry experience;
    experience.observable_state = observable_state;
    experience.controllable_input = controllable_input;
    experience.predicted_state = predicted_state;
    experience.actual_state = actual_state;
    if (time_info) experience.time_info = *time_info;
    experience.personal_id = personal_id;
    experience.reward = -loss; // Negative loss as reward
    experience.timestamp = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    AddExperience(experience);
    
    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end_time - start_time);
    UpdatePerformanceStats(0.0, elapsed.count(), loss);
}

Tensor STATransformer::STAStep(const Tensor& observable_state,
                              const Tensor& current_input,
                              const MetaEvaluationFunction& meta_eval,
                              const TimeInfo* time_info,
                              int personal_id) {
    
    return ComputeOptimalControl(observable_state, current_input, meta_eval, time_info, personal_id);
}

void STATransformer::AddExperience(const ExperienceEntry& experience) {
    if (experience_buffer_.size() < static_cast<size_t>(config_.buffer_size)) {
        experience_buffer_.push_back(experience);
    } else {
        experience_buffer_[buffer_index_] = experience;
        buffer_index_ = (buffer_index_ + 1) % config_.buffer_size;
    }
}

float STATransformer::GetPredictionConfidence(const Tensor& observable_state,
                                             const Tensor& controllable_input) {
    
    if (!config_.enable_uncertainty) {
        return 1.0f;
    }
    
    auto [prediction, uncertainty] = PredictWithUncertainty(observable_state, controllable_input);
    
    float avg_uncertainty = 0.0f;
    for (int i = 0; i < config_.predicted_state_dim; ++i) {
        avg_uncertainty += uncertainty.GetData()[i];
    }
    avg_uncertainty /= config_.predicted_state_dim;
    
    float confidence = 1.0f / (1.0f + avg_uncertainty);
    return std::max(0.0f, std::min(1.0f, confidence));
}

std::string STATransformer::GetPerformanceStats() const {
    if (prediction_count_ == 0) {
        return "No predictions made yet";
    }
    
    double avg_prediction_time = total_prediction_time_ / prediction_count_;
    double avg_update_time = total_update_time_ / prediction_count_;
    float avg_loss = cumulative_loss_ / prediction_count_;
    
    std::string stats = "STA Transformer Performance Statistics:\n";
    stats += "  Total predictions: " + std::to_string(prediction_count_) + "\n";
    stats += "  Average prediction time: " + std::to_string(avg_prediction_time) + " ms\n";
    stats += "  Average update time: " + std::to_string(avg_update_time) + " ms\n";
    stats += "  Average loss: " + std::to_string(avg_loss) + "\n";
    stats += "  Experience buffer size: " + std::to_string(experience_buffer_.size()) + "\n";
    stats += "  Personal embeddings: " + std::to_string(personal_embeddings_.size()) + "\n";
    stats += "  Streaming mode: " + std::string(streaming_mode_ ? "enabled" : "disabled") + "\n";
    
    return stats;
}

void STATransformer::Reset() {
    prediction_count_ = 0;
    total_prediction_time_ = 0.0;
    total_update_time_ = 0.0;
    cumulative_loss_ = 0.0f;
    buffer_index_ = 0;
    
    experience_buffer_.clear();
    gradient_cache_.clear();
    
    if (prediction_model_) {
        prediction_model_->Reset();
    }
    
    for (auto& model : ensemble_models_) {
        if (model) {
            model->Reset();
        }
    }
    
    personal_embeddings_.clear();
    if (config_.enable_personal_adaptation) {
        InitializePersonalEmbeddings();
    }
}

bool STATransformer::SaveModel(const std::string& filepath) const {
    if (prediction_model_) {
        return prediction_model_->SaveModel(filepath);
    }
    return false;
}

bool STATransformer::LoadModel(const std::string& filepath) {
    if (prediction_model_) {
        return prediction_model_->LoadModel(filepath);
    }
    return false;
}

void STATransformer::UpdatePersonalEmbedding(int personal_id, const Tensor& gradient) {
    if (!config_.enable_personal_adaptation || personal_id < 0) return;
    
    auto it = personal_embeddings_.find(personal_id);
    if (it == personal_embeddings_.end()) {
        Tensor new_embedding({config_.transformer_config.personal_embed_dim});
        new_embedding.Fill(0.0f);
        personal_embeddings_[personal_id] = std::move(new_embedding);
        it = personal_embeddings_.find(personal_id);
    }
    
    float* embed_data = it->second.GetData();
    int embed_dim = config_.transformer_config.personal_embed_dim;
    
    int update_dim = std::min(embed_dim, gradient.GetSize());
    for (int i = 0; i < update_dim; ++i) {
        embed_data[i] -= config_.personal_learning_rate * gradient.GetData()[i];
    }
}

void STATransformer::BackpropagateGradients(const Tensor& prediction_error) {
    gradient_cache_.clear();
    gradient_cache_.push_back(prediction_error);
    
    // TODO: Implement proper backpropagation through transformer layers
}

float STATransformer::ComputePredictionLoss(const Tensor& predicted, const Tensor& actual) const {
    if (predicted.GetSize() != actual.GetSize()) {
        throw std::invalid_argument("Predicted and actual tensors must have same size");
    }
    
    float loss = 0.0f;
    for (int i = 0; i < predicted.GetSize(); ++i) {
        float diff = predicted.GetData()[i] - actual.GetData()[i];
        loss += diff * diff;
    }
    
    return loss / predicted.GetSize();
}

void STATransformer::UpdatePerformanceStats(double prediction_time, 
                                           double update_time, 
                                           float loss) {
    prediction_count_++;
    total_prediction_time_ += prediction_time;
    total_update_time_ += update_time;
    cumulative_loss_ += loss;
}

bool STATransformer::ValidateInputDimensions(const Tensor& observable_state,
                                           const Tensor& controllable_input) const {
    
    auto obs_shape = observable_state.Shape();
    auto ctrl_shape = controllable_input.Shape();
    
    if (obs_shape.size() != 1 || obs_shape[0] != config_.observable_state_dim) {
        return false;
    }
    
    if (ctrl_shape.size() != 1 || ctrl_shape[0] != config_.controllable_input_dim) {
        return false;
    }
    
    return true;
}

} // namespace Core
} // namespace LwTT