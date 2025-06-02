/**
 * @file TimeEncoding.cpp
 * @brief Time-aware encoding implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "../../include/LwTT/core/TimeEncoding.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace crllwtt {
namespace Core {

TimeEncoding::TimeEncoding(const TimeEncodingConfig& config)
    : config_(config), 
      positional_encoding_({config_.max_seq_len, config_.d_model}),
      time_scale_weights_({config_.num_time_scales, config_.d_model}),
      personal_embeddings_({config_.max_persons, config_.personal_embed_dim}),
      personal_projection_({config_.personal_embed_dim, config_.d_model}) {
    
    // Validate configuration
    if (config_.d_model <= 0 || config_.max_seq_len <= 0) {
        throw std::invalid_argument("Invalid time encoding configuration");
    }
    
    InitializeEncodingTables();
}

TimeEncoding::~TimeEncoding() {
    // Destructor implementation
}

void TimeEncoding::InitializeEncodingTables() {
    // Initialize positional encoding values
    for (int pos = 0; pos < config_.max_seq_len; ++pos) {
        for (int i = 0; i < config_.d_model; i += 2) {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / config_.d_model);
            positional_encoding_.Set({pos, i}, std::sin(angle));
            if (i + 1 < config_.d_model) {
                positional_encoding_.Set({pos, i + 1}, std::cos(angle));
            }
        }
    }
    
    // Initialize time scale weights if enabled
    if (config_.enable_time_scaling && config_.num_time_scales > 1) {
        time_scale_weights_.RandomNormal(0.0f, 0.02f);
    }
    
    // Initialize personal embeddings if enabled
    if (config_.personal_embed_dim > 0 && config_.max_persons > 0) {
        personal_embeddings_.RandomNormal(0.0f, 0.02f);
        personal_projection_.RandomNormal(0.0f, 0.02f);
    }
}

Tensor TimeEncoding::Apply(const Tensor& input, 
                          const TimeInfo* time_info, 
                          int personal_id) const {
    
    auto input_shape = input.Shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Input must be 3D [batch_size, seq_len, d_model]");
    }
    
    [[maybe_unused]] int batch_size = input_shape[0];
    int seq_len = input_shape[1];
    int d_model = input_shape[2];
    
    if (d_model != config_.d_model) {
        throw std::invalid_argument("Input d_model must match config d_model");
    }
    
    if (seq_len > config_.max_seq_len) {
        throw std::invalid_argument("Sequence length exceeds maximum");
    }
    
    // Start with input
    Tensor encoded = input;
    
    // Add positional encoding
    AddPositionalEncoding(encoded, seq_len);
    
    // Add time-aware encoding if time info is provided
    if (time_info && config_.enable_time_encoding) {
        AddTimeAwareEncoding(encoded, *time_info);
    }
    
    // Add personal encoding if personal_id is provided
    if (personal_id >= 0 && config_.personal_embed_dim > 0) {
        AddPersonalEncoding(encoded, personal_id);
    }
    
    return encoded;
}

void TimeEncoding::AddPositionalEncoding(Tensor& input, int seq_len) const {
    auto input_shape = input.Shape();
    int batch_size = input_shape[0];
    
    // Add positional encoding to each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int d = 0; d < config_.d_model; ++d) {
                float pos_encoding = positional_encoding_.Get({pos, d});
                float current_val = input.Get({b, pos, d});
                input.Set({b, pos, d}, current_val + pos_encoding);
            }
        }
    }
}

void TimeEncoding::AddTimeAwareEncoding(Tensor& input, const TimeInfo& time_info) const {
    auto input_shape = input.Shape();
    int batch_size = input_shape[0];
    int seq_len = input_shape[1];
    
    // Apply time scaling if enabled
    float time_scale = config_.time_scale;
    
    // Simple time encoding based on time differences
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len && t < static_cast<int>(time_info.timestamps.size()); ++t) {
            float timestamp = time_info.timestamps[t] * time_scale;
            
            // Apply time-dependent modulation
            for (int d = 0; d < config_.d_model; d += 2) {
                float time_angle = timestamp / std::pow(10000.0f, (2.0f * d) / config_.d_model);
                float time_sin = std::sin(time_angle);
                float time_cos = std::cos(time_angle);
                
                // Modulate existing features
                float current_val1 = input.Get({b, t, d});
                input.Set({b, t, d}, current_val1 * (1.0f + 0.1f * time_sin));
                
                if (d + 1 < config_.d_model) {
                    float current_val2 = input.Get({b, t, d + 1});
                    input.Set({b, t, d + 1}, current_val2 * (1.0f + 0.1f * time_cos));
                }
            }
        }
    }
    
    // Apply personal delay compensation if available
    if (time_info.personal_delay > 0.0f) {
        ApplyDelayCompensation(input, time_info.personal_delay);
    }
}

void TimeEncoding::AddPersonalEncoding(Tensor& input, int personal_id) const {
    if (personal_id < 0 || personal_id >= config_.max_persons) {
        return; // Invalid personal ID, skip
    }
    
    auto input_shape = input.Shape();
    int batch_size = input_shape[0];
    int seq_len = input_shape[1];
    
    // Get personal embedding
    Tensor personal_embed({config_.personal_embed_dim});
    for (int i = 0; i < config_.personal_embed_dim; ++i) {
        personal_embed.Set({i}, personal_embeddings_.Get({personal_id, i}));
    }
    
    // Project personal embedding to model dimension
    Tensor projected_personal({config_.d_model});
    projected_personal.Fill(0.0f);
    
    for (int d = 0; d < config_.d_model; ++d) {
        float sum = 0.0f;
        for (int p = 0; p < config_.personal_embed_dim; ++p) {
            sum += personal_embed.Get({p}) * personal_projection_.Get({p, d});
        }
        projected_personal.Set({d}, sum);
    }
    
    // Add personal encoding to all positions
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < config_.d_model; ++d) {
                float current_val = input.Get({b, t, d});
                float personal_val = projected_personal.Get({d});
                input.Set({b, t, d}, current_val + personal_val);
            }
        }
    }
}

void TimeEncoding::ApplyDelayCompensation(Tensor& input, float personal_delay) const {
    // Simple delay compensation by shifting temporal features
    auto input_shape = input.Shape();
    int batch_size = input_shape[0];
    int seq_len = input_shape[1];
    
    // Calculate delay in time steps (simplified)
    int delay_steps = static_cast<int>(std::round(personal_delay * 10.0f)); // Assume 0.1s per step
    delay_steps = std::max(0, std::min(delay_steps, seq_len - 1));
    
    if (delay_steps > 0) {
        // Shift features to compensate for delay
        Tensor temp_input = input;
        
        for (int b = 0; b < batch_size; ++b) {
            for (int t = delay_steps; t < seq_len; ++t) {
                for (int d = 0; d < config_.d_model; ++d) {
                    float shifted_val = temp_input.Get({b, t - delay_steps, d});
                    input.Set({b, t, d}, shifted_val);
                }
            }
            
            // Fill the first few positions with zeros or extrapolation
            for (int t = 0; t < delay_steps; ++t) {
                for (int d = 0; d < config_.d_model; ++d) {
                    input.Set({b, t, d}, 0.0f);
                }
            }
        }
    }
}

Tensor TimeEncoding::GetPersonalEmbedding(int personal_id) const {
    if (personal_id < 0 || personal_id >= config_.max_persons || config_.personal_embed_dim <= 0) {
        throw std::invalid_argument("Invalid personal ID or personal embeddings not enabled");
    }
    
    Tensor embedding({config_.personal_embed_dim});
    for (int i = 0; i < config_.personal_embed_dim; ++i) {
        embedding.Set({i}, personal_embeddings_.Get({personal_id, i}));
    }
    
    return embedding;
}

void TimeEncoding::UpdatePersonalEmbedding(int personal_id, const Tensor& new_embedding) {
    if (personal_id < 0 || personal_id >= config_.max_persons || config_.personal_embed_dim <= 0) {
        throw std::invalid_argument("Invalid personal ID or personal embeddings not enabled");
    }
    
    auto emb_shape = new_embedding.Shape();
    if (emb_shape.size() != 1 || emb_shape[0] != config_.personal_embed_dim) {
        throw std::invalid_argument("Embedding dimension mismatch");
    }
    
    for (int i = 0; i < config_.personal_embed_dim; ++i) {
        personal_embeddings_.Set({personal_id, i}, new_embedding.Get({i}));
    }
}

void TimeEncoding::SetTraining(bool training) {
    training_ = training;
}

// TimeEncodingUtils implementations
namespace TimeEncodingUtils {

TimeInfo CreateTimeInfo(const std::vector<float>& timestamps, float personal_delay) {
    TimeInfo info;
    info.timestamps = timestamps;
    info.personal_delay = personal_delay;
    info.time_scale = 1.0f;
    info.ComputeDeltas();
    if (personal_delay != 0.0f) {
        info.SetPersonalDelay(personal_delay);
    }
    return info;
}

TimeInfo CreateTimeInfoWithEnvironment(const std::vector<float>& timestamps,
                                     const std::vector<std::vector<float>>& environment_inputs,
                                     float personal_delay) {
    TimeInfo info = CreateTimeInfo(timestamps, personal_delay);
    
    // Initialize environment input history
    info.environment_input_history = environment_inputs;
    info.environment_memory_length = std::max(10, static_cast<int>(environment_inputs.size()));
    
    // Compute initial environment influence weights
    if (!environment_inputs.empty()) {
        info.environment_influence_weights.clear();
        info.environment_influence_weights.reserve(environment_inputs.size());
        
        for (size_t i = 0; i < environment_inputs.size(); ++i) {
            float age = static_cast<float>(environment_inputs.size() - 1 - i);
            float weight = std::exp(-age * info.environment_adaptation_rate);
            info.environment_influence_weights.push_back(weight);
        }
    }
    
    return info;
}

void UpdateTimeInfoEnvironment(TimeInfo& time_info,
                             const std::vector<float>& current_input,
                             float timestamp) {
    // Add timestamp if provided
    if (!time_info.timestamps.empty() && timestamp > time_info.timestamps.back()) {
        time_info.timestamps.push_back(timestamp);
        time_info.ComputeDeltas();
    }
    
    // Update environment input using existing method
    time_info.UpdateEnvironmentInput(current_input, timestamp);
}

TimeInfo CreateRegularTimeInfo(int seq_len, float time_step, float start_time) {
    std::vector<float> timestamps;
    timestamps.reserve(seq_len);
    
    for (int i = 0; i < seq_len; ++i) {
        timestamps.push_back(start_time + i * time_step);
    }
    
    return CreateTimeInfo(timestamps);
}

std::vector<float> GenerateTimestamps(int seq_len, float dt, float start_time) {
    std::vector<float> timestamps;
    timestamps.reserve(seq_len);
    
    for (int i = 0; i < seq_len; ++i) {
        timestamps.push_back(start_time + i * dt);
    }
    
    return timestamps;
}

} // namespace TimeEncodingUtils

} // namespace Core
} // namespace LwTT
