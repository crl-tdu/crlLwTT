/**
 * @file TimeEncoding.hpp
 * @brief Time-aware Positional Encoding for Lightweight Transformer
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_CORE_TIME_ENCODING_HPP
#define LWTT_CORE_TIME_ENCODING_HPP

#include "Tensor.hpp"
#include <vector>
#include <unordered_map>
#include <memory>

namespace crllwtt {
namespace Core {

/**
 * @brief Time information structure for time-aware encoding
 */
struct TimeInfo {
    std::vector<float> timestamps;        ///< Timestamps for each position
    std::vector<float> time_deltas;       ///< Time deltas between positions
    float global_time_offset = 0.0f;     ///< Global time offset
    float time_scale = 1.0f;              ///< Time scaling factor
    int sequence_id = -1;                 ///< Sequence identifier

    // Personal delay compensation
    float personal_delay = 0.0f;          ///< Individual response delay (tau)
    std::vector<float> delay_weights;     ///< Delay-dependent weights

    // Environmental input temporal influence (STA extension)
    std::vector<std::vector<float>> environment_input_history; ///< History of environment inputs u[k-N:k]
    std::vector<float> environment_influence_weights;          ///< Temporal influence weights ψ(u_influence)
    float environment_adaptation_rate = 0.01f;                ///< Adaptation rate for environment influence
    int environment_memory_length = 10;                       ///< Number of past environment inputs to consider

    // Contextual information
    std::vector<float> task_difficulty;   ///< Task difficulty at each step
    std::vector<int> operation_modes;     ///< Operation mode indicators

    TimeInfo() = default;

    TimeInfo(const std::vector<float>& ts) : timestamps(ts) {
        ComputeDeltas();
    }

    void ComputeDeltas() {
        if (timestamps.size() <= 1) return;

        time_deltas.clear();
        time_deltas.reserve(timestamps.size() - 1);

        for (size_t i = 1; i < timestamps.size(); ++i) {
            time_deltas.push_back(timestamps[i] - timestamps[i-1]);
        }
    }

    void SetPersonalDelay(float delay) {
        personal_delay = delay;
        ComputeDelayWeights();
    }

    /**
     * @brief Update environment input history and compute influence weights
     * @param current_input Current environment input vector
     * @param timestamp Current timestamp
     */
    void UpdateEnvironmentInput(const std::vector<float>& current_input, [[maybe_unused]] float timestamp) {
        // Add current input to history
        environment_input_history.push_back(current_input);
        
        // Maintain memory length
        if (environment_input_history.size() > static_cast<size_t>(environment_memory_length)) {
            environment_input_history.erase(environment_input_history.begin());
        }
        
        ComputeEnvironmentInfluenceWeights();
    }

    /**
     * @brief Get environment influence encoding for time-aware transformer
     * @param pos Position in sequence
     * @param d_model Model dimension
     * @return Environment influence encoding
     */
    std::vector<float> GetEnvironmentInfluence(int pos, int d_model) const {
        std::vector<float> influence(d_model, 0.0f);
        
        if (environment_input_history.empty() || environment_influence_weights.empty()) {
            return influence;
        }
        
        // Weighted combination of past environment inputs
        for (size_t i = 0; i < environment_input_history.size() && i < environment_influence_weights.size(); ++i) {
            float weight = environment_influence_weights[i];
            const auto& input_vec = environment_input_history[i];
            
            for (size_t j = 0; j < input_vec.size() && j < static_cast<size_t>(d_model); ++j) {
                // Apply sinusoidal encoding with environment influence
                float env_influence = input_vec[j] * weight;
                influence[j] += std::sin(pos / std::pow(10000.0f, 2.0f * j / d_model) + env_influence);
                
                if (j + 1 < static_cast<size_t>(d_model)) {
                    influence[j + 1] += std::cos(pos / std::pow(10000.0f, 2.0f * j / d_model) + env_influence);
                }
            }
        }
        
        return influence;
    }

private:
    void ComputeDelayWeights() {
        delay_weights.clear();
        delay_weights.reserve(timestamps.size());

        for (size_t i = 0; i < timestamps.size(); ++i) {
            float adjusted_time = timestamps[i] - personal_delay;
            float weight = std::exp(-std::abs(adjusted_time) / time_scale);
            delay_weights.push_back(weight);
        }
    }

    void ComputeEnvironmentInfluenceWeights() {
        environment_influence_weights.clear();
        environment_influence_weights.reserve(environment_input_history.size());

        // Compute exponential decay weights for environment influence
        // More recent inputs have higher influence
        for (size_t i = 0; i < environment_input_history.size(); ++i) {
            float age = static_cast<float>(environment_input_history.size() - 1 - i);
            float weight = std::exp(-age * environment_adaptation_rate);
            environment_influence_weights.push_back(weight);
        }
    }
};

/**
 * @brief Configuration for time-aware encoding
 */
struct TimeEncodingConfig {
    int d_model = 256;                    ///< Model dimension
    int max_seq_len = 512;                ///< Maximum sequence length
    float time_scale = 1.0f;              ///< Time scaling factor
    bool enable_time_encoding = true;     ///< Enable time encoding
    bool enable_time_scaling = false;     ///< Enable time scaling
    int num_time_scales = 1;              ///< Number of time scales
    int personal_embed_dim = 32;          ///< Personal embedding dimension
    int max_persons = 1000;               ///< Maximum number of persons
};

/**
 * @brief Time-aware positional encoding implementation
 */
class TimeEncoding {
public:
    /**
     * @brief Constructor
     * @param config Configuration
     */
    explicit TimeEncoding(const TimeEncodingConfig& config);

    /**
     * @brief Destructor
     */
    ~TimeEncoding();

    /**
     * @brief Apply time-aware encoding to input
     * @param input Input tensor [batch_size, seq_len, d_model]
     * @param time_info Time information (optional)
     * @param personal_id Personal ID for personalization (optional)
     * @return Encoded tensor
     */
    Tensor Apply(const Tensor& input,
                 const TimeInfo* time_info = nullptr,
                 int personal_id = -1) const;

    /**
     * @brief Generate standard positional encoding
     * @param seq_len Sequence length
     * @param d_model Model dimension
     * @return Positional encoding tensor
     */
    Tensor GeneratePositionalEncoding(int seq_len, int d_model) const;

    /**
     * @brief Generate time-aware encoding
     * @param time_info Time information
     * @param seq_len Sequence length
     * @return Time-aware encoding tensor
     */
    Tensor GenerateTimeAwareEncoding(const TimeInfo& time_info, int seq_len) const;

    /**
     * @brief Generate personal embedding
     * @param personal_id Personal ID
     * @return Personal embedding tensor
     */
    Tensor GetPersonalEmbedding(int personal_id) const;

    /**
     * @brief Update personal embedding
     * @param personal_id Personal ID
     * @param embedding New embedding
     */
    void UpdatePersonalEmbedding(int personal_id, const Tensor& embedding);

    /**
     * @brief Enable/disable training mode
     * @param training Training mode flag
     */
    void SetTraining(bool training);

    /**
     * @brief Get configuration
     * @return Configuration
     */
    const TimeEncodingConfig& GetConfig() const { return config_; }

    /**
     * @brief Save encoding parameters
     * @param filepath File path
     * @return true if successful
     */
    bool SaveParameters(const std::string& filepath) const;

    /**
     * @brief Load encoding parameters
     * @param filepath File path
     * @return true if successful
     */
    bool LoadParameters(const std::string& filepath);

private:
    TimeEncodingConfig config_;
    bool training_ = false;

    // Encoding tables
    Tensor positional_encoding_;        ///< Positional encoding table
    Tensor time_scale_weights_;         ///< Time scale weights
    Tensor personal_embeddings_;        ///< Personal embeddings
    Tensor personal_projection_;        ///< Personal projection layer

    // Private methods
    void InitializeEncodingTables();
    void AddPositionalEncoding(Tensor& input, int seq_len) const;
    void AddTimeAwareEncoding(Tensor& input, const TimeInfo& time_info) const;
    void AddPersonalEncoding(Tensor& input, int personal_id) const;
    void ApplyDelayCompensation(Tensor& input, float personal_delay) const;
};

/**
 * @brief Utility functions for time encoding
 */
namespace TimeEncodingUtils {

    /**
     * @brief Create TimeInfo from timestamp vector
     * @param timestamps Vector of timestamps
     * @param personal_delay Personal delay parameter
     * @return TimeInfo structure
     */
    TimeInfo CreateTimeInfo(const std::vector<float>& timestamps,
                           float personal_delay = 0.0f);

    /**
     * @brief Create TimeInfo for regular time intervals
     * @param seq_len Sequence length
     * @param time_step Time step
     * @param start_time Start time
     * @return TimeInfo structure
     */
    TimeInfo CreateRegularTimeInfo(int seq_len,
                                  float time_step = 1.0f,
                                  float start_time = 0.0f);

    /**
     * @brief Estimate personal delay from cross-correlation
     * @param events Event timestamps
     * @param responses Response timestamps
     * @return Estimated delay
     */
    float EstimatePersonalDelay(const std::vector<float>& events,
                               const std::vector<float>& responses);

    /**
     * @brief Compute cross-correlation between two time series
     * @param x First time series
     * @param y Second time series
     * @param max_lag Maximum lag to consider
     * @return Cross-correlation values
     */
    std::vector<float> CrossCorrelation(const std::vector<float>& x,
                                       const std::vector<float>& y,
                                       int max_lag);

    /**
     * @brief Apply temporal smoothing to time series
     * @param data Input data
     * @param window_size Smoothing window size
     * @return Smoothed data
     */
    std::vector<float> TemporalSmoothing(const std::vector<float>& data,
                                        int window_size);

    /**
     * @brief Detect concept drift in time series
     * @param data Input data
     * @param window_size Detection window size
     * @param threshold Drift detection threshold
     * @return Drift detection points
     */
    std::vector<int> DetectConceptDrift(const std::vector<float>& data,
                                       int window_size,
                                       float threshold);

    /**
     * @brief Create TimeInfo with environment input integration
     * @param timestamps Vector of timestamps
     * @param environment_inputs History of environment inputs
     * @param personal_delay Personal delay parameter
     * @return TimeInfo structure with environment influence
     */
    TimeInfo CreateTimeInfoWithEnvironment(const std::vector<float>& timestamps,
                                         const std::vector<std::vector<float>>& environment_inputs,
                                         float personal_delay = 0.0f);

    /**
     * @brief Update TimeInfo with new environment input
     * @param time_info TimeInfo to update
     * @param current_input Current environment input
     * @param timestamp Current timestamp
     */
    void UpdateTimeInfoEnvironment(TimeInfo& time_info,
                                 const std::vector<float>& current_input,
                                 float timestamp);

    /**
     * @brief Interpolate missing timestamps
     * @param timestamps Original timestamps (may contain NaN)
     * @param method Interpolation method ("linear", "cubic")
     * @return Interpolated timestamps
     */
    std::vector<float> InterpolateTimestamps(const std::vector<float>& timestamps,
                                           const std::string& method = "linear");

} // namespace TimeEncodingUtils

} // namespace Core
} // namespace LwTT

#endif // LWTT_CORE_TIME_ENCODING_HPP