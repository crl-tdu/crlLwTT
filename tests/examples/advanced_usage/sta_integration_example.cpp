/**
 * @file sta_integration_example.cpp
 * @brief STA (Sense The Ambience) Architecture Integration Example
 * @version 1.0.0
 * @date 2025-05-25
 */

#include <LwTT/LwTT.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>

/**
 * @brief Example meta-evaluation function for concentration optimization
 */
class ConcentrationOptimizer : public crllwtt::Core::MetaEvaluationFunction {
private:
    float target_concentration_ = 0.8f; // Target concentration level (0-1)
    float stress_penalty_weight_ = 0.5f; // Weight for stress penalty
    
public:
    ConcentrationOptimizer(float target_concentration = 0.8f, float stress_penalty = 0.5f)
        : target_concentration_(target_concentration), stress_penalty_weight_(stress_penalty) {}
    
    float Evaluate(const crllwtt::Core::Tensor& predicted_state, 
                   const crllwtt::Core::Tensor* uncertainty = nullptr) const override {
        
        // Assume predicted_state contains [concentration, stress, fatigue, alertness]
        if (predicted_state.GetSize() < 4) {
            throw std::invalid_argument("Predicted state must have at least 4 dimensions");
        }
        
        const float* data = predicted_state.GetData();
        float concentration = data[0];
        float stress = data[1];
        float fatigue = data[2];
        float alertness = data[3];
        
        // Reward function: maximize concentration, minimize stress and fatigue, maintain alertness
        float reward = 0.0f;
        
        // Concentration reward (higher is better, target around 0.8)
        float concentration_error = std::abs(concentration - target_concentration_);
        reward += 1.0f - concentration_error;
        
        // Stress penalty (lower stress is better)
        reward -= stress_penalty_weight_ * stress;
        
        // Fatigue penalty (lower fatigue is better)
        reward -= 0.3f * fatigue;
        
        // Alertness reward (higher alertness is better)
        reward += 0.4f * alertness;
        
        // Uncertainty penalty if provided
        if (uncertainty) {
            float avg_uncertainty = 0.0f;
            for (int i = 0; i < uncertainty->GetSize(); ++i) {
                avg_uncertainty += uncertainty->GetData()[i];
            }
            avg_uncertainty /= uncertainty->GetSize();
            reward -= 0.1f * avg_uncertainty;
        }
        
        return reward;
    }
    
    crllwtt::Core::Tensor ComputeGradient(const crllwtt::Core::Tensor& predicted_state) const override {
        // Compute gradient of evaluation function w.r.t. predicted state
        crllwtt::Core::Tensor gradient({predicted_state.GetSize()});
        gradient.Fill(0.0f);
        
        if (predicted_state.GetSize() >= 4) {
            float* grad_data = gradient.GetData();
            const float* state_data = predicted_state.GetData();
            
            // Gradient w.r.t. concentration
            float concentration = state_data[0];
            if (concentration > target_concentration_) {
                grad_data[0] = -1.0f; // Decrease concentration if too high
            } else {
                grad_data[0] = 1.0f;  // Increase concentration if too low
            }
            
            // Gradient w.r.t. stress (always negative to minimize stress)
            grad_data[1] = -stress_penalty_weight_;
            
            // Gradient w.r.t. fatigue (negative to minimize fatigue)
            grad_data[2] = -0.3f;
            
            // Gradient w.r.t. alertness (positive to maximize alertness)
            grad_data[3] = 0.4f;
        }
        
        return gradient;
    }
    
    void SetTargetConcentration(float target) {
        target_concentration_ = std::max(0.0f, std::min(1.0f, target));
    }
};

/**
 * @brief Simulate human observable state (sensor data)
 */
crllwtt::Core::Tensor SimulateObservableState(int time_step) {
    // Simulate 8 observable parameters: heart rate, skin conductance, eye blink rate, 
    // posture stability, keystroke dynamics, mouse movement, facial expression, voice stress
    
    crllwtt::Core::Tensor obs_state({8});
    float* data = obs_state.GetData();
    
    // Add some realistic variation with time
    float time_factor = std::sin(time_step * 0.1f) * 0.1f;
    
    data[0] = 72.0f + time_factor * 10.0f;  // Heart rate (bpm)
    data[1] = 0.3f + time_factor * 0.1f;    // Skin conductance
    data[2] = 15.0f + time_factor * 5.0f;   // Eye blink rate (per minute)
    data[3] = 0.8f - std::abs(time_factor * 0.2f); // Posture stability
    data[4] = 200.0f + time_factor * 50.0f; // Keystroke interval (ms)
    data[5] = 0.5f + time_factor * 0.3f;    // Mouse movement variability
    data[6] = 0.6f + time_factor * 0.2f;    // Facial expression (valence)
    data[7] = 0.4f - time_factor * 0.1f;    // Voice stress indicator
    
    return obs_state;
}

/**
 * @brief Simulate actual human internal state (ground truth)
 */
crllwtt::Core::Tensor SimulateActualState(int time_step, const crllwtt::Core::Tensor& control_input) {
    // Simulate 4 internal state parameters: concentration, stress, fatigue, alertness
    
    crllwtt::Core::Tensor actual_state({4});
    float* data = actual_state.GetData();
    
    // Get control inputs (lighting, sound, temperature, notification frequency)
    const float* control_data = control_input.GetData();
    float lighting = control_data[0];
    float sound_volume = control_data[1];
    float temperature = control_data[2];
    float notification_freq = control_data[3];
    
    // Base state with time variation
    float time_factor = std::cos(time_step * 0.05f);
    
    // Concentration affected by lighting and notifications
    data[0] = 0.6f + 0.2f * lighting - 0.3f * notification_freq + 0.1f * time_factor;
    data[0] = std::max(0.0f, std::min(1.0f, data[0]));
    
    // Stress affected by sound and notifications
    data[1] = 0.3f + 0.4f * sound_volume + 0.5f * notification_freq - 0.1f * time_factor;
    data[1] = std::max(0.0f, std::min(1.0f, data[1]));
    
    // Fatigue increases over time, affected by lighting
    data[2] = 0.2f + time_step * 0.01f - 0.1f * lighting;
    data[2] = std::max(0.0f, std::min(1.0f, data[2]));
    
    // Alertness affected by temperature and sound
    data[3] = 0.7f + 0.2f * (1.0f - std::abs(temperature - 0.5f)) + 0.1f * sound_volume;
    data[3] = std::max(0.0f, std::min(1.0f, data[3]));
    
    return actual_state;
}

/**
 * @brief Print state information
 */
void PrintState(const std::string& label, const crllwtt::Core::Tensor& state) {
    std::cout << label << ": [";
    for (int i = 0; i < state.GetSize(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << state.GetData()[i];
        if (i < state.GetSize() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "=== STA (Sense The Ambience) Architecture Example ===" << std::endl;
    std::cout << "Demonstrating real-time adaptive control for human state optimization" << std::endl;
    
    // Initialize the library
    if (!crllwtt::Initialize()) {
        std::cerr << "Failed to initialize LwTT library" << std::endl;
        return -1;
    }
    
    try {
        // Configure STA Transformer
        auto sta_config = crllwtt::Core::STABuilder()
            .SetObservableStateDim(8)      // 8 sensor inputs
            .SetControllableInputDim(4)    // 4 environmental controls
            .SetPredictedStateDim(4)       // 4 internal states to predict
            .SetLearningRate(0.001f)
            .SetControlGain(0.1f)
            .EnableUncertainty(true, 3)    // Ensemble of 3 models
            .EnablePersonalAdaptation(true, 10)
            .SetMaxInferenceTime(1.0f)     // 1ms max inference time
            .SetBufferSize(1000)
            .Build();
        
        // Create concentration optimizer
        ConcentrationOptimizer optimizer(0.8f, 0.5f);
        
        // Simulation parameters
        const int simulation_steps = 100;
        const int person_id = 1;
        
        // Initial control input (lighting, sound, temperature, notifications)
        crllwtt::Core::Tensor control_input({4});
        control_input.GetData()[0] = 0.7f;  // Lighting
        control_input.GetData()[1] = 0.3f;  // Sound volume
        control_input.GetData()[2] = 0.6f;  // Temperature
        control_input.GetData()[3] = 0.2f;  // Notification frequency
        
        std::cout << "\nStarting simulation with " << simulation_steps << " time steps..." << std::endl;
        std::cout << "Person ID: " << person_id << std::endl;
        
        // Performance tracking
        std::vector<float> concentration_history;
        std::vector<float> stress_history;
        std::vector<float> reward_history;
        
        // Main simulation loop
        for (int step = 0; step < simulation_steps; ++step) {
            // Create time information
            std::vector<float> timestamps = {static_cast<float>(step)};
            auto time_info = crllwtt::Core::TimeEncodingUtils::CreateTimeInfo(timestamps, 0.1f);
            
            // Simulate observable state (sensor data)
            auto observable_state = SimulateObservableState(step);
            
            // Predict future internal state
            auto [predicted_state, uncertainty] = sta_config->PredictWithUncertainty(
                observable_state, control_input, &time_info, person_id);
            
            // Compute current reward
            float current_reward = optimizer.Evaluate(predicted_state, &uncertainty);
            
            // Simulate actual internal state (ground truth)
            auto actual_state = SimulateActualState(step, control_input);
            
            // Update model with real observation (online learning)
            sta_config->UpdateModel(observable_state, control_input, actual_state, 
                                   &time_info, person_id);
            
            // Compute optimal control for next step
            auto optimal_control = sta_config->ComputeOptimalControl(
                observable_state, control_input, optimizer, &time_info, person_id);
            
            // Store performance data
            concentration_history.push_back(actual_state.GetData()[0]);
            stress_history.push_back(actual_state.GetData()[1]);
            reward_history.push_back(current_reward);
            
            // Print progress every 10 steps
            if (step % 10 == 0) {
                std::cout << "\n--- Step " << step << " ---" << std::endl;
                PrintState("Observable", observable_state);
                PrintState("Control", control_input);
                PrintState("Predicted", predicted_state);
                PrintState("Actual", actual_state);
                PrintState("Uncertainty", uncertainty);
                PrintState("Optimal Control", optimal_control);
                std::cout << "Reward: " << std::fixed << std::setprecision(3) << current_reward << std::endl;
                std::cout << "Confidence: " << std::fixed << std::setprecision(3) 
                         << sta_config->GetPredictionConfidence(observable_state, control_input) << std::endl;
            }
            
            // Update control input for next iteration
            control_input = optimal_control;
            
            // Add small delay to simulate real-time processing
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Print final statistics
        std::cout << "\n=== Simulation Results ===" << std::endl;
        
        // Calculate averages
        float avg_concentration = 0.0f, avg_stress = 0.0f, avg_reward = 0.0f;
        for (size_t i = 0; i < concentration_history.size(); ++i) {
            avg_concentration += concentration_history[i];
            avg_stress += stress_history[i];
            avg_reward += reward_history[i];
        }
        avg_concentration /= concentration_history.size();
        avg_stress /= stress_history.size();
        avg_reward /= reward_history.size();
        
        std::cout << "Average Concentration: " << std::fixed << std::setprecision(3) << avg_concentration << std::endl;
        std::cout << "Average Stress: " << std::fixed << std::setprecision(3) << avg_stress << std::endl;
        std::cout << "Average Reward: " << std::fixed << std::setprecision(3) << avg_reward << std::endl;
        
        // Final concentration vs target
        float target_concentration = 0.8f;
        float final_concentration = concentration_history.back();
        float concentration_error = std::abs(final_concentration - target_concentration);
        
        std::cout << "Target Concentration: " << target_concentration << std::endl;
        std::cout << "Final Concentration: " << std::fixed << std::setprecision(3) << final_concentration << std::endl;
        std::cout << "Concentration Error: " << std::fixed << std::setprecision(3) << concentration_error << std::endl;
        
        // Model performance statistics
        std::cout << "\n" << sta_config->GetPerformanceStats() << std::endl;
        
        // Demonstrate sensitivity analysis
        std::cout << "\n=== Sensitivity Analysis ===" << std::endl;
        auto time_info_final = crllwtt::Core::TimeEncodingUtils::CreateTimeInfo({static_cast<float>(simulation_steps-1)}, 0.1f);
        auto sensitivity = sta_config->ComputeSensitivity(
            SimulateObservableState(simulation_steps-1), control_input, &time_info_final, person_id);
        
        std::cout << "Sensitivity matrix (∂state/∂control):" << std::endl;
        std::cout << "Controls: [Lighting, Sound, Temperature, Notifications]" << std::endl;
        std::cout << "States: [Concentration, Stress, Fatigue, Alertness]" << std::endl;
        for (int s = 0; s < 4; ++s) { // 4 predicted states
            std::cout << "State " << s << ": [";
            for (int u = 0; u < 4; ++u) { // 4 control inputs
                float sens_val = sensitivity.GetData()[s * 4 + u];
                std::cout << std::fixed << std::setprecision(3) << sens_val;
                if (u < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "\nSTA integration example completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        crllwtt::Cleanup();
        return -1;
    }
    
    // Cleanup
    crllwtt::Cleanup();
    return 0;
}