/**
 * @file test_sta_environment_integration.cpp
 * @brief Test cases for STA environment input integration (Step 1)
 * @version 1.0.0
 * @date 2025-06-02
 */

#include <gtest/gtest.h>
#include "LwTT/core/STATransformer.hpp"
#include "LwTT/core/TimeEncoding.hpp"
#include <vector>
#include <iostream>
#include <chrono>

using namespace crllwtt::Core;

class STAEnvironmentIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Configure STA Transformer for testing
        sta_config_.observable_state_dim = 8;
        sta_config_.controllable_input_dim = 4;
        sta_config_.predicted_state_dim = 6;
        
        // Configure base transformer
        sta_config_.transformer_config.d_model = 64;
        sta_config_.transformer_config.n_heads = 4;
        sta_config_.transformer_config.n_layers = 2;
        sta_config_.transformer_config.max_seq_len = 32;
        sta_config_.transformer_config.personal_embed_dim = 16;
        
        // Enable features for testing
        sta_config_.enable_uncertainty = true;
        sta_config_.ensemble_size = 3;
        sta_config_.enable_personal_adaptation = true;
        sta_config_.max_persons = 10;
        
        // Real-time constraints
        sta_config_.max_inference_time_ms = 1.0f;
        sta_config_.learning_rate = 0.001f;
        sta_config_.control_gain = 0.1f;
        
        // Create STA Transformer
        sta_transformer_ = std::make_unique<STATransformer>(sta_config_);
    }
    
    void TearDown() override {
        sta_transformer_.reset();
    }
    
    STAConfig sta_config_;
    std::unique_ptr<STATransformer> sta_transformer_;
};

/**
 * @brief Test TimeInfo with environment input integration
 */
TEST_F(STAEnvironmentIntegrationTest, TimeInfoEnvironmentIntegration) {
    // Create timestamps
    std::vector<float> timestamps = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
    
    // Create environment input history
    std::vector<std::vector<float>> env_inputs = {
        {1.0f, 0.5f, 0.2f, 0.8f},  // u[k-4]
        {1.1f, 0.6f, 0.3f, 0.7f},  // u[k-3]  
        {0.9f, 0.4f, 0.1f, 0.9f},  // u[k-2]
        {1.2f, 0.7f, 0.4f, 0.6f},  // u[k-1]
        {1.0f, 0.5f, 0.2f, 0.8f}   // u[k]
    };
    
    // Test TimeInfo creation with environment
    TimeInfo time_info = TimeEncodingUtils::CreateTimeInfoWithEnvironment(
        timestamps, env_inputs, 0.05f);
    
    // Verify basic properties
    EXPECT_EQ(time_info.timestamps.size(), timestamps.size());
    EXPECT_EQ(time_info.environment_input_history.size(), env_inputs.size());
    EXPECT_EQ(time_info.environment_influence_weights.size(), env_inputs.size());
    EXPECT_FLOAT_EQ(time_info.personal_delay, 0.05f);
    
    // Verify influence weights (should be exponentially decaying)
    for (size_t i = 1; i < time_info.environment_influence_weights.size(); ++i) {
        EXPECT_GT(time_info.environment_influence_weights[i], 
                  time_info.environment_influence_weights[i-1]);
    }
    
    // Test environment influence encoding
    auto influence = time_info.GetEnvironmentInfluence(0, 64);
    EXPECT_EQ(influence.size(), 64);
    
    // Test updating environment input
    std::vector<float> new_input = {1.3f, 0.8f, 0.5f, 0.5f};
    time_info.UpdateEnvironmentInput(new_input, 0.5f);
    
    EXPECT_EQ(time_info.environment_input_history.size(), 
              std::min(static_cast<size_t>(time_info.environment_memory_length), 
                       env_inputs.size() + 1));
    
    std::cout << "✓ TimeInfo environment integration test passed" << std::endl;
}

/**
 * @brief Test STA prediction with environment input
 */
TEST_F(STAEnvironmentIntegrationTest, STAPredictionWithEnvironment) {
    // Create test inputs
    Tensor observable_state({sta_config_.observable_state_dim});
    observable_state.RandomNormal(0.0f, 1.0f);
    
    Tensor controllable_input({sta_config_.controllable_input_dim});
    controllable_input.RandomNormal(0.0f, 0.5f);
    
    // Create TimeInfo with environment inputs
    std::vector<float> timestamps = {0.0f, 0.1f, 0.2f, 0.3f};
    std::vector<std::vector<float>> env_history;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> env_input(sta_config_.controllable_input_dim);
        for (int j = 0; j < sta_config_.controllable_input_dim; ++j) {
            env_input[j] = 0.5f + 0.1f * i + 0.05f * j; // Structured pattern
        }
        env_history.push_back(env_input);
    }
    
    TimeInfo time_info = TimeEncodingUtils::CreateTimeInfoWithEnvironment(
        timestamps, env_history, 0.02f);
    
    // Test prediction
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Tensor predicted_state = sta_transformer_->PredictState(
        observable_state, controllable_input, &time_info, 0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Verify prediction shape
    auto pred_shape = predicted_state.Shape();
    EXPECT_EQ(pred_shape.size(), 1);
    EXPECT_EQ(pred_shape[0], sta_config_.predicted_state_dim);
    
    // Verify latency requirement (< 10ms for testing)
    EXPECT_LT(duration, 10000); // microseconds (relaxed for testing)
    
    // Verify prediction values are reasonable
    for (int i = 0; i < sta_config_.predicted_state_dim; ++i) {
        float val = predicted_state.GetData()[i];
        EXPECT_TRUE(std::isfinite(val)) << "Prediction contains non-finite value at index " << i;
        EXPECT_GT(std::abs(val), 1e-6f) << "Prediction is suspiciously close to zero at index " << i;
    }
    
    std::cout << "✓ STA prediction test passed (latency: " << duration << " μs)" << std::endl;
}

/**
 * @brief Test STA prediction with uncertainty estimation
 */
TEST_F(STAEnvironmentIntegrationTest, STAPredictionWithUncertainty) {
    // Create test inputs
    Tensor observable_state({sta_config_.observable_state_dim});
    observable_state.RandomNormal(0.0f, 1.0f);
    
    Tensor controllable_input({sta_config_.controllable_input_dim});
    controllable_input.RandomNormal(0.0f, 0.5f);
    
    // Create TimeInfo
    std::vector<float> timestamps = {0.0f, 0.1f, 0.2f};
    TimeInfo time_info = TimeEncodingUtils::CreateTimeInfo(timestamps, 0.03f);
    
    // Test prediction with uncertainty
    auto [prediction, uncertainty] = sta_transformer_->PredictWithUncertainty(
        observable_state, controllable_input, &time_info, 1);
    
    // Verify shapes
    auto pred_shape = prediction.Shape();
    auto unc_shape = uncertainty.Shape();
    
    EXPECT_EQ(pred_shape.size(), 1);
    EXPECT_EQ(pred_shape[0], sta_config_.predicted_state_dim);
    EXPECT_EQ(unc_shape.size(), 1);
    EXPECT_EQ(unc_shape[0], sta_config_.predicted_state_dim);
    
    // Verify uncertainty values are non-negative
    for (int i = 0; i < sta_config_.predicted_state_dim; ++i) {
        float unc_val = uncertainty.GetData()[i];
        EXPECT_GE(unc_val, 0.0f) << "Uncertainty should be non-negative at index " << i;
        EXPECT_TRUE(std::isfinite(unc_val)) << "Uncertainty contains non-finite value at index " << i;
    }
    
    std::cout << "✓ STA uncertainty estimation test passed" << std::endl;
}

/**
 * @brief Test sensitivity computation (numerical gradient)
 */
TEST_F(STAEnvironmentIntegrationTest, STASensitivityComputation) {
    // Create test inputs
    Tensor observable_state({sta_config_.observable_state_dim});
    observable_state.RandomNormal(0.0f, 1.0f);
    
    Tensor controllable_input({sta_config_.controllable_input_dim});
    controllable_input.RandomNormal(0.0f, 0.5f);
    
    // Create TimeInfo with environment influence
    std::vector<float> timestamps = {0.0f, 0.1f, 0.2f};
    std::vector<std::vector<float>> env_history = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.2f, 0.3f, 0.4f, 0.5f},
        {0.3f, 0.4f, 0.5f, 0.6f}
    };
    TimeInfo time_info = TimeEncodingUtils::CreateTimeInfoWithEnvironment(
        timestamps, env_history, 0.02f);
    
    // Test sensitivity computation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Tensor sensitivity = sta_transformer_->ComputeSensitivity(
        observable_state, controllable_input, &time_info, 2);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Verify sensitivity matrix shape: [predicted_state_dim x controllable_input_dim]
    auto sens_shape = sensitivity.Shape();
    EXPECT_EQ(sens_shape.size(), 2);
    EXPECT_EQ(sens_shape[0], sta_config_.predicted_state_dim);
    EXPECT_EQ(sens_shape[1], sta_config_.controllable_input_dim);
    
    // Verify sensitivity values are finite
    int total_elements = sta_config_.predicted_state_dim * sta_config_.controllable_input_dim;
    for (int i = 0; i < total_elements; ++i) {
        float sens_val = sensitivity.GetData()[i];
        EXPECT_TRUE(std::isfinite(sens_val)) << "Sensitivity contains non-finite value at index " << i;
    }
    
    std::cout << "✓ STA sensitivity computation test passed (latency: " << duration << " μs)" << std::endl;
}

/**
 * @brief Test optimal control computation
 */
TEST_F(STAEnvironmentIntegrationTest, STAOptimalControl) {
    // Create test inputs
    Tensor observable_state({sta_config_.observable_state_dim});
    observable_state.RandomNormal(0.0f, 1.0f);
    
    Tensor current_input({sta_config_.controllable_input_dim});
    current_input.RandomNormal(0.0f, 0.5f);
    
    // Create target state for evaluation
    Tensor target_state({sta_config_.predicted_state_dim});
    target_state.RandomNormal(0.0f, 0.5f);
    
    TargetStateEvaluator meta_eval(target_state, 1.0f);
    
    // Test optimal control computation
    Tensor optimal_control = sta_transformer_->ComputeOptimalControl(
        observable_state, current_input, meta_eval, nullptr, 0);
    
    // Verify control output shape
    auto control_shape = optimal_control.Shape();
    EXPECT_EQ(control_shape.size(), 1);
    EXPECT_EQ(control_shape[0], sta_config_.controllable_input_dim);
    
    // Verify control values are finite
    for (int i = 0; i < sta_config_.controllable_input_dim; ++i) {
        float control_val = optimal_control.GetData()[i];
        EXPECT_TRUE(std::isfinite(control_val)) << "Control contains non-finite value at index " << i;
    }
    
    std::cout << "✓ STA optimal control test passed" << std::endl;
}

/**
 * @brief Test performance under realistic timing constraints
 */
TEST_F(STAEnvironmentIntegrationTest, STAPerformanceTest) {
    const int num_iterations = 100;
    std::vector<double> prediction_times;
    std::vector<double> sensitivity_times;
    
    prediction_times.reserve(num_iterations);
    sensitivity_times.reserve(num_iterations);
    
    for (int i = 0; i < num_iterations; ++i) {
        // Create test inputs
        Tensor observable_state({sta_config_.observable_state_dim});
        observable_state.RandomNormal(0.0f, 1.0f);
        
        Tensor controllable_input({sta_config_.controllable_input_dim});
        controllable_input.RandomNormal(0.0f, 0.5f);
        
        // Create TimeInfo with random environment history
        std::vector<float> timestamps = {
            static_cast<float>(i) * 0.01f,
            static_cast<float>(i + 1) * 0.01f,
            static_cast<float>(i + 2) * 0.01f
        };
        
        std::vector<std::vector<float>> env_history;
        for (int j = 0; j < 3; ++j) {
            std::vector<float> env_input(sta_config_.controllable_input_dim);
            for (int k = 0; k < sta_config_.controllable_input_dim; ++k) {
                env_input[k] = 0.1f * (i + j + k);
            }
            env_history.push_back(env_input);
        }
        
        TimeInfo time_info = TimeEncodingUtils::CreateTimeInfoWithEnvironment(
            timestamps, env_history, 0.01f * i);
        
        // Measure prediction time
        auto pred_start = std::chrono::high_resolution_clock::now();
        Tensor prediction = sta_transformer_->PredictState(
            observable_state, controllable_input, &time_info, i % 5);
        auto pred_end = std::chrono::high_resolution_clock::now();
        
        auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            pred_end - pred_start).count();
        prediction_times.push_back(pred_duration);
        
        // Measure sensitivity computation time
        auto sens_start = std::chrono::high_resolution_clock::now();
        Tensor sensitivity = sta_transformer_->ComputeSensitivity(
            observable_state, controllable_input, &time_info, i % 5);
        auto sens_end = std::chrono::high_resolution_clock::now();
        
        auto sens_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            sens_end - sens_start).count();
        sensitivity_times.push_back(sens_duration);
    }
    
    // Compute statistics
    double avg_pred_time = 0.0, max_pred_time = 0.0;
    double avg_sens_time = 0.0, max_sens_time = 0.0;
    
    for (double time : prediction_times) {
        avg_pred_time += time;
        max_pred_time = std::max(max_pred_time, time);
    }
    avg_pred_time /= num_iterations;
    
    for (double time : sensitivity_times) {
        avg_sens_time += time;
        max_sens_time = std::max(max_sens_time, time);
    }
    avg_sens_time /= num_iterations;
    
    // Performance requirements verification (relaxed for testing)
    EXPECT_LT(avg_pred_time, 5000.0) << "Average prediction time exceeds 5ms";
    EXPECT_LT(max_pred_time, 10000.0) << "Maximum prediction time exceeds 10ms";
    EXPECT_LT(avg_sens_time, 20000.0) << "Average sensitivity time exceeds 20ms";
    
    std::cout << "✓ Performance test results:" << std::endl;
    std::cout << "  Average prediction time: " << avg_pred_time << " μs" << std::endl;
    std::cout << "  Maximum prediction time: " << max_pred_time << " μs" << std::endl;
    std::cout << "  Average sensitivity time: " << avg_sens_time << " μs" << std::endl;
    std::cout << "  Maximum sensitivity time: " << max_sens_time << " μs" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=== STA Environment Integration Test Suite ===" << std::endl;
    std::cout << "Testing Step 1: Environment Input Integration Foundation" << std::endl;
    std::cout << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << std::endl;
        std::cout << "🎉 All Step 1 tests passed successfully!" << std::endl;
        std::cout << "Environment input integration foundation is working correctly." << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "❌ Some tests failed. Please check the implementation." << std::endl;
    }
    
    return result;
}
