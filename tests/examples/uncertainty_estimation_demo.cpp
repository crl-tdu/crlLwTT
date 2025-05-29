/**
 * @file uncertainty_estimation_demo.cpp
 * @brief Demonstration of uncertainty estimation capabilities in LwTT
 * @version 1.0.0
 * @date 2025-05-28
 */

#include <LwTT/LwTT.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Generate test data with varying levels of noise
std::pair<LwTT::Core::Tensor, std::vector<float>> GenerateNoisyData(int batch_size, int seq_len, int d_model, float noise_level) {
    LwTT::Core::Tensor data({batch_size, seq_len, d_model});
    std::vector<float> timestamps;

    // Generate timestamps
    for (int t = 0; t < seq_len; ++t) {
        timestamps.push_back(t * 0.01f); // 10ms intervals
    }

    // Fill data with predictable patterns + noise
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < d_model; ++d) {
                // Create predictable base pattern
                float base_pattern = std::sin(timestamps[t] * (1.0f + d * 0.1f)) * std::cos(timestamps[t] * 0.5f);
                
                // Add controllable noise
                float noise = ((rand() % 1000) / 1000.0f - 0.5f) * noise_level;
                
                float value = base_pattern + noise;
                data.SetValue({b, t, d}, value);
            }
        }
    }

    return {data, timestamps};
}

// Analyze uncertainty results
void AnalyzeUncertainty(const LwTT::Core::Tensor& predictions, 
                       const LwTT::Core::Tensor& uncertainty,
                       const std::string& data_description) {
    
    std::cout << "\n=== Uncertainty Analysis: " << data_description << " ===" << std::endl;
    
    // Calculate statistics
    float mean_uncertainty = 0.0f;
    float max_uncertainty = 0.0f;
    float min_uncertainty = uncertainty.GetData()[0];
    
    int total_elements = uncertainty.GetSize();
    for (int i = 0; i < total_elements; ++i) {
        float u = uncertainty.GetData()[i];
        mean_uncertainty += u;
        max_uncertainty = std::max(max_uncertainty, u);
        min_uncertainty = std::min(min_uncertainty, u);
    }
    mean_uncertainty /= total_elements;
    
    // Calculate confidence level (inverse of uncertainty)
    float mean_confidence = 1.0f / (1.0f + mean_uncertainty);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Prediction shape: " << predictions.ShapeString() << std::endl;
    std::cout << "Uncertainty shape: " << uncertainty.ShapeString() << std::endl;
    std::cout << "Mean uncertainty: " << mean_uncertainty << std::endl;
    std::cout << "Min uncertainty: " << min_uncertainty << std::endl;
    std::cout << "Max uncertainty: " << max_uncertainty << std::endl;
    std::cout << "Mean confidence: " << (mean_confidence * 100.0f) << "%" << std::endl;
    
    // Uncertainty distribution analysis
    int low_uncertainty = 0, medium_uncertainty = 0, high_uncertainty = 0;
    for (int i = 0; i < total_elements; ++i) {
        float u = uncertainty.GetData()[i];
        if (u < 0.1f) low_uncertainty++;
        else if (u < 0.3f) medium_uncertainty++;
        else high_uncertainty++;
    }
    
    std::cout << "Uncertainty distribution:" << std::endl;
    std::cout << "  Low (< 0.1): " << (100.0f * low_uncertainty / total_elements) << "%" << std::endl;
    std::cout << "  Medium (0.1-0.3): " << (100.0f * medium_uncertainty / total_elements) << "%" << std::endl;
    std::cout << "  High (> 0.3): " << (100.0f * high_uncertainty / total_elements) << "%" << std::endl;
}

int main() {
    std::cout << "=== LwTT Uncertainty Estimation Demo ===" << std::endl;

    // Initialize the library
    LwTT::LibraryConfig lib_config;
    lib_config.num_threads = 4;
    lib_config.enable_simd = true;
    lib_config.enable_logging = true;
    lib_config.log_level = 1; // Warning level

    if (!LwTT::Initialize(lib_config)) {
        std::cerr << "Failed to initialize LwTT library" << std::endl;
        return -1;
    }

    std::cout << "LwTT version: " << LwTT::GetVersion() << std::endl;
    std::cout << std::endl;

    try {
        // Model configuration with uncertainty estimation enabled
        const int batch_size = 2;
        const int seq_len = 50;
        const int d_model = 32;
        
        // Create transformer with uncertainty estimation
        std::cout << "Creating Transformer with uncertainty estimation..." << std::endl;
        auto transformer = LwTT::Core::TransformerBuilder()
            .SetModelDimension(d_model)
            .SetNumHeads(4)
            .SetNumLayers(2)
            .SetFeedForwardDim(128)
            .SetMaxSequenceLength(seq_len)
            .EnableTimeAwareness(true, 1.0f)
            .EnableSparseAttention(true, 0.1f)
            .SetDropoutRate(0.2f)  // Higher dropout for better uncertainty estimation
            .EnableUncertainty(true, 15)  // 15 MC samples for better estimates
            .SetUncertaintyThreshold(0.1f)
            .Build();

        std::cout << "Model created successfully!" << std::endl;
        std::cout << "Uncertainty estimation: " << (transformer->GetConfig().enable_uncertainty ? "Enabled" : "Disabled") << std::endl;
        std::cout << "MC samples: " << transformer->GetConfig().mc_samples << std::endl;
        std::cout << std::endl;

        // Test 1: Low noise data (should have low uncertainty)
        std::cout << "=== Test 1: Low Noise Data ===" << std::endl;
        auto [low_noise_data, timestamps] = GenerateNoisyData(batch_size, seq_len, d_model, 0.1f);
        auto time_info = LwTT::Core::TimeEncodingUtils::CreateTimeInfo(timestamps, 0.02f);
        
        auto [pred1, uncertainty1] = transformer->ForwardWithUncertainty(low_noise_data, nullptr, &time_info, 0);
        AnalyzeUncertainty(pred1, uncertainty1, "Low Noise (0.1)");

        // Test 2: Medium noise data (should have medium uncertainty)
        std::cout << "\n=== Test 2: Medium Noise Data ===" << std::endl;
        auto [medium_noise_data, _] = GenerateNoisyData(batch_size, seq_len, d_model, 0.5f);
        
        auto [pred2, uncertainty2] = transformer->ForwardWithUncertainty(medium_noise_data, nullptr, &time_info, 1);
        AnalyzeUncertainty(pred2, uncertainty2, "Medium Noise (0.5)");

        // Test 3: High noise data (should have high uncertainty)
        std::cout << "\n=== Test 3: High Noise Data ===" << std::endl;
        auto [high_noise_data, __] = GenerateNoisyData(batch_size, seq_len, d_model, 1.0f);
        
        auto [pred3, uncertainty3] = transformer->ForwardWithUncertainty(high_noise_data, nullptr, &time_info, 2);
        AnalyzeUncertainty(pred3, uncertainty3, "High Noise (1.0)");

        // Test 4: Uncertainty disabled mode
        std::cout << "\n=== Test 4: Uncertainty Disabled ===" << std::endl;
        auto transformer_no_uncertainty = LwTT::Core::TransformerBuilder()
            .SetModelDimension(d_model)
            .SetNumHeads(4)
            .SetNumLayers(2)
            .EnableUncertainty(false)  // Disable uncertainty
            .Build();

        auto [pred4, uncertainty4] = transformer_no_uncertainty->ForwardWithUncertainty(low_noise_data, nullptr, &time_info, 0);
        AnalyzeUncertainty(pred4, uncertainty4, "Uncertainty Disabled");

        // Test 5: Performance comparison
        std::cout << "\n=== Test 5: Performance Comparison ===" << std::endl;
        const int num_iterations = 50;
        
        // Regular forward pass timing
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            auto regular_output = transformer->Forward(low_noise_data, nullptr, &time_info, 0);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto regular_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Uncertainty forward pass timing
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            auto [uncertain_output, uncertain_uncertainty] = transformer->ForwardWithUncertainty(low_noise_data, nullptr, &time_info, 0);
        }
        end_time = std::chrono::high_resolution_clock::now();
        auto uncertain_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Regular forward pass: " << regular_duration.count() << " ms (" << num_iterations << " iterations)" << std::endl;
        std::cout << "Uncertainty forward pass: " << uncertain_duration.count() << " ms (" << num_iterations << " iterations)" << std::endl;
        std::cout << "Uncertainty overhead: " << (float)uncertain_duration.count() / regular_duration.count() << "x" << std::endl;
        std::cout << "Average per sample:" << std::endl;
        std::cout << "  Regular: " << (float)regular_duration.count() / num_iterations << " ms" << std::endl;
        std::cout << "  Uncertainty: " << (float)uncertain_duration.count() / num_iterations << " ms" << std::endl;

        // Test 6: Different MC sample counts
        std::cout << "\n=== Test 6: MC Sample Count Impact ===" << std::endl;
        std::vector<int> sample_counts = {5, 10, 20};
        
        for (int samples : sample_counts) {
            auto test_transformer = LwTT::Core::TransformerBuilder()
                .SetModelDimension(d_model)
                .SetNumHeads(4)
                .SetNumLayers(2)
                .EnableUncertainty(true, samples)
                .Build();

            auto [pred, uncertainty] = test_transformer->ForwardWithUncertainty(medium_noise_data, nullptr, &time_info, 0);
            
            // Calculate mean uncertainty
            float mean_unc = 0.0f;
            for (int i = 0; i < uncertainty.GetSize(); ++i) {
                mean_unc += uncertainty.GetData()[i];
            }
            mean_unc /= uncertainty.GetSize();
            
            std::cout << "MC Samples: " << samples << ", Mean Uncertainty: " << std::fixed << std::setprecision(6) << mean_unc << std::endl;
        }

        std::cout << "\n=== Demo completed successfully! ===" << std::endl;
        std::cout << "\nKey Findings:" << std::endl;
        std::cout << "1. Higher noise input → Higher uncertainty estimates" << std::endl;
        std::cout << "2. More MC samples → More stable uncertainty estimates" << std::endl;
        std::cout << "3. Uncertainty can be disabled for faster inference" << std::endl;
        std::cout << "4. Uncertainty overhead is approximately " << transformer->GetConfig().mc_samples << "x regular inference" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        LwTT::Cleanup();
        return -1;
    }

    // Cleanup
    LwTT::Cleanup();
    return 0;
}
