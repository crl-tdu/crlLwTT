/**
 * @file simple_transformer.cpp
 * @brief Simple example of LwTT usage for time series prediction
 * @version 1.0.0
 * @date 2025-05-25
 */

#include <LwTT/LwTT.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// Generate synthetic time series data (sine wave with noise)
std::pair<LwTT::Core::Tensor, std::vector<float>> GenerateTestData(int batch_size, int seq_len, int d_model) {
    LwTT::Core::Tensor data({batch_size, seq_len, d_model});
    std::vector<float> timestamps;

    // Generate timestamps
    for (int t = 0; t < seq_len; ++t) {
        timestamps.push_back(t * 0.01f); // 10ms intervals
    }

    // Fill data with synthetic patterns
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < d_model; ++d) {
                // Create sine wave with different frequencies and phases
                float freq = 0.1f + (d * 0.01f);
                float phase = b * 0.5f + d * 0.1f;
                float noise = ((rand() % 1000) / 1000.0f - 0.5f) * 0.1f;

                float value = sin(timestamps[t] * freq + phase) + noise;
                data.SetValue({b, t, d}, value);
            }
        }
    }

    return {data, timestamps};
}

int main() {
    std::cout << "=== LwTT Simple Transformer Example ===" << std::endl;

    // Initialize the library
    LwTT::LibraryConfig lib_config;
    lib_config.num_threads = 4;
    lib_config.enable_simd = true;
    lib_config.enable_logging = true;
    lib_config.log_level = 2; // Info level

    if (!LwTT::Initialize(lib_config)) {
        std::cerr << "Failed to initialize LwTT library" << std::endl;
        return -1;
    }

    std::cout << "LwTT version: " << LwTT::GetVersion() << std::endl;
    std::cout << "Build info: " << LwTT::GetBuildInfo() << std::endl;
    std::cout << "SIMD supported: " << (LwTT::IsSIMDSupported() ? "Yes" : "No") << std::endl;
    std::cout << "Hardware threads: " << LwTT::GetHardwareConcurrency() << std::endl;
    std::cout << std::endl;

    try {
        // Model configuration
        const int batch_size = 2;
        const int seq_len = 100;
        const int d_model = 64;
        const int prediction_steps = 5;

        // Create transformer using builder pattern
        std::cout << "Creating Transformer model..." << std::endl;
        auto transformer = LwTT::Core::TransformerBuilder()
            .SetModelDimension(d_model)
            .SetNumHeads(8)
            .SetNumLayers(4)
            .SetFeedForwardDim(256)
            .SetMaxSequenceLength(seq_len)
            .EnableTimeAwareness(true, 1.0f)
            .EnableSparseAttention(true, 0.1f)
            .SetDropoutRate(0.1f)
            .SetNumThreads(4)
            .SetMemoryPoolSize(128) // 128MB
            .Build();

        std::cout << "Model created successfully!" << std::endl;
        std::cout << "Parameter count: " << transformer->GetParameterCount() << std::endl;
        std::cout << "Memory usage: " << transformer->GetMemoryUsage() / 1024 / 1024 << " MB" << std::endl;
        std::cout << std::endl;

        // Generate test data
        std::cout << "Generating test data..." << std::endl;
        auto [input_data, timestamps] = GenerateTestData(batch_size, seq_len, d_model);
        std::cout << "Input data shape: " << input_data.ShapeString() << std::endl;

        // Create time information with personal delay compensation
        auto time_info = LwTT::Core::TimeEncodingUtils::CreateTimeInfo(timestamps, 0.05f);
        time_info.personal_delay = 0.05f; // 50ms personal delay

        // Set personal context
        time_info.task_difficulty.resize(seq_len, 1.0f); // Medium difficulty
        for (int i = 0; i < seq_len; ++i) {
            time_info.operation_modes.push_back(i % 3); // Cycling operation modes
        }

        std::cout << "Time info created with " << time_info.timestamps.size() << " timestamps" << std::endl;
        std::cout << "Personal delay: " << time_info.personal_delay << "s" << std::endl;
        std::cout << std::endl;

        // Enable profiling
        transformer->EnableProfiling(true);

        // Warm-up runs
        std::cout << "Performing warm-up runs..." << std::endl;
        for (int i = 0; i < 5; ++i) {
            auto warmup_output = transformer->Forward(input_data, nullptr, &time_info, 0);
        }

        // Benchmark forward pass
        std::cout << "Benchmarking forward pass..." << std::endl;
        const int num_iterations = 100;

        auto start_time = std::chrono::high_resolution_clock::now();

        LwTT::Core::Tensor output;
        for (int i = 0; i < num_iterations; ++i) {
            output = transformer->Forward(input_data, nullptr, &time_info, i % 10); // Rotate personal IDs
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        double avg_time_ms = duration.count() / 1000.0 / num_iterations;
        double throughput = (batch_size * num_iterations) / (duration.count() / 1000000.0);

        std::cout << "Output shape: " << output.ShapeString() << std::endl;
        std::cout << "Average inference time: " << avg_time_ms << " ms" << std::endl;
        std::cout << "Throughput: " << throughput << " samples/sec" << std::endl;
        std::cout << std::endl;

        // Test forward pass with uncertainty estimation
        std::cout << "Testing uncertainty estimation..." << std::endl;
        auto [pred_output, uncertainty] = transformer->ForwardWithUncertainty(input_data, nullptr, &time_info, 0);

        std::cout << "Prediction shape: " << pred_output.ShapeString() << std::endl;
        std::cout << "Uncertainty shape: " << uncertainty.ShapeString() << std::endl;

        // Calculate average uncertainty
        float avg_uncertainty = 0.0f;
        int total_elements = uncertainty.GetSize();
        for (int i = 0; i < total_elements; ++i) {
            avg_uncertainty += uncertainty.GetData()[i];
        }
        avg_uncertainty /= total_elements;
        std::cout << "Average uncertainty: " << avg_uncertainty << std::endl;
        std::cout << std::endl;

        // Test multi-step prediction
        std::cout << "Testing multi-step prediction..." << std::endl;
        auto multi_pred = transformer->PredictMultiStep(input_data, prediction_steps, nullptr, &time_info);
        std::cout << "Multi-step prediction shape: " << multi_pred.ShapeString() << std::endl;
        std::cout << std::endl;

        // Get attention weights for interpretability
        std::cout << "Analyzing attention patterns..." << std::endl;
        auto attention_weights = transformer->GetAttentionWeights();
        std::cout << "Number of attention layers: " << attention_weights.size() << std::endl;

        for (size_t layer = 0; layer < attention_weights.size(); ++layer) {
            std::cout << "Layer " << layer << " attention shape: " << attention_weights[layer].ShapeString() << std::endl;
        }
        std::cout << std::endl;

        // Optimize model for inference
        std::cout << "Optimizing model for inference..." << std::endl;
        transformer->OptimizeForInference(2); // Medium optimization level

        // Benchmark optimized model
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            output = transformer->Forward(input_data, nullptr, &time_info, 0);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        double optimized_time_ms = duration.count() / 1000.0 / num_iterations;
        double speedup = avg_time_ms / optimized_time_ms;

        std::cout << "Optimized inference time: " << optimized_time_ms << " ms" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << std::endl;

        // Show profiling results
        std::cout << "Profiling Results:" << std::endl;
        std::cout << transformer->GetProfilingResults() << std::endl;

        // Test model serialization
        std::cout << "Testing model serialization..." << std::endl;
        const std::string model_path = "/tmp/lwtt_test_model.bin";

        if (transformer->SaveModel(model_path)) {
            std::cout << "Model saved to: " << model_path << std::endl;

            // Create new model and load
            auto loaded_transformer = LwTT::Core::TransformerBuilder()
                .SetModelDimension(d_model)
                .SetNumHeads(8)
                .SetNumLayers(4)
                .Build();

            if (loaded_transformer->LoadModel(model_path)) {
                std::cout << "Model loaded successfully!" << std::endl;

                // Verify loaded model works
                auto loaded_output = loaded_transformer->Forward(input_data, nullptr, &time_info, 0);
                std::cout << "Loaded model output shape: " << loaded_output.ShapeString() << std::endl;
            } else {
                std::cout << "Failed to load model" << std::endl;
            }
        } else {
            std::cout << "Failed to save model" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "=== Example completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        LwTT::Cleanup();
        return -1;
    }

    // Cleanup
    LwTT::Cleanup();
    return 0;
}