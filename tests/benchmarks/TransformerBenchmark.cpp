#include <benchmark/benchmark.h>
#include <LwTT/core/Transformer.hpp>
#include <LwTT/core/TimeEncoding.hpp>
#include <vector>
#include <random>

// ランダムテンソルを生成するヘルパー関数
LwTT::Core::Tensor GenerateRandomTensor(const std::vector<int>& shape) {
    LwTT::Core::Tensor tensor(shape);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < tensor.GetSize(); ++i) {
        tensor.GetData()[i] = dist(gen);
    }
    
    return tensor;
}

// TimeInfoを生成するヘルパー関数
LwTT::Core::TimeInfo GenerateTimeInfo(int seq_len) {
    std::vector<float> timestamps;
    for (int i = 0; i < seq_len; ++i) {
        timestamps.push_back(i * 0.01f);
    }
    return LwTT::Core::TimeEncodingUtils::CreateTimeInfo(timestamps);
}

// Transformerの順伝播のベンチマーク（小）
static void BM_TransformerForward_Small(benchmark::State& state) {
    // 小さいモデルの設定
    LwTT::Core::TransformerConfig config;
    config.d_model = 64;
    config.n_heads = 4;
    config.n_layers = 2;
    config.max_seq_len = 32;
    config.enable_time_encoding = true;
    config.use_sparse_attention = false;
    
    auto transformer = std::make_unique<LwTT::Core::Transformer>(config);
    transformer->OptimizeForInference(3);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(transformer->Forward(input, nullptr, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// Transformerの順伝播のベンチマーク（中）
static void BM_TransformerForward_Medium(benchmark::State& state) {
    // 中程度のモデルの設定
    LwTT::Core::TransformerConfig config;
    config.d_model = 128;
    config.n_heads = 8;
    config.n_layers = 4;
    config.max_seq_len = 64;
    config.enable_time_encoding = true;
    config.use_sparse_attention = false;
    
    auto transformer = std::make_unique<LwTT::Core::Transformer>(config);
    transformer->OptimizeForInference(3);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(transformer->Forward(input, nullptr, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// スパースアテンションのベンチマーク
static void BM_SparseAttention(benchmark::State& state) {
    // モデルの設定
    LwTT::Core::TransformerConfig config;
    config.d_model = 128;
    config.n_heads = 8;
    config.n_layers = 4;
    config.max_seq_len = state.range(1);
    config.enable_time_encoding = true;
    config.use_sparse_attention = true;
    config.sparsity_ratio = 0.1f; // 10%のスパース化
    
    auto transformer = std::make_unique<LwTT::Core::Transformer>(config);
    transformer->OptimizeForInference(3);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(transformer->Forward(input, nullptr, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// 不確実性推定のベンチマーク
static void BM_UncertaintyEstimation(benchmark::State& state) {
    // モデルの設定
    LwTT::Core::TransformerConfig config;
    config.d_model = 128;
    config.n_heads = 8;
    config.n_layers = 4;
    config.max_seq_len = state.range(1);
    config.enable_time_encoding = true;
    config.use_sparse_attention = false;
    
    auto transformer = std::make_unique<LwTT::Core::Transformer>(config);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(transformer->ForwardWithUncertainty(input, nullptr, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// マルチステップ予測のベンチマーク
static void BM_MultiStepPrediction(benchmark::State& state) {
    // モデルの設定
    LwTT::Core::TransformerConfig config;
    config.d_model = 128;
    config.n_heads = 8;
    config.n_layers = 4;
    config.max_seq_len = state.range(1);
    config.enable_time_encoding = true;
    config.use_sparse_attention = false;
    
    auto transformer = std::make_unique<LwTT::Core::Transformer>(config);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(transformer->PredictMultiStep(input, 5, nullptr, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// 時間エンコーディングのベンチマーク
static void BM_TimeEncoding(benchmark::State& state) {
    // 時間エンコーディングの設定
    LwTT::Core::TimeEncodingConfig config;
    config.d_model = 128;
    config.max_seq_len = static_cast<int>(state.range(1));
    config.time_scale = 1.0f;
    config.enable_time_encoding = true;
    config.enable_time_scaling = true;
    config.personal_embed_dim = 32;
    
    auto time_encoding = std::make_unique<LwTT::Core::TimeEncoding>(config);
    
    // 入力テンソルの生成
    auto input = GenerateRandomTensor({static_cast<int>(state.range(0)), static_cast<int>(state.range(1)), config.d_model});
    auto time_info = GenerateTimeInfo(static_cast<int>(state.range(1)));
    
    // ベンチマーク実行
    for (auto _ : state) {
        benchmark::DoNotOptimize(time_encoding->Apply(input, &time_info));
    }
    
    // メトリクスの設定
    state.SetItemsProcessed(state.iterations() * state.range(0));
    state.SetBytesProcessed(state.iterations() * state.range(0) * state.range(1) * config.d_model * sizeof(float));
}

// ベンチマークの設定
// 第一引数: バッチサイズ、第二引数: シーケンス長
BENCHMARK(BM_TransformerForward_Small)->Args({1, 32})->Args({8, 32})->Args({16, 32});
BENCHMARK(BM_TransformerForward_Medium)->Args({1, 64})->Args({8, 64})->Args({16, 64});
BENCHMARK(BM_SparseAttention)->Args({1, 128})->Args({8, 128})->Args({1, 256})->Args({8, 256});
BENCHMARK(BM_UncertaintyEstimation)->Args({1, 64})->Args({8, 64});
BENCHMARK(BM_MultiStepPrediction)->Args({1, 64})->Args({8, 64});
BENCHMARK(BM_TimeEncoding)->Args({1, 128})->Args({8, 128});

// メインエントリポイント
BENCHMARK_MAIN();
