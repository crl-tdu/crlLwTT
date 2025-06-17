# crlLwTT API リファレンス

> **完全なAPIドキュメント** - 全クラス、メソッド、設定オプションの詳細仕様

## 📖 目次

- [基本ライブラリ](#基本ライブラリ)
- [コアクラス](#コアクラス)
- [最適化モジュール](#最適化モジュール)
- [ユーティリティ](#ユーティリティ)
- [設定と初期化](#設定と初期化)
- [エラーハンドリング](#エラーハンドリング)

---

## 🚀 基本ライブラリ

### LwTT 名前空間

```cpp
namespace crllwtt {
    // 初期化・終了処理
    bool Initialize(const GlobalConfig& config = GlobalConfig{});
    void Cleanup();
    
    // バージョン情報
    std::string GetVersion();
    std::string GetBuildInfo();
    
    // ログ設定
    void SetLogLevel(LogLevel level);
    void EnableFileLogging(const std::string& filename);
}
```

### 初期化設定

```cpp
struct GlobalConfig {
    // 並列処理設定
    int num_threads = -1;                    // -1で自動検出
    bool enable_openmp = true;               // OpenMP使用
    
    // メモリ設定
    size_t memory_pool_size = 1024 * 1024 * 1024;  // 1GB
    bool enable_memory_pool = true;          // メモリプール使用
    
    // 最適化設定
    bool enable_simd = true;                 // SIMD最適化
    bool enable_kernel_fusion = true;       // カーネル融合
    
    // ログ設定
    LogLevel log_level = LogLevel::Info;
    std::string log_file = "";
};

enum class LogLevel {
    Debug, Info, Warning, Error, Critical
};
```

---

## 🧠 コアクラス

### 1. Tensor クラス

**基本テンソル操作**

```cpp
namespace crllwtt::Core {

class Tensor {
public:
    // コンストラクタ
    Tensor();                                    // 空テンソル
    Tensor(const std::vector<int>& shape);       // 形状指定
    Tensor(const std::vector<int>& shape, float value);  // 初期値指定
    Tensor(const Tensor& other);                 // コピーコンストラクタ
    Tensor(Tensor&& other);                      // ムーブコンストラクタ
    
    // 基本プロパティ
    const std::vector<int>& GetShape() const;
    int GetSize() const;                         // 総要素数
    int GetDimension() const;                    // 次元数
    std::string ShapeString() const;             // 形状文字列表現
    
    // データアクセス
    float* GetData();                            // 生データポインタ
    const float* GetData() const;
    float GetValue(const std::vector<int>& indices) const;
    void SetValue(const std::vector<int>& indices, float value);
    
    // 形状操作
    void Resize(const std::vector<int>& new_shape);
    Tensor Reshape(const std::vector<int>& new_shape) const;
    Tensor Transpose(const std::vector<int>& axes = {}) const;
    Tensor Squeeze(int axis = -1) const;         // 次元削除
    Tensor Unsqueeze(int axis) const;            // 次元追加
    
    // 数学演算
    Tensor Add(const Tensor& other) const;       // 要素ごと加算
    Tensor Subtract(const Tensor& other) const;  // 要素ごと減算
    Tensor Multiply(const Tensor& other) const;  // 要素ごと乗算
    Tensor Divide(const Tensor& other) const;    // 要素ごと除算
    Tensor MatMul(const Tensor& other) const;    // 行列乗算
    
    // スカラー演算
    Tensor MultiplyScalar(float scalar) const;
    Tensor AddScalar(float scalar) const;
    
    // 統計関数
    float Sum() const;                           // 全要素の和
    float Mean() const;                          // 平均値
    float Max() const;                           // 最大値
    float Min() const;                           // 最小値
    float Norm() const;                          // ユークリッドノルム
    
    // 初期化
    void Random(float min = 0.0f, float max = 1.0f);  // 乱数初期化
    void Zero();                                 // ゼロ初期化
    void Fill(float value);                      // 定数初期化
    void Normal(float mean = 0.0f, float std = 1.0f); // 正規分布初期化
    
    // ユーティリティ
    bool IsEmpty() const;
    bool IsScalar() const;
    bool IsVector() const;
    bool IsMatrix() const;
    void Print(const std::string& name = "") const;
};

}
```

### 2. TransformerBuilder クラス

**高性能Transformer構築**

```cpp
namespace crllwtt::Core {

class TransformerBuilder {
public:
    TransformerBuilder();
    
    // アーキテクチャ設定
    TransformerBuilder& SetModelDimension(int d_model);
    TransformerBuilder& SetNumHeads(int num_heads);
    TransformerBuilder& SetNumLayers(int num_layers);
    TransformerBuilder& SetMaxSequenceLength(int max_seq_len);
    TransformerBuilder& SetFeedForwardDim(int ff_dim);
    
    // 最適化設定
    TransformerBuilder& EnableSparseAttention(bool enable, float sparsity_ratio = 0.1f);
    TransformerBuilder& EnableTimeAwareness(bool enable, float delay_compensation = 0.0f);
    TransformerBuilder& EnableKernelFusion(bool enable);
    TransformerBuilder& SetMaxInferenceTime(float time_ms);
    
    // 正規化・ドロップアウト
    TransformerBuilder& SetDropoutRate(float rate);
    TransformerBuilder& SetLayerNormEpsilon(float epsilon);
    TransformerBuilder& EnablePreNorm(bool enable);
    
    // 並列処理
    TransformerBuilder& SetNumThreads(int num_threads);
    TransformerBuilder& EnableAsyncExecution(bool enable);
    
    // 高度な設定
    TransformerBuilder& SetActivationFunction(ActivationType activation);
    TransformerBuilder& EnableGradientCheckpointing(bool enable);
    TransformerBuilder& SetPrecisionMode(PrecisionMode mode);
    
    // 構築
    std::unique_ptr<Transformer> Build();
    
private:
    TransformerConfig config_;
};

// 設定構造体
struct TransformerConfig {
    // アーキテクチャ
    int d_model = 512;
    int num_heads = 8;
    int num_layers = 6;
    int max_seq_len = 512;
    int ff_dim = 2048;
    
    // 最適化
    bool enable_sparse_attention = false;
    float sparsity_ratio = 0.1f;
    bool enable_time_awareness = false;
    float delay_compensation = 0.0f;
    bool enable_kernel_fusion = true;
    float max_inference_time_ms = 10.0f;
    
    // 正規化
    float dropout_rate = 0.1f;
    float layer_norm_epsilon = 1e-5f;
    bool use_pre_norm = true;
    
    // 並列処理
    int num_threads = -1;
    bool async_execution = false;
    
    // 高度な設定
    ActivationType activation = ActivationType::GELU;
    bool gradient_checkpointing = false;
    PrecisionMode precision = PrecisionMode::FP32;
};

enum class ActivationType { ReLU, GELU, Swish, Tanh, Sigmoid };
enum class PrecisionMode { FP32, FP16, INT8, MIXED };

}
```

### 3. Transformer クラス

**メインTransformerモデル**

```cpp
namespace crllwtt::Core {

class Transformer {
public:
    explicit Transformer(const TransformerConfig& config);
    virtual ~Transformer();
    
    // 推論
    Tensor Forward(const Tensor& input, 
                  const Tensor* attention_mask = nullptr,
                  const TimeEncodingInfo* time_info = nullptr,
                  int person_id = -1);
    
    // バッチ推論
    std::vector<Tensor> ForwardBatch(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor*>& masks = {},
                                   const std::vector<TimeEncodingInfo*>& time_infos = {},
                                   const std::vector<int>& person_ids = {});
    
    // 不確実性付き推論
    std::pair<Tensor, Tensor> ForwardWithUncertainty(
        const Tensor& input,
        const Tensor* attention_mask = nullptr,
        const TimeEncodingInfo* time_info = nullptr,
        int num_samples = 10);
    
    // マルチステップ予測
    Tensor PredictMultiStep(const Tensor& input, int num_steps);
    
    // ストリーミング推論
    Tensor ForwardStreaming(const Tensor& new_input,
                          bool is_first_chunk = false,
                          bool is_last_chunk = false);
    
    // モデル情報
    const TransformerConfig& GetConfig() const;
    size_t GetParameterCount() const;
    size_t GetMemoryUsage() const;
    
    // 最適化
    void OptimizeForInference(int optimization_level = 1);
    void EnableBenchmarkMode(bool enable);
    
    // 統計
    struct InferenceStats {
        float last_inference_time_ms;
        float average_inference_time_ms;
        size_t total_inferences;
        float cache_hit_ratio;
        size_t peak_memory_usage;
    };
    
    InferenceStats GetInferenceStats() const;
    void ResetStats();
    
    // パラメータアクセス（高度な用途）
    std::vector<Tensor> GetParameters() const;
    void SetParameters(const std::vector<Tensor>& parameters);
    
    // 保存・読み込み
    void SaveModel(const std::string& filepath) const;
    void LoadModel(const std::string& filepath);
    
private:
    class TransformerImpl;
    std::unique_ptr<TransformerImpl> impl_;
};

}
```

### 4. STATransformer クラス

**環境制御専用アーキテクチャ**

```cpp
namespace crllwtt::Core {

class STATransformer {
public:
    explicit STATransformer(const STAConfig& config);
    virtual ~STATransformer();
    
    // 状態予測
    Tensor PredictState(const Tensor& observable_state,
                       const Tensor& control_input,
                       const TimeEncodingInfo* time_info = nullptr,
                       int person_id = -1);
    
    // 不確実性付き予測
    std::pair<Tensor, Tensor> PredictWithUncertainty(
        const Tensor& observable_state,
        const Tensor& control_input,
        const TimeEncodingInfo* time_info = nullptr,
        int person_id = -1,
        int num_samples = 5);
    
    // 最適制御計算
    Tensor ComputeOptimalControl(const Tensor& observable_state,
                               const Tensor& current_control,
                               const Tensor& target_state,
                               float control_constraint = 1.0f);
    
    // 感度解析
    Tensor ComputeSensitivity(const Tensor& observable_state,
                            const Tensor& control_input,
                            int output_dim = -1);
    
    // リアルタイム学習
    void UpdateModel(const Tensor& observable_state,
                    const Tensor& control_input,
                    const Tensor& actual_state,
                    float learning_rate = -1.0f);
    
    // 個人適応
    void AdaptToUser(int person_id, 
                    const std::vector<Tensor>& user_data,
                    const std::vector<Tensor>& user_responses);
    
    // 制御統計
    struct ControlStats {
        float control_effectiveness;    // 制御効果 (0-1)
        float prediction_accuracy;      // 予測精度
        float adaptation_speed;         // 適応速度
        int num_updates;               // 更新回数
        float average_response_time_ms; // 平均応答時間
    };
    
    ControlStats GetControlStats() const;
    
    // 設定
    const STAConfig& GetConfig() const;
    void SetTargetResponseTime(float time_ms);
    void EnablePersonalAdaptation(bool enable);
    
private:
    class STATransformerImpl;
    std::unique_ptr<STATransformerImpl> impl_;
};

// STA設定
struct STAConfig {
    // 入出力次元
    int observable_state_dim = 8;      // 観測可能状態次元
    int controllable_input_dim = 4;    // 制御可能入力次元  
    int predicted_state_dim = 4;       // 予測状態次元
    
    // モデルアーキテクチャ
    int d_model = 256;
    int num_heads = 8;
    int num_layers = 4;
    int max_seq_len = 128;
    
    // 学習設定
    float learning_rate = 0.001f;
    float control_gain = 0.1f;
    bool enable_uncertainty = true;
    int ensemble_size = 3;
    
    // 個人化
    bool enable_personal_adaptation = true;
    int personal_embed_dim = 32;
    
    // 制約
    float max_response_time_ms = 1.0f;
    float control_smoothness = 0.1f;
    
    // 最適化
    bool enable_sparse_attention = true;
    float sparsity_ratio = 0.2f;
    bool enable_kernel_fusion = true;
};

}
```

### 5. AdaptiveGradient クラス

**リアルタイム勾配計算**

```cpp
namespace crllwtt::Core {

class AdaptiveGradient {
public:
    explicit AdaptiveGradient(const AdaptiveGradientConfig& config = {});
    
    // 勾配計算
    GradientResult ComputeGradient(
        std::function<Tensor(const std::vector<Tensor>&)> loss_function,
        const std::vector<Tensor>& parameters,
        float max_time_ms = -1.0f);
    
    // 偏微分計算
    GradientResult ComputePartialDerivative(
        std::function<Tensor(const Tensor&)> function,
        const Tensor& input,
        int output_idx = 0,
        float max_time_ms = -1.0f);
    
    // 感度計算（STA用）
    GradientResult ComputeSensitivity(
        std::function<Tensor(const Tensor&, const Tensor&)> state_predictor,
        const Tensor& observable_state,
        const Tensor& control_input,
        float max_time_ms = -1.0f);
    
    // キャッシュ管理
    void CacheGradient(size_t input_hash, const Tensor& gradient, 
                      const GradientResult& computation_info);
    std::unique_ptr<GradientResult> GetCachedGradient(size_t input_hash) const;
    void ClearCache();
    
    // 精度適応
    float AdaptPrecision(float available_time_ms) const;
    
    // 統計
    struct GradientStats {
        size_t total_computations;
        size_t cache_hits;
        size_t cache_misses;
        float average_computation_time_ms;
        float cache_hit_ratio;
        size_t numerical_fallbacks;
    };
    
    GradientStats GetStats() const;
    void ResetStats();
    
    // バッファマネージャー設定
    void SetBufferManager(std::shared_ptr<Utils::PreallocatedBuffers> buffers);
    
private:
    AdaptiveGradientConfig config_;
    // ... 内部実装
};

// 勾配計算結果
struct GradientResult {
    Tensor gradient;
    float computation_time_ms;
    float precision_achieved;
    bool used_cache;
    bool used_numerical;
    
    GradientResult();
};

// 勾配計算設定
struct AdaptiveGradientConfig {
    float max_computation_time_ms = 1.0f;
    float precision_threshold = 1e-6f;
    int max_cache_size = 1000;
    bool enable_numerical_fallback = true;
    bool enable_gradient_clipping = true;
    float gradient_clip_value = 1.0f;
    bool enable_adaptive_precision = true;
};

}
```

---

## ⚡ 最適化モジュール

### 1. SparseAttention クラス

**効率的スパース注意機構**

```cpp
namespace crllwtt::Core {

class SparseAttention {
public:
    explicit SparseAttention(const SparseAttentionConfig& config = {});
    
    // スパース注意計算
    Tensor Forward(const Tensor& query,
                  const Tensor& key,
                  const Tensor& value,
                  const Tensor* mask = nullptr);
    
    // スパースマスク生成
    Tensor CreateSparseMask(int seq_len, SparsePatternType pattern_type = SparsePatternType::Adaptive);
    
    // 設定
    void SetSparsityRatio(float ratio);
    float GetSparsityRatio() const;
    void SetAdaptiveSparsity(bool enable);
    const SparseAttentionConfig& GetConfig() const;
    
private:
    SparseAttentionConfig config_;
    bool adaptive_sparsity_ = true;
    
    // 内部メソッド
    Tensor ComputeAttentionScores(const Tensor& query, const Tensor& key);
    Tensor ApplySparsityMask(const Tensor& attention_scores, const Tensor& mask);
    Tensor ComputeAdaptiveMask(const Tensor& attention_scores);
    Tensor CreateLocalWindowMask(int seq_len);
    Tensor CreateBlockSparseMask(int seq_len);
};

// スパース注意設定
struct SparseAttentionConfig {
    float sparsity_ratio = 0.1f;        // スパース率
    int block_size = 64;                 // ブロックサイズ
    bool use_random_sparsity = false;    // ランダムスパース性
    bool use_local_attention = true;     // ローカル注意
    int local_window_size = 128;         // ローカルウィンドウサイズ
};

// スパースパターンタイプ
enum class SparsePatternType {
    Random,         // ランダムスパース
    Structured,     // 構造化スパース
    LocalWindow,    // ローカルウィンドウ
    BlockSparse,    // ブロックスパース
    Adaptive        // 適応的スパース
};

// ユーティリティ関数
namespace SparseAttentionUtils {
    float ComputeSparsity(const Tensor& attention_weights);
    Tensor CreateStridedMask(int seq_len, int stride);
    Tensor OptimizeSparsityPattern(const Tensor& query, const Tensor& key, float target_sparsity);
}

}
```

### 2. KernelFusion クラス

**演算子融合最適化**

```cpp
namespace crllwtt::Optimization {

class KernelFusion {
public:
    explicit KernelFusion(const KernelFusionConfig& config = {});
    
    // 融合演算
    Tensor FusedLinearReLU(const Tensor& input, 
                          const Tensor& weight, 
                          const Tensor& bias);
    
    Tensor FusedAttentionSoftmax(const Tensor& query,
                                const Tensor& key,
                                const Tensor& value,
                                const Tensor* mask = nullptr,
                                float scale = 1.0f);
    
    Tensor FusedLayerNormReLU(const Tensor& input,
                             const Tensor& gamma,
                             const Tensor& beta,
                             float epsilon = 1e-5f);
    
    Tensor FusedGELUDropout(const Tensor& input, 
                           float dropout_rate = 0.1f,
                           bool training = true);
    
    // 計算グラフ最適化
    void OptimizeComputationGraph(const std::vector<Operation>& operations);
    
    // 統計
    struct OptimizationStats {
        int total_operations = 0;
        int fused_operations = 0;
        int single_operations = 0;
        float fusion_ratio = 0.0f;
        float estimated_speedup = 1.0f;
    };
    
    OptimizationStats GetOptimizationStats() const;
    const std::vector<FusedOperation>& GetFusedOperations() const;
    
    // 静的ユーティリティ
    static Tensor MatMulBiasActivation(const Tensor& input,
                                      const Tensor& weight,
                                      const Tensor& bias,
                                      ActivationType activation = ActivationType::None);
    
    Tensor FusedTransformerBlock(const Tensor& input,
                                const TransformerBlockParams& params);
    
private:
    KernelFusionConfig config_;
    std::vector<FusedOperation> fused_operations_;
};

// カーネル融合設定
struct KernelFusionConfig {
    bool enable_linear_relu_fusion = true;
    bool enable_attention_fusion = true;
    bool enable_layernorm_fusion = true;
    bool enable_gelu_dropout_fusion = true;
    bool enable_transformer_block_fusion = true;
    float fusion_threshold = 0.1f;
};

// 演算タイプ
enum class OperationType {
    Linear, ReLU, GELU, Tanh, Sigmoid, LayerNorm, Attention, Dropout, Softmax
};

enum class FusedOperationType {
    LinearReLU, LayerNormReLU, GELUDropout, AttentionSoftmax, TransformerBlock, Single
};

// 高レベル融合パターン
namespace FusionPatterns {
    Tensor FusedMultiHeadAttention(const Tensor& input, /* ... */);
    Tensor FusedFeedForward(const Tensor& input, /* ... */);
    Tensor FusedResidualLayerNorm(const Tensor& input, /* ... */);
}

}
```

---

## 🛠️ ユーティリティ

### 1. SIMDUtils クラス

**SIMD最適化ユーティリティ**

```cpp
namespace crllwtt::Utils {

class SIMDUtils {
public:
    // SIMD情報
    static const int kSIMDWidth;
    static std::string GetSIMDInfo();
    static bool IsSIMDSupported();
    static int GetOptimalAlignment();
    
    // ベクトル演算
    static void VectorAdd(const float* a, const float* b, float* result, int size);
    static void VectorMul(const float* a, const float* b, float* result, int size);
    static float DotProduct(const float* a, const float* b, int size);
    static void VectorScale(const float* input, float scale, float* output, int size);
    
    // 行列演算
    static void MatrixMultiply(const float* a, const float* b, float* c,
                              int m, int n, int k,
                              bool transpose_a = false, bool transpose_b = false);
    
    // 活性化関数
    static void ApplyActivation(const float* input, float* output, int size, ActivationType activation);
    static void SoftMax(const float* input, float* output, int size);
    
private:
    SIMDUtils() = delete; // 静的クラス
};

// SIMD対応アロケーター
class SIMDAllocator {
public:
    static float* AllocateAligned(size_t size);
    static void FreeAligned(float* ptr);
    static bool IsAligned(const void* ptr);
};

}
```

### 2. PreallocatedBuffers クラス

**メモリ効率管理**

```cpp
namespace crllwtt::Utils {

class PreallocatedBuffers {
public:
    explicit PreallocatedBuffers(const BufferPoolConfig& config = {});
    ~PreallocatedBuffers();
    
    // テンソル管理
    Tensor GetWorkTensor(const std::vector<int>& shape);
    void ReturnWorkTensor(Tensor&& tensor);
    Tensor GetAttentionBuffer(int seq_len, int num_heads);
    
    // 勾配管理
    void StoreGradient(const Tensor& gradient, const std::string& param_name);
    Tensor GetCachedGradient(const std::string& param_name, size_t offset = 0) const;
    
    // モデル固有事前割り当て
    void PreallocateForModel(int max_seq_len, int d_model, int num_heads, int num_layers);
    
    // 統計
    struct MemoryStats {
        size_t total_allocated_mb;
        size_t tensor_pool_usage_mb;
        size_t gradient_buffer_usage_mb;
        size_t attention_buffer_usage_mb;
        double memory_efficiency;
    };
    
    MemoryStats GetMemoryStats() const;
    void Reset();
    void SetRealTimeMode(bool enable);
    
private:
    BufferPoolConfig config_;
    // ... 内部実装
};

// バッファプール設定
struct BufferPoolConfig {
    size_t max_tensor_size = 1024 * 1024;  // 1MB
    size_t tensor_pool_size = 16;
    size_t gradient_buffer_size = 512;
    size_t attention_buffer_size = 256;
    bool enable_simd_alignment = true;
    size_t alignment = 32;
};

// RAIIヘルパー
class ScopedTensor {
public:
    ScopedTensor(Tensor&& tensor, PreallocatedBuffers* buffers);
    ~ScopedTensor();
    
    Tensor& operator*();
    const Tensor& operator*() const;
    Tensor* operator->();
    const Tensor* operator->() const;
    
private:
    Tensor tensor_;
    PreallocatedBuffers* buffers_;
};

}
```

### 3. TimeEncoding ユーティリティ

**時間認識エンコーディング**

```cpp
namespace crllwtt::Core {

// 時間情報構造体
struct TimeEncodingInfo {
    std::vector<float> timestamps;      // タイムスタンプ
    float personal_delay = 0.0f;        // 個人遅延
    float temporal_scale = 1.0f;        // 時間スケール
    bool enable_multiscale = true;      // マルチスケール有効
    
    TimeEncodingInfo() = default;
    TimeEncodingInfo(const std::vector<float>& ts, float delay = 0.0f);
};

// 時間エンコーディングユーティリティ
namespace TimeEncodingUtils {
    
    // 時間情報作成
    TimeEncodingInfo CreateTimeInfo(const std::vector<float>& timestamps, 
                                   float personal_delay = 0.0f);
    
    // 時間エンコーディング計算
    Tensor ComputeTimeEncoding(const TimeEncodingInfo& time_info, 
                              int seq_len, int d_model);
    
    // マルチスケール時間特徴
    Tensor ComputeMultiscaleTimeFeatures(const std::vector<float>& timestamps,
                                        const std::vector<float>& scales);
    
    // 個人遅延補償
    std::vector<float> ApplyDelayCompensation(const std::vector<float>& timestamps,
                                            float personal_delay);
    
    // 時間パターン分析
    struct TemporalPattern {
        float dominant_frequency;
        float periodicity_strength;
        std::vector<float> harmonic_components;
    };
    
    TemporalPattern AnalyzeTemporalPattern(const std::vector<float>& timestamps);
    
    // 適応的時間エンコーディング
    Tensor AdaptiveTimeEncoding(const TimeEncodingInfo& time_info,
                               const TemporalPattern& pattern,
                               int d_model);
}

}
```

---

## ⚙️ 設定と初期化

### グローバル設定

```cpp
namespace crllwtt {

// パフォーマンス設定
struct PerformanceConfig {
    // 並列処理
    int num_threads = -1;               // -1で自動検出
    bool enable_openmp = true;
    bool enable_async_execution = false;
    
    // SIMD設定
    bool enable_simd = true;
    bool force_simd_alignment = true;
    
    // メモリ設定
    size_t memory_pool_size = 1GB;
    bool enable_memory_prefetch = true;
    bool enable_numa_optimization = false;
    
    // キャッシュ設定
    size_t attention_cache_size = 100MB;
    size_t gradient_cache_size = 50MB;
    bool enable_computation_cache = true;
};

// プロファイリング設定
struct ProfilingConfig {
    bool enable_timing = false;
    bool enable_memory_tracking = false;
    bool enable_cache_statistics = false;
    std::string profile_output_file = "";
    int profile_sampling_rate = 1000;  // Hz
};

// デバッグ設定
struct DebugConfig {
    LogLevel log_level = LogLevel::Info;
    bool enable_assertions = true;
    bool enable_nan_checking = false;
    bool enable_bounds_checking = false;
    std::string debug_output_dir = "./debug";
};

// 完全な初期化
bool Initialize(const PerformanceConfig& perf_config = {},
               const ProfilingConfig& prof_config = {},
               const DebugConfig& debug_config = {});

}
```

### 環境変数

```bash
# パフォーマンス関連
export LWTT_NUM_THREADS=8
export LWTT_ENABLE_SIMD=1
export LWTT_MEMORY_POOL_SIZE=2048M

# 最適化関連
export LWTT_ENABLE_SPARSE_ATTENTION=1
export LWTT_SPARSITY_RATIO=0.1
export LWTT_ENABLE_KERNEL_FUSION=1

# デバッグ関連
export LWTT_LOG_LEVEL=INFO
export LWTT_ENABLE_PROFILING=0
export LWTT_DEBUG_OUTPUT_DIR=./logs
```

---

## 🚨 エラーハンドリング

### 例外クラス

```cpp
namespace crllwtt {

// 基底例外クラス
class LwTTException : public std::exception {
public:
    explicit LwTTException(const std::string& message);
    const char* what() const noexcept override;
    
protected:
    std::string message_;
};

// 具体的な例外クラス
class InvalidShapeException : public LwTTException {
public:
    InvalidShapeException(const std::vector<int>& expected, 
                         const std::vector<int>& actual);
};

class ComputationTimeoutException : public LwTTException {
public:
    ComputationTimeoutException(float timeout_ms, float actual_ms);
};

class MemoryAllocationException : public LwTTException {
public:
    MemoryAllocationException(size_t requested_bytes);
};

class ModelLoadException : public LwTTException {
public:
    ModelLoadException(const std::string& filepath, const std::string& reason);
};

class ConfigurationException : public LwTTException {
public:
    ConfigurationException(const std::string& parameter, const std::string& issue);
};

}
```

### エラーハンドリングのベストプラクティス

```cpp
#include <LwTT/LwTT.hpp>

int main() {
    try {
        // ライブラリ初期化
        if (!LwTT::Initialize()) {
            std::cerr << "LwTT初期化失敗" << std::endl;
            return -1;
        }
        
        // モデル構築
        auto transformer = LwTT::Core::TransformerBuilder()
            .SetModelDimension(256)
            .SetNumHeads(8)
            .SetMaxInferenceTime(1.0f)  // 1ms制約
            .Build();
        
        // 推論実行
        LwTT::Core::Tensor input({1, 100, 256});
        input.Random();
        
        auto output = transformer->Forward(input);
        
        std::cout << "推論成功" << std::endl;
        
    } catch (const LwTT::ComputationTimeoutException& e) {
        std::cerr << "計算時間超過: " << e.what() << std::endl;
    } catch (const LwTT::InvalidShapeException& e) {
        std::cerr << "形状エラー: " << e.what() << std::endl;
    } catch (const LwTT::LwTTException& e) {
        std::cerr << "LwTTエラー: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "一般エラー: " << e.what() << std::endl;
    }
    
    LwTT::Cleanup();
    return 0;
}
```

---

## 📊 パフォーマンス監視

### ベンチマーク用API

```cpp
namespace crllwtt::Benchmark {

class PerformanceMonitor {
public:
    // タイマー
    void StartTimer(const std::string& name);
    void StopTimer(const std::string& name);
    float GetElapsedTime(const std::string& name) const;
    
    // メモリ監視
    size_t GetCurrentMemoryUsage() const;
    size_t GetPeakMemoryUsage() const;
    void ResetMemoryStats();
    
    // 統計取得
    struct BenchmarkStats {
        std::map<std::string, float> timing_stats;
        size_t current_memory_mb;
        size_t peak_memory_mb;
        float cpu_utilization;
        float cache_hit_ratio;
    };
    
    BenchmarkStats GetStats() const;
    void ExportStats(const std::string& filename) const;
    
    // プロファイリング
    void EnableDetailedProfiling(bool enable);
    void SetSamplingRate(int hz);
    
private:
    // 内部実装
};

// グローバルモニター
extern PerformanceMonitor g_performance_monitor;

// RAIIタイマー
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();
    
private:
    std::string name_;
};

// マクロ
#define LWTT_PROFILE_SCOPE(name) ScopedTimer _timer(name)
#define LWTT_PROFILE_FUNCTION() LWTT_PROFILE_SCOPE(__FUNCTION__)

}
```

---

このAPIリファレンスにより、crlLwTTライブラリの全機能を効率的に活用できます。各クラスとメソッドの詳細な使用方法については、対応するヘッダーファイルのドキュメンテーションコメントも参照してください。