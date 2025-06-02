# crlLwTT Submodule使用ガイド

## 概要

crlLwTTは軽量なTransformerライブラリとして設計されており、他のプロジェクトのsubmoduleとして効率的に利用できます。このドキュメントでは、submoduleとしての導入と利用方法について詳しく説明します。

## 特徴

### 🚀 高速化されたSubmoduleビルド

- **選択的コンポーネント**: 必要な機能のみビルド
- **簡略化された依存関係**: 軽量な構成で高速ビルド
- **自動検知システム**: メインプロジェクトとsubmoduleを自動判別

### 📦 最適化されたビルドオプション

| オプション | メインプロジェクト | Submodule | 説明 |
|---|---|---|---|
| `LWTT_BUILD_TESTS` | ON | OFF | テスト・ベンチマーク |
| `LWTT_BUILD_DOCS` | ON | OFF | ドキュメント生成 |
| `LWTT_BUILD_PYTHON_BINDINGS` | ON | OFF | Python バインディング |
| `LWTT_BUILD_SHARED` | ON | OFF | 共有ライブラリ |
| `LWTT_BUILD_STATIC` | ON | ON | 静的ライブラリ |
| `LWTT_INSTALL` | ON | OFF | インストール設定 |

## セットアップ

### 1. Submoduleとして追加

```bash
# プロジェクトルートで実行
git submodule add https://github.com/your-org/crlLwTT.git external/crlLwTT
git submodule update --init --recursive
```

### 2. CMakeLists.txtに追加

```cmake
cmake_minimum_required(VERSION 3.16)
project(YourProject VERSION 1.0.0 LANGUAGES CXX)

# C++23標準を設定
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# crlLwTT submoduleを追加
add_subdirectory(external/crlLwTT)

# あなたのプロジェクトのターゲット
add_executable(your_app src/main.cpp)

# LwTTとリンク
target_link_libraries(your_app PRIVATE LwTT)
```

### 3. 基本的なC++使用例

```cpp
#include <LwTT/core/Tensor.hpp>
#include <LwTT/layers/Attention.hpp>
#include <LwTT/models/Transformer.hpp>

int main() {
    // Transformerモデル設定
    LwTT::ModelConfig config;
    config.sequence_length = 512;
    config.hidden_size = 768;
    config.num_heads = 12;
    config.num_layers = 6;
    
    // モデル作成
    LwTT::Transformer model(config);
    
    // 入力テンソル作成
    auto input = LwTT::Tensor::randn({1, 512, 768});
    
    // 推論実行
    auto output = model.forward(input);
    
    std::cout << "Output shape: " << output.shape() << std::endl;
    return 0;
}
```

## 依存関係の管理

### 最小構成（推奨）

```cmake
# 必要最小限の設定
option(LWTT_USE_EIGEN "Use Eigen for linear algebra" ON)
option(LWTT_ENABLE_OPENMP "Enable OpenMP" ON)
option(LWTT_ENABLE_SIMD "Enable SIMD optimizations" ON)

add_subdirectory(external/crlLwTT)
```

### 機能拡張構成

```cmake
# 高機能設定
option(LWTT_ENABLE_QUANTIZATION "Enable quantization support" ON)
option(LWTT_ENABLE_PROFILING "Enable profiling" ON)

# PyTorchとの連携
find_package(Torch QUIET)
if(Torch_FOUND)
    set(LWTT_HAS_TORCH TRUE)
endif()

add_subdirectory(external/crlLwTT)
```

## 高度な使用例

### カスタムAttentionレイヤー

```cpp
#include <LwTT/layers/MultiHeadAttention.hpp>

class CustomAttention : public LwTT::MultiHeadAttention {
public:
    CustomAttention(int hidden_size, int num_heads) 
        : MultiHeadAttention(hidden_size, num_heads) {}
    
    LwTT::Tensor forward(const LwTT::Tensor& input) override {
        // カスタム実装
        auto qkv = compute_qkv(input);
        auto attention_weights = scaled_dot_product_attention(qkv);
        return apply_output_projection(attention_weights);
    }
};
```

### 時系列データ処理

```cpp
#include <LwTT/core/TimeAwareProcessor.hpp>

int main() {
    // 時系列認識設定
    LwTT::TimeConfig time_config;
    time_config.enable_temporal_encoding = true;
    time_config.max_sequence_length = 1024;
    
    LwTT::TimeAwareProcessor processor(time_config);
    
    // 時系列データの処理
    auto time_series = LwTT::Tensor::from_vector(your_data);
    auto processed = processor.encode_temporal_features(time_series);
    
    return 0;
}
```

## ビルドパフォーマンス最適化

### 並列ビルド

```bash
# 並列数を指定してビルド
cmake --build build --parallel 8
```

### キャッシュ利用

```bash
# ccacheを使用して高速化
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake -B build
cmake --build build
```

### 段階的ビルド

```cmake
# 段階的にコンポーネントを有効化
if(ENABLE_ADVANCED_FEATURES)
    set(LWTT_ENABLE_QUANTIZATION ON)
    set(LWTT_ENABLE_PROFILING ON)
endif()
```

## トラブルシューティング

### よくある問題

#### 1. 依存関係エラー

```bash
# Eigenが見つからない場合
sudo apt-get install libeigen3-dev  # Ubuntu/Debian
brew install eigen                   # macOS
```

#### 2. コンパイルエラー

```bash
# C++23対応コンパイラが必要
# GCC 11+, Clang 14+, MSVC 19.29+
```

#### 3. LibTorchとの競合

```cmake
# LibTorchを優先的に検索
find_package(Torch REQUIRED)
set(LWTT_HAS_TORCH TRUE)
```

### デバッグビルド

```cmake
# デバッグ情報を有効化
set(CMAKE_BUILD_TYPE Debug)
set(LWTT_ENABLE_PROFILING ON)
```

## パフォーマンス比較

| 構成 | 初回ビルド時間 | 増分ビルド時間 | バイナリサイズ |
|---|---|---|---|
| フル機能 | ~5分 | ~30秒 | ~50MB |
| Submodule最適化 | ~2分 | ~10秒 | ~15MB |
| 最小構成 | ~1分 | ~5秒 | ~8MB |

## 更新とメンテナンス

### Submoduleの更新

```bash
# 最新版に更新
git submodule update --remote external/crlLwTT

# 特定のバージョンに固定
cd external/crlLwTT
git checkout v1.2.0
cd ../..
git add external/crlLwTT
git commit -m "Update crlLwTT to v1.2.0"
```

### 設定の確認

```bash
# CMake設定の表示
cmake -B build -DCMAKE_BUILD_TYPE=Release
# ビルド後、設定サマリーが表示されます
```

## サポートとリソース

- **プロジェクトページ**: [crlLwTT Repository]
- **API ドキュメント**: `docs/api/`
- **サンプルコード**: `examples/`
- **課題報告**: GitHub Issues

---

このガイドでcrlLwTTを効率的にsubmoduleとして活用し、高性能なTransformerベースのアプリケーションを構築してください！
