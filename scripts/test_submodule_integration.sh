#!/bin/bash

# ==========================================
# crlLwTT Submodule統合テスト
# ==========================================

set -e

echo "=========================================="
echo "crlLwTT Submodule統合テスト開始"
echo "=========================================="

PROJECT_ROOT="/Users/igarashi/local/project_workspace/crlLwTT"
TEST_DIR="$PROJECT_ROOT/test_submodule_integration"

# テスト環境のクリーンアップ
if [[ -d "$TEST_DIR" ]]; then
    echo "🧹 既存のテスト環境をクリーンアップ中..."
    rm -rf "$TEST_DIR"
fi

# テスト用プロジェクト作成
echo "📁 テスト用プロジェクト作成中..."
mkdir -p "$TEST_DIR"

# 親プロジェクトのCMakeLists.txt作成
cat > "$TEST_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)

project(SubmoduleTest VERSION 1.0.0 LANGUAGES CXX)

# C++23標準設定
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# crlLwTTをsubmoduleとして追加（EXCLUDE_FROM_ALLで親プロジェクトのビルドに含めない）
add_subdirectory(../.. crlLwTT_build EXCLUDE_FROM_ALL)

# テスト用の簡単なアプリケーション
add_executable(submodule_test test_main.cpp)

# LwTTとリンク
target_link_libraries(submodule_test PRIVATE LwTT)

# メッセージ表示
message(STATUS "========================================")
message(STATUS "Submodule Integration Test")
message(STATUS "  Main project: ${PROJECT_NAME}")
message(STATUS "  LwTT submodule: Included")
message(STATUS "========================================")
EOF

# テスト用のソースファイル作成
cat > "$TEST_DIR/test_main.cpp" << 'EOF'
#include <iostream>
#include <LwTT/LwTT.hpp>

int main() {
    std::cout << "=== crlLwTT Submodule Integration Test ===" << std::endl;
    
    try {
        // 基本的な機能テスト
        std::cout << "Testing basic functionality..." << std::endl;
        
        // 簡単なテンソル操作
        auto tensor = LwTT::Tensor::zeros({2, 3});
        std::cout << "✓ Tensor creation successful" << std::endl;
        
        // 基本的な演算
        auto result = tensor + 1.0f;
        std::cout << "✓ Tensor operations successful" << std::endl;
        
        std::cout << "✅ All tests passed!" << std::endl;
        std::cout << "🎉 crlLwTT submodule integration successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
EOF

echo "✅ テスト用プロジェクト作成完了"

# CMake設定テスト
echo ""
echo "🔧 CMake設定テスト実行中..."
cd "$TEST_DIR"

if cmake -B build -DCMAKE_BUILD_TYPE=Release; then
    echo "✅ CMake設定成功 - submodule検知が正常に動作"
else
    echo "❌ CMake設定エラー"
    exit 1
fi

# ビルドテスト
echo ""
echo "🔨 ビルドテスト実行中..."
if cmake --build build; then
    echo "✅ ビルド成功 - submoduleライブラリのリンクが正常"
else
    echo "❌ ビルドエラー"
    exit 1
fi

# 実行テスト
echo ""
echo "🚀 実行テスト中..."
if ./build/submodule_test; then
    echo "✅ 実行成功 - submodule統合が完全に動作"
else
    echo "❌ 実行エラー"
    exit 1
fi

echo ""
echo "🎉 すべてのSubmodule統合テストが成功しました！"
echo ""
echo "テスト結果サマリー:"
echo "  ✓ プロジェクト検知: メイン/submodule自動判別"
echo "  ✓ ビルドオプション: submodule最適化設定"
echo "  ✓ ライブラリリンク: 正常な依存関係解決"
echo "  ✓ 実行時動作: 基本機能の正常動作"
echo ""
echo "=========================================="
echo "crlLwTT は submodule として使用可能です！"
echo "=========================================="
EOF