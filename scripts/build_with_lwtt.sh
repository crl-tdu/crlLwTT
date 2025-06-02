#!/bin/bash

# ==========================================
# crlLwTT 効率的ビルドスクリプト
# ==========================================

set -e  # エラー時に終了

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# デフォルト設定
BUILD_TYPE="Release"
BUILD_DIR="build"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CLEAN=false
VERBOSE=false
RUN_TESTS=false
SUBMODULE_MODE=false
INSTALL=false

# 使用方法を表示
show_usage() {
    echo "crlLwTT 効率的ビルドスクリプト"
    echo ""
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -t, --type TYPE        ビルドタイプ (Debug/Release/RelWithDebInfo) [デフォルト: Release]"
    echo "  -d, --dir DIR          ビルドディレクトリ [デフォルト: build]"
    echo "  -j, --jobs N           並列ジョブ数 [デフォルト: $JOBS]"
    echo "  -c, --clean            クリーンビルド"
    echo "  -v, --verbose          詳細出力"
    echo "  -s, --submodule        submoduleモードでビルド"
    echo "  -i, --install          インストール実行"
    echo "  --test                 テスト実行"
    echo "  --minimal              最小構成ビルド"
    echo "  --full                 フル機能ビルド"
    echo "  -h, --help             このヘルプを表示"
}

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -d|--dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--submodule)
            SUBMODULE_MODE=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --minimal)
            MINIMAL_BUILD=true
            shift
            ;;
        --full)
            FULL_BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "未知のオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

echo "=========================================="
echo "crlLwTT ビルドスクリプト"
echo "=========================================="
echo "プロジェクトルート: $PROJECT_ROOT"
echo "ビルドタイプ: $BUILD_TYPE"
echo "ビルドディレクトリ: $BUILD_DIR"
echo "並列ジョブ数: $JOBS"
echo "Submoduleモード: $SUBMODULE_MODE"
echo "=========================================="

# ビルドディレクトリの処理
if [[ "$CLEAN" == true ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "🧹 クリーンビルド: $BUILD_DIR を削除中..."
    rm -rf "$BUILD_DIR"
fi

# ビルドディレクトリ作成
mkdir -p "$BUILD_DIR"

# CMakeオプションの設定
CMAKE_OPTS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

# Verboseモード
if [[ "$VERBOSE" == true ]]; then
    CMAKE_OPTS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi

# ビルド構成の設定
if [[ "$MINIMAL_BUILD" == true ]]; then
    echo "📦 最小構成ビルド"
    CMAKE_OPTS+=(
        "-DLWTT_BUILD_TESTS=OFF"
        "-DLWTT_BUILD_DOCS=OFF"
        "-DLWTT_BUILD_PYTHON_BINDINGS=OFF"
        "-DLWTT_BUILD_SHARED=OFF"
        "-DLWTT_ENABLE_QUANTIZATION=OFF"
    )
elif [[ "$FULL_BUILD" == true ]]; then
    echo "🚀 フル機能ビルド"
    CMAKE_OPTS+=(
        "-DLWTT_BUILD_TESTS=ON"
        "-DLWTT_BUILD_DOCS=ON"
        "-DLWTT_BUILD_PYTHON_BINDINGS=ON"
        "-DLWTT_BUILD_SHARED=ON"
        "-DLWTT_ENABLE_QUANTIZATION=ON"
        "-DLWTT_ENABLE_PROFILING=ON"
    )
elif [[ "$SUBMODULE_MODE" == true ]]; then
    echo "📦 Submodule最適化ビルド"
    CMAKE_OPTS+=(
        "-DLWTT_BUILD_TESTS=OFF"
        "-DLWTT_BUILD_DOCS=OFF"
        "-DLWTT_BUILD_PYTHON_BINDINGS=OFF"
        "-DLWTT_BUILD_SHARED=OFF"
        "-DLWTT_INSTALL=OFF"
    )
else
    echo "⚡ 標準ビルド"
fi

# キャッシュ最適化
if command -v ccache >/dev/null 2>&1; then
    echo "🚀 ccache 検出: ビルドを高速化"
    CMAKE_OPTS+=("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
fi

# CMake設定実行
echo ""
echo "🔧 CMake設定実行中..."
echo "cmake -B $BUILD_DIR ${CMAKE_OPTS[@]}"

if cmake -B "$BUILD_DIR" "${CMAKE_OPTS[@]}"; then
    echo "✅ CMake設定完了"
else
    echo "❌ CMake設定エラー"
    exit 1
fi

# ビルド実行
echo ""
echo "🔨 ビルド実行中... (並列度: $JOBS)"

BUILD_OPTS=("--parallel" "$JOBS")
if [[ "$VERBOSE" == true ]]; then
    BUILD_OPTS+=("--verbose")
fi

if cmake --build "$BUILD_DIR" "${BUILD_OPTS[@]}"; then
    echo "✅ ビルド完了"
else
    echo "❌ ビルドエラー"
    exit 1
fi

# テスト実行
if [[ "$RUN_TESTS" == true ]]; then
    echo ""
    echo "🧪 テスト実行中..."
    if ctest --test-dir "$BUILD_DIR" --output-on-failure; then
        echo "✅ テスト完了"
    else
        echo "❌ テストエラー"
        exit 1
    fi
fi

# インストール実行
if [[ "$INSTALL" == true ]]; then
    echo ""
    echo "📦 インストール実行中..."
    if cmake --install "$BUILD_DIR"; then
        echo "✅ インストール完了"
    else
        echo "❌ インストールエラー"
        exit 1
    fi
fi

# 完了メッセージ
echo ""
echo "🎉 ビルドスクリプト完了！"
echo ""
echo "ビルド結果:"
echo "  ビルドディレクトリ: $BUILD_DIR"
echo "  ビルドタイプ: $BUILD_TYPE"

# ビルド成果物の情報
if [[ -f "$BUILD_DIR/libLwTT.a" ]]; then
    LIB_SIZE=$(du -h "$BUILD_DIR/libLwTT.a" | cut -f1)
    echo "  静的ライブラリ: $BUILD_DIR/libLwTT.a ($LIB_SIZE)"
fi

if [[ -f "$BUILD_DIR/libLwTT.so" ]] || [[ -f "$BUILD_DIR/libLwTT.dylib" ]]; then
    SHARED_LIB=$(find "$BUILD_DIR" -name "libLwTT.*" -type f | grep -E '\.(so|dylib)$' | head -1)
    if [[ -n "$SHARED_LIB" ]]; then
        LIB_SIZE=$(du -h "$SHARED_LIB" | cut -f1)
        echo "  共有ライブラリ: $SHARED_LIB ($LIB_SIZE)"
    fi
fi

echo ""
echo "使用方法:"
echo "  ライブラリ: target_link_libraries(your_target PRIVATE LwTT)"
echo "  ヘッダー: #include <LwTT/core/Tensor.hpp>"
