# LibTorch インストールガイド

## 概要

crlGRUライブラリは内部でLibTorch（PyTorchのC++版）を使用しています。このガイドでは、macOSとLinuxでLibTorchをインストールする方法を説明します。

## 動作環境

- **macOS**: 10.14以降（ARM64/Intel）
- **Linux**: Ubuntu 18.04以降、CentOS 7以降
- **C++コンパイラ**: C++17対応（GCC 7以降、Clang 5以降）
- **CMake**: 3.18以降

## インストール方法

### 方法1: 推奨インストール（事前ビルド版）

#### macOS (Apple Silicon)

```bash
# LibTorchダウンロード（CPU版）
cd ~/local
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.2.zip
unzip libtorch-macos-arm64-2.1.2.zip
rm libtorch-macos-arm64-2.1.2.zip

# 環境変数設定（~/.zshrcに追加）
echo 'export LIBTORCH_HOME=$HOME/local/libtorch' >> ~/.zshrc
echo 'export CMAKE_PREFIX_PATH=$LIBTORCH_HOME:$CMAKE_PREFIX_PATH' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH=$LIBTORCH_HOME/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

#### macOS (Intel)

```bash
# LibTorchダウンロード（CPU版）
cd ~/local
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.2.zip
unzip libtorch-macos-x86_64-2.1.2.zip
rm libtorch-macos-x86_64-2.1.2.zip

# 環境変数設定は上記と同じ
```

#### Linux (x86_64)

```bash
# LibTorchダウンロード（CPU版）
cd ~/local
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu.zip

# 環境変数設定（~/.bashrcに追加）
echo 'export LIBTORCH_HOME=$HOME/local/libtorch' >> ~/.bashrc
echo 'export CMAKE_PREFIX_PATH=$LIBTORCH_HOME:$CMAKE_PREFIX_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LIBTORCH_HOME/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 方法2: CUDA対応版のインストール

#### CUDA 11.8版

```bash
# Linux用
cd ~/local
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu118.zip
rm libtorch-cxx11-abi-shared-with-deps-2.1.2+cu118.zip
```

#### CUDA 12.1版

```bash
# Linux用
cd ~/local
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
rm libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
```

### 方法3: Homebrewを使用（macOSのみ）

```bash
# Homebrewでインストール
brew install pytorch

# シンボリックリンクを作成
ln -s $(brew --prefix)/opt/pytorch/lib ~/local/libtorch
```

## インストール確認

### 1. ファイル構造の確認

```bash
# 正しくインストールされているか確認
ls -la ~/local/libtorch/
```

期待される構造：
```
libtorch/
├── bin/          # 実行可能ファイル
├── include/      # ヘッダファイル
│   ├── ATen/
│   ├── c10/
│   ├── torch/
│   └── ...
├── lib/          # ライブラリファイル
│   ├── libc10.dylib (macOS) / libc10.so (Linux)
│   ├── libtorch.dylib / libtorch.so
│   ├── libtorch_cpu.dylib / libtorch_cpu.so
│   └── ...
└── share/        # CMake設定ファイル
    └── cmake/
        └── Torch/
```

### 2. 簡単なテストプログラム

`test_libtorch.cpp`を作成：

```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    // LibTorchバージョン確認
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    
    // 簡単なテンソル操作
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random tensor:\n" << tensor << std::endl;
    
    // CUDA利用可能か確認
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA not available. Training on CPU." << std::endl;
    }
    
    return 0;
}
```

コンパイルと実行：

```bash
# コンパイル
g++ -std=c++17 \
    -I$HOME/local/libtorch/include \
    -I$HOME/local/libtorch/include/torch/csrc/api/include \
    -L$HOME/local/libtorch/lib \
    -lc10 -ltorch_cpu -ltorch \
    -Wl,-rpath,$HOME/local/libtorch/lib \
    test_libtorch.cpp -o test_libtorch

# 実行
./test_libtorch
```

## トラブルシューティング

### エラー: "cannot find -ltorch"

```bash
# ライブラリパスを確認
echo $LIBTORCH_HOME
ls -la $LIBTORCH_HOME/lib/

# CMakeキャッシュをクリア
rm -rf build/CMakeCache.txt
```

### エラー: "Library not loaded: @rpath/libc10.dylib"

```bash
# macOSの場合
install_name_tool -add_rpath $HOME/local/libtorch/lib your_executable

# または環境変数で設定
export DYLD_LIBRARY_PATH=$HOME/local/libtorch/lib:$DYLD_LIBRARY_PATH
```

### エラー: "version GLIBCXX_3.4.26 not found"

```bash
# GCCのバージョンを確認
gcc --version

# 必要に応じて新しいGCCをインストール
sudo apt-get update
sudo apt-get install gcc-9 g++-9
```

### CMakeでLibTorchが見つからない

`CMakeLists.txt`に以下を追加：

```cmake
# LibTorchの場所を明示的に指定
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{HOME}/local/libtorch")

# または
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/local/libtorch")

find_package(Torch REQUIRED)
```

## Fish Shell用の設定

Fish shellを使用している場合は、`~/.config/fish/config.fish`に以下を追加：

```fish
# LibTorch設定
set -gx LIBTORCH_HOME $HOME/local/libtorch
set -gx CMAKE_PREFIX_PATH $LIBTORCH_HOME $CMAKE_PREFIX_PATH
set -gx DYLD_LIBRARY_PATH $LIBTORCH_HOME/lib $DYLD_LIBRARY_PATH  # macOS
set -gx LD_LIBRARY_PATH $LIBTORCH_HOME/lib $LD_LIBRARY_PATH      # Linux
```

## バージョン管理

複数のLibTorchバージョンを管理する場合：

```bash
# バージョン別にインストール
cd ~/local
mkdir libtorch-2.1.2
cd libtorch-2.1.2
# ... ダウンロードと展開 ...

# シンボリックリンクで切り替え
ln -sfn ~/local/libtorch-2.1.2 ~/local/libtorch
```

## アンインストール

```bash
# LibTorchディレクトリを削除
rm -rf ~/local/libtorch

# 環境変数の設定を削除（.zshrc/.bashrcから該当行を削除）
```

## 参考リンク

- [PyTorch公式サイト - LibTorchダウンロード](https://pytorch.org/get-started/locally/)
- [LibTorch C++ API ドキュメント](https://pytorch.org/cppdocs/)
- [PyTorch GitHub リポジトリ](https://github.com/pytorch/pytorch)

---

**注意**: LibTorchのバージョンはcrlGRUライブラリとの互換性を確認してください。推奨バージョンは2.0以降です。
