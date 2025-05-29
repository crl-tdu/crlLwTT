#!/bin/bash

# LwTT (Lightweight Time-aware Transformer) Build Script
# Version: 1.0.0
# Date: 2025-05-25

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
BUILD_TYPE="Release"
BUILD_DIR="$PROJECT_ROOT/build"
INSTALL_PREFIX="$PROJECT_ROOT/install"
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CLEAN_BUILD=false
RUN_TESTS=true
BUILD_EXAMPLES=true
BUILD_BENCHMARKS=true
ENABLE_SIMD=true
ENABLE_OPENMP=true
USE_EIGEN=true
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for LwTT (Lightweight Time-aware Transformer) library

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Build type: Debug, Release, RelWithDebInfo (default: Release)
    -b, --build-dir DIR     Build directory (default: build)
    -i, --install-prefix DIR Install prefix (default: install)
    -j, --jobs NUM          Number of parallel jobs (default: auto-detect)
    -c, --clean             Clean build (remove build directory first)
    --no-tests              Don't build tests
    --no-examples           Don't build examples
    --no-benchmarks         Don't build benchmarks
    --no-simd               Disable SIMD optimizations
    --no-openmp             Disable OpenMP support
    --no-eigen              Don't use Eigen library
    -v, --verbose           Verbose output

EXAMPLES:
    $0                      # Build with default settings
    $0 -t Debug -c          # Clean debug build
    $0 --no-tests -j 8      # Release build without tests, 8 jobs
    $0 -t RelWithDebInfo -v # Release with debug info, verbose output

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -i|--install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES=false
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS=false
            shift
            ;;
        --no-simd)
            ENABLE_SIMD=false
            shift
            ;;
        --no-openmp)
            ENABLE_OPENMP=false
            shift
            ;;
        --no-eigen)
            USE_EIGEN=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    Debug|Release|RelWithDebInfo|MinSizeRel)
        ;;
    *)
        print_error "Invalid build type: $BUILD_TYPE"
        print_info "Valid types: Debug, Release, RelWithDebInfo, MinSizeRel"
        exit 1
        ;;
esac

# Print configuration
print_info "LwTT Build Configuration:"
echo "  Project Root:    $PROJECT_ROOT"
echo "  Build Type:      $BUILD_TYPE"
echo "  Build Directory: $BUILD_DIR"
echo "  Install Prefix:  $INSTALL_PREFIX"
echo "  Parallel Jobs:   $NUM_JOBS"
echo "  Clean Build:     $CLEAN_BUILD"
echo "  Build Tests:     $RUN_TESTS"
echo "  Build Examples:  $BUILD_EXAMPLES"
echo "  Build Benchmarks: $BUILD_BENCHMARKS"
echo "  Enable SIMD:     $ENABLE_SIMD"
echo "  Enable OpenMP:   $ENABLE_OPENMP"
echo "  Use Eigen:       $USE_EIGEN"
echo "  Verbose:         $VERBOSE"
echo ""

# Check for required tools
print_info "Checking required tools..."

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake is required but not installed"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_success "CMake found: $CMAKE_VERSION"

# Check compiler
if command -v g++ &> /dev/null; then
    COMPILER="g++"
    COMPILER_VERSION=$(g++ --version | head -n1)
elif command -v clang++ &> /dev/null; then
    COMPILER="clang++"
    COMPILER_VERSION=$(clang++ --version | head -n1)
else
    print_error "No suitable C++ compiler found (g++ or clang++)"
    exit 1
fi
print_success "Compiler found: $COMPILER_VERSION"

# Check for optional dependencies
if $USE_EIGEN; then
    if command -v pkg-config &> /dev/null && pkg-config --exists eigen3; then
        EIGEN_VERSION=$(pkg-config --modversion eigen3)
        print_success "Eigen3 found: $EIGEN_VERSION"
    else
        print_warning "Eigen3 not found via pkg-config, will try to find via CMake"
    fi
fi

# Clean build if requested
if $CLEAN_BUILD; then
    print_info "Cleaning build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    fi
fi

# Create build directory
print_info "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake options
CMAKE_OPTIONS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    "-DLWTT_BUILD_TESTS=$RUN_TESTS"
    "-DLWTT_BUILD_EXAMPLES=$BUILD_EXAMPLES"
    "-DLWTT_BUILD_BENCHMARKS=$BUILD_BENCHMARKS"
    "-DLWTT_ENABLE_SIMD=$ENABLE_SIMD"
    "-DLWTT_ENABLE_OPENMP=$ENABLE_OPENMP"
    "-DLWTT_USE_EIGEN=$USE_EIGEN"
)

if $VERBOSE; then
    CMAKE_OPTIONS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi

# Run CMake configuration
print_info "Configuring with CMake..."
if $VERBOSE; then
    cmake "${CMAKE_OPTIONS[@]}" "$PROJECT_ROOT"
else
    cmake "${CMAKE_OPTIONS[@]}" "$PROJECT_ROOT" > cmake_config.log 2>&1
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed. See cmake_config.log for details"
        tail -20 cmake_config.log
        exit 1
    fi
fi
print_success "CMake configuration completed"

# Build the project
print_info "Building LwTT library..."
BUILD_START_TIME=$(date +%s)

if $VERBOSE; then
    cmake --build . --config "$BUILD_TYPE" --parallel "$NUM_JOBS"
else
    cmake --build . --config "$BUILD_TYPE" --parallel "$NUM_JOBS" > build.log 2>&1
    if [ $? -ne 0 ]; then
        print_error "Build failed. See build.log for details"
        tail -30 build.log
        exit 1
    fi
fi

BUILD_END_TIME=$(date +%s)
BUILD_TIME=$((BUILD_END_TIME - BUILD_START_TIME))
print_success "Build completed in ${BUILD_TIME} seconds"

# Run tests if requested
if $RUN_TESTS; then
    print_info "Running tests..."
    if command -v ctest &> /dev/null; then
        if $VERBOSE; then
            ctest --verbose --parallel "$NUM_JOBS"
        else
            ctest --output-on-failure --parallel "$NUM_JOBS" > test.log 2>&1
            if [ $? -ne 0 ]; then
                print_error "Some tests failed. See test.log for details"
                tail -20 test.log
                exit 1
            fi
        fi
        print_success "All tests passed"
    else
        print_warning "CTest not found, skipping tests"
    fi
fi

# Install the library
print_info "Installing LwTT library..."
if $VERBOSE; then
    cmake --install . --config "$BUILD_TYPE"
else
    cmake --install . --config "$BUILD_TYPE" > install.log 2>&1
    if [ $? -ne 0 ]; then
        print_error "Installation failed. See install.log for details"
        exit 1
    fi
fi
print_success "Installation completed to $INSTALL_PREFIX"

# Print summary
echo ""
print_success "Build Summary:"
echo "  Build Type:      $BUILD_TYPE"
echo "  Build Time:      ${BUILD_TIME} seconds"
echo "  Install Path:    $INSTALL_PREFIX"
echo "  Library:         $INSTALL_PREFIX/lib/libLwTT.*"
echo "  Headers:         $INSTALL_PREFIX/include/LwTT/"
echo "  Examples:        $INSTALL_PREFIX/bin/ (if built)"
echo ""

# Show next steps
print_info "Next Steps:"
echo "  1. Add $INSTALL_PREFIX/lib to your LD_LIBRARY_PATH"
echo "  2. Add $INSTALL_PREFIX/include to your include path"
echo "  3. Link with -lLwTT in your projects"
echo "  4. See examples in $PROJECT_ROOT/examples/"
echo ""

if [ -f "$PROJECT_ROOT/examples/basic_usage/simple_transformer" ]; then
    print_info "Try running a simple example:"
    echo "  cd $PROJECT_ROOT/examples/basic_usage"
    echo "  ./simple_transformer"
fi

print_success "LwTT build completed successfully!"