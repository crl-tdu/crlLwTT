/**
 * @file LwTT.hpp
 * @brief Lightweight Time-aware Transformer Library - Main Header
 * @version 1.0.0
 * @date 2025-05-25
 *
 * @copyright Copyright (c) 2025
 *
 * Main header file for the Lightweight Time-aware Transformer (LwTT) library.
 * This library provides high-performance, memory-efficient implementations of
 * time-aware transformer architectures optimized for real-time applications.
 */

#ifndef LWTT_HPP
#define LWTT_HPP

// Version information
#define LWTT_VERSION_MAJOR 1
#define LWTT_VERSION_MINOR 0
#define LWTT_VERSION_PATCH 0
#define LWTT_VERSION_STRING "1.0.0"

// Core components
#include "LwTT/core/Tensor.hpp"
#include "LwTT/core/Transformer.hpp"
#include "LwTT/core/TimeEncoding.hpp"
#include "LwTT/core/SparseAttention.hpp"
#include "LwTT/core/STATransformer.hpp"

// Layer components
#include "LwTT/layers/TransformerBlock.hpp"

// Utilities
#include "LwTT/utils/Memory.hpp"
#include "LwTT/utils/Threading.hpp"

namespace crl {
namespace lwtt {

/**
 * @brief Library initialization
 * @param num_threads Number of threads to use (0 = auto-detect)
 * @param enable_simd Enable SIMD optimizations
 * @return true if initialization successful
 */
bool Initialize(int num_threads = 0, bool enable_simd = true);

/**
 * @brief Library cleanup
 */
void Cleanup();

/**
 * @brief Get library version string
 * @return Version string
 */
const char* GetVersion();

/**
 * @brief Get build information
 * @return Build information string
 */
const char* GetBuildInfo();

/**
 * @brief Check if SIMD optimizations are available
 * @return true if SIMD is supported
 */
bool IsSIMDSupported();

/**
 * @brief Get number of available hardware threads
 * @return Number of threads
 */
int GetHardwareConcurrency();

/**
 * @brief Library configuration structure
 */
struct LibraryConfig {
    int num_threads = 0;        ///< Number of threads (0 = auto)
    bool enable_simd = true;    ///< Enable SIMD optimizations
    bool enable_logging = true; ///< Enable logging
    int log_level = 2;          ///< Log level (0=Error, 1=Warning, 2=Info, 3=Debug)
    size_t memory_pool_size_mb = 512; ///< Memory pool size in MB
    bool enable_profiling = false;    ///< Enable performance profiling
};

/**
 * @brief Initialize library with custom configuration
 * @param config Configuration structure
 * @return true if initialization successful
 */
bool Initialize(const LibraryConfig& config);

} // namespace lwtt
} // namespace crl

#endif // LWTT_HPP