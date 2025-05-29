/**
 * @file LwTT.cpp
 * @brief Implementation of library initialization and utility functions
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/LwTT.hpp"
#include <thread>
#include <cstdio>

namespace LwTT {

// Global state
static bool g_initialized = false;
static LibraryConfig g_config;

bool Initialize(int num_threads, bool enable_simd) {
    LibraryConfig config;
    config.num_threads = num_threads;
    config.enable_simd = enable_simd;
    return Initialize(config);
}

bool Initialize(const LibraryConfig& config) {
    if (g_initialized) {
        return true;
    }
    
    g_config = config;
    
    // Initialize threading
    if (g_config.num_threads <= 0) {
        g_config.num_threads = std::thread::hardware_concurrency();
    }
    
    g_initialized = true;
    return true;
}

void Cleanup() {
    if (g_initialized) {
        g_initialized = false;
    }
}

const char* GetVersion() {
    return LWTT_VERSION_STRING;
}

const char* GetBuildInfo() {
    static char build_info[256];
    std::snprintf(build_info, sizeof(build_info), 
                 "Built on %s %s with C++ compiler", __DATE__, __TIME__);
    return build_info;
}

bool IsSIMDSupported() {
    // For simplicity, assume SIMD is supported on modern systems
    return true;
}

int GetHardwareConcurrency() {
    return static_cast<int>(std::thread::hardware_concurrency());
}

} // namespace LwTT
