/**
 * @file Platform.hpp
 * @brief Platform-specific definitions and utilities for LwTT
 */

#pragma once

#include <string>
#include <algorithm>

#ifdef LWTT_HAS_THREADS
    #include <thread>
#endif

#ifdef LWTT_ENABLE_OPENMP
    #include <omp.h>
#endif

namespace crllwtt {
namespace Platform {

inline const char* GetPlatformName() {
#ifdef LWTT_PLATFORM_APPLE
    return "Apple";
#elif defined(LWTT_PLATFORM_WIN32)
    return "Windows";
#elif defined(LWTT_PLATFORM_LINUX)
    return "Linux";
#elif defined(LWTT_PLATFORM_BSD)
    return "BSD";
#elif defined(LWTT_PLATFORM_UNIX)
    return "Unix";
#else
    return "Unknown";
#endif
}

inline const char* GetArchitectureName() {
#ifdef LWTT_ARCH_X86_64
    return "x86_64";
#elif defined(LWTT_ARCH_ARM64)
    return "ARM64";
#elif defined(LWTT_ARCH_X86)
    return "x86";
#else
    return "Unknown";
#endif
}

inline const char* GetCompilerName() {
#ifdef LWTT_COMPILER_GCC
    return "GCC";
#elif defined(LWTT_COMPILER_APPLE_CLANG)
    return "Apple Clang";
#elif defined(LWTT_COMPILER_LLVM_CLANG)
    return "LLVM Clang";
#elif defined(LWTT_COMPILER_CLANG)
    return "Clang";
#elif defined(LWTT_COMPILER_MSVC)
    return "MSVC";
#else
    return "Unknown";
#endif
}

inline unsigned int GetHardwareConcurrency() {
#ifdef LWTT_HAS_THREADS
    return std::thread::hardware_concurrency();
#else
    return 1;
#endif
}

inline int GetOptimalThreadCount() {
#ifdef LWTT_ENABLE_OPENMP
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();
    
    #ifdef LWTT_PLATFORM_APPLE
        return std::min(max_threads, num_procs * 3 / 4);
    #elif defined(LWTT_PLATFORM_WIN32)
        return std::min(max_threads, num_procs);
    #else
        return max_threads;
    #endif
#else
    return 1;
#endif
}

} // namespace Platform
} // namespace crllwtt

// Performance macros
#ifdef LWTT_COMPILER_GCC
    #define LWTT_FORCE_INLINE __attribute__((always_inline)) inline
    #define LWTT_LIKELY(x)   __builtin_expect(\!\!(x), 1)
    #define LWTT_UNLIKELY(x) __builtin_expect(\!\!(x), 0)
#elif defined(LWTT_COMPILER_CLANG)
    #define LWTT_FORCE_INLINE __attribute__((always_inline)) inline
    #define LWTT_LIKELY(x)   __builtin_expect(\!\!(x), 1)
    #define LWTT_UNLIKELY(x) __builtin_expect(\!\!(x), 0)
#elif defined(LWTT_COMPILER_MSVC)
    #define LWTT_FORCE_INLINE __forceinline
    #define LWTT_LIKELY(x)   (x)
    #define LWTT_UNLIKELY(x) (x)
#else
    #define LWTT_FORCE_INLINE inline
    #define LWTT_LIKELY(x)   (x)
    #define LWTT_UNLIKELY(x) (x)
#endif

// SIMD support
#ifdef LWTT_ENABLE_SIMD
    #ifdef LWTT_ARCH_X86_64
        #include <immintrin.h>
        #define LWTT_HAS_SSE 1
        #define LWTT_HAS_AVX 1
    #elif defined(LWTT_ARCH_ARM64)
        #include <arm_neon.h>
        #define LWTT_HAS_NEON 1
    #endif
#endif
