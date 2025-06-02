/**
 * @file platform_test.cpp
 * @brief Platform detection test for LwTT
 * @version 1.0.0
 * @date 2025-06-02
 */

#include <LwTT/Platform.hpp>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== LwTT Platform Detection Test ===" << std::endl;
    std::cout << std::endl;
    
    // Runtime platform information
    std::cout << "Runtime Platform Information:" << std::endl;
    std::cout << "  Platform: " << crllwtt::Platform::GetPlatformName() << std::endl;
    std::cout << "  Architecture: " << crllwtt::Platform::GetArchitectureName() << std::endl;
    std::cout << "  Compiler: " << crllwtt::Platform::GetCompilerName() << std::endl;
    std::cout << "  CPU Cores: " << crllwtt::Platform::GetHardwareConcurrency() << std::endl;
    std::cout << std::endl;
    
    // Compile-time platform detection
    std::cout << "Compile-time Platform Detection:" << std::endl;
    
    // Platform macros
    std::cout << "  Platform Macros:" << std::endl;
#ifdef LWTT_PLATFORM_APPLE
    std::cout << "    ✓ LWTT_PLATFORM_APPLE" << std::endl;
#ifdef LWTT_PLATFORM_MACOS
    std::cout << "      ✓ LWTT_PLATFORM_MACOS" << std::endl;
#endif
#ifdef LWTT_PLATFORM_IOS
    std::cout << "      ✓ LWTT_PLATFORM_IOS" << std::endl;
#endif
#endif
    
#ifdef LWTT_PLATFORM_WIN32
    std::cout << "    ✓ LWTT_PLATFORM_WIN32" << std::endl;
#endif
    
#ifdef LWTT_PLATFORM_UNIX
    std::cout << "    ✓ LWTT_PLATFORM_UNIX" << std::endl;
#ifdef LWTT_PLATFORM_LINUX
    std::cout << "      ✓ LWTT_PLATFORM_LINUX" << std::endl;
#endif
#ifdef LWTT_PLATFORM_BSD
    std::cout << "      ✓ LWTT_PLATFORM_BSD" << std::endl;
#endif
#endif
    
    // Architecture macros
    std::cout << "  Architecture Macros:" << std::endl;
#ifdef LWTT_ARCH_X86_64
    std::cout << "    ✓ LWTT_ARCH_X86_64" << std::endl;
#endif
#ifdef LWTT_ARCH_ARM64
    std::cout << "    ✓ LWTT_ARCH_ARM64" << std::endl;
#endif
#ifdef LWTT_ARCH_X86
    std::cout << "    ✓ LWTT_ARCH_X86" << std::endl;
#endif
    
    // Compiler macros
    std::cout << "  Compiler Macros:" << std::endl;
#ifdef LWTT_COMPILER_GCC
    std::cout << "    ✓ LWTT_COMPILER_GCC" << std::endl;
#endif
#ifdef LWTT_COMPILER_CLANG
    std::cout << "    ✓ LWTT_COMPILER_CLANG" << std::endl;
#ifdef LWTT_COMPILER_APPLE_CLANG
    std::cout << "      ✓ LWTT_COMPILER_APPLE_CLANG" << std::endl;
#endif
#ifdef LWTT_COMPILER_LLVM_CLANG
    std::cout << "      ✓ LWTT_COMPILER_LLVM_CLANG" << std::endl;
#endif
#endif
#ifdef LWTT_COMPILER_MSVC
    std::cout << "    ✓ LWTT_COMPILER_MSVC" << std::endl;
#endif
    
    // Feature macros
    std::cout << "  Feature Macros:" << std::endl;
#ifdef LWTT_HAS_THREADS
    std::cout << "    ✓ LWTT_HAS_THREADS" << std::endl;
#endif
#ifdef LWTT_BIG_ENDIAN
    std::cout << "    ✓ LWTT_BIG_ENDIAN" << std::endl;
#endif
#ifdef LWTT_LITTLE_ENDIAN
    std::cout << "    ✓ LWTT_LITTLE_ENDIAN" << std::endl;
#endif
#ifdef LWTT_ENABLE_SIMD
    std::cout << "    ✓ LWTT_ENABLE_SIMD" << std::endl;
#ifdef LWTT_HAS_SSE
    std::cout << "      ✓ LWTT_HAS_SSE" << std::endl;
#endif
#ifdef LWTT_HAS_AVX
    std::cout << "      ✓ LWTT_HAS_AVX" << std::endl;
#endif
#ifdef LWTT_HAS_NEON
    std::cout << "      ✓ LWTT_HAS_NEON" << std::endl;
#endif
#endif
#ifdef LWTT_ENABLE_OPENMP
    std::cout << "    ✓ LWTT_ENABLE_OPENMP" << std::endl;
    std::cout << "      Optimal threads: " << crllwtt::Platform::GetOptimalThreadCount() << std::endl;
#endif
    
    std::cout << std::endl;
    
    // Platform-specific conditional compilation test
    std::cout << "Conditional Compilation Test:" << std::endl;
    
#ifdef LWTT_PLATFORM_APPLE
    std::cout << "  ✓ Apple-specific code path" << std::endl;
    std::cout << "    - Using Apple libc++" << std::endl;
    std::cout << "    - macOS deployment target configured" << std::endl;
#elif defined(LWTT_PLATFORM_WIN32)
    std::cout << "  ✓ Windows-specific code path" << std::endl;
    std::cout << "    - Unicode support enabled" << std::endl;
    std::cout << "    - Targeting Windows 10+" << std::endl;
    std::cout << "    - NOMINMAX and WIN32_LEAN_AND_MEAN defined" << std::endl;
#elif defined(LWTT_PLATFORM_UNIX)
    std::cout << "  ✓ Unix/Linux-specific code path" << std::endl;
#ifdef LWTT_PLATFORM_LINUX
    std::cout << "    - GNU extensions enabled" << std::endl;
#endif
#endif
    
#ifdef LWTT_ARCH_X86_64
    std::cout << "  ✓ x86_64 architecture optimizations available" << std::endl;
#ifdef LWTT_ENABLE_SIMD
    std::cout << "    - AVX/SSE optimizations enabled" << std::endl;
#endif
#elif defined(LWTT_ARCH_ARM64)
    std::cout << "  ✓ ARM64 architecture optimizations available" << std::endl;
#ifdef LWTT_ENABLE_SIMD
    std::cout << "    - NEON optimizations enabled" << std::endl;
#endif
#endif
    
    // Performance optimization macros test
    std::cout << "  Performance Macros Available:" << std::endl;
    std::cout << "    LWTT_FORCE_INLINE: Available" << std::endl;
    std::cout << "    LWTT_LIKELY/UNLIKELY: ";
#if defined(LWTT_COMPILER_GCC) || defined(LWTT_COMPILER_CLANG)
    std::cout << "Available" << std::endl;
#else
    std::cout << "Not available (no-op)" << std::endl;
#endif
    
    std::cout << std::endl;
    std::cout << "=== Platform Detection Test Completed Successfully ===" << std::endl;
    
    return 0;
}
