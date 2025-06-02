# PlatformDetection.cmake
# Platform detection and macro definitions for LwTT multi-platform support
# 
# This module detects the target platform and sets appropriate preprocessor definitions
# for conditional compilation in C++ source code.
#
# Defines the following macros:
#   LWTT_PLATFORM_APPLE   - macOS/iOS platforms
#   LWTT_PLATFORM_WIN32   - Windows platforms
#   LWTT_PLATFORM_UNIX    - Unix-like platforms (Linux, BSD, etc.)
#   LWTT_PLATFORM_LINUX   - Specifically Linux
#   LWTT_PLATFORM_BSD     - BSD variants
#
# Additional architecture and compiler information:
#   LWTT_ARCH_X86_64      - x86_64 architecture
#   LWTT_ARCH_ARM64       - ARM64/AArch64 architecture
#   LWTT_ARCH_X86         - x86 32-bit architecture
#   LWTT_COMPILER_GCC     - GNU GCC compiler
#   LWTT_COMPILER_CLANG   - Clang compiler
#   LWTT_COMPILER_MSVC    - Microsoft Visual C++ compiler

cmake_minimum_required(VERSION 3.16)

# Function to add platform-specific definitions
function(lwtt_detect_platform)
    message(STATUS "Detecting platform for LwTT...")
    
    # ========================================================================
    # Platform Detection
    # ========================================================================
    
    # Apple platforms (macOS, iOS, watchOS, tvOS)
    if(APPLE)
        add_compile_definitions(LWTT_PLATFORM_APPLE=1)
        message(STATUS "Platform: Apple (macOS/iOS)")
        
        # Specific Apple platform detection
        if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
            add_compile_definitions(LWTT_PLATFORM_MACOS=1)
            message(STATUS "  Specific platform: macOS")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
            add_compile_definitions(LWTT_PLATFORM_IOS=1)
            message(STATUS "  Specific platform: iOS")
        endif()
        
    # Windows platforms
    elseif(WIN32)
        add_compile_definitions(LWTT_PLATFORM_WIN32=1)
        message(STATUS "Platform: Windows")
        
        # Windows version detection
        if(CMAKE_SYSTEM_VERSION)
            message(STATUS "  Windows version: ${CMAKE_SYSTEM_VERSION}")
        endif()
        
    # Unix-like platforms
    elseif(UNIX)
        add_compile_definitions(LWTT_PLATFORM_UNIX=1)
        message(STATUS "Platform: Unix-like")
        
        # Specific Unix platform detection
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_compile_definitions(LWTT_PLATFORM_LINUX=1)
            message(STATUS "  Specific platform: Linux")
            
            # Linux distribution detection (optional)
            if(EXISTS "/etc/os-release")
                file(READ "/etc/os-release" OS_RELEASE_CONTENT)
                string(REGEX MATCH "ID=([a-zA-Z0-9_]+)" _ "${OS_RELEASE_CONTENT}")
                if(CMAKE_MATCH_1)
                    message(STATUS "  Linux distribution: ${CMAKE_MATCH_1}")
                endif()
            endif()
            
        elseif(CMAKE_SYSTEM_NAME MATCHES "BSD")
            add_compile_definitions(LWTT_PLATFORM_BSD=1)
            message(STATUS "  Specific platform: BSD variant (${CMAKE_SYSTEM_NAME})")
            
        else()
            message(STATUS "  Specific platform: Other Unix (${CMAKE_SYSTEM_NAME})")
        endif()
        
    else()
        message(WARNING "Unknown platform: ${CMAKE_SYSTEM_NAME}")
        add_compile_definitions(LWTT_PLATFORM_UNKNOWN=1)
    endif()
    
    # ========================================================================
    # Architecture Detection
    # ========================================================================
    
    message(STATUS "Detecting architecture...")
    
    # Normalize processor architecture
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SYSTEM_PROCESSOR_LOWER)
    
    if(SYSTEM_PROCESSOR_LOWER MATCHES "x86_64|amd64")
        add_compile_definitions(LWTT_ARCH_X86_64=1)
        message(STATUS "Architecture: x86_64")
        
    elseif(SYSTEM_PROCESSOR_LOWER MATCHES "arm64|aarch64")
        add_compile_definitions(LWTT_ARCH_ARM64=1)
        message(STATUS "Architecture: ARM64")
        
    elseif(SYSTEM_PROCESSOR_LOWER MATCHES "i386|i686|x86")
        add_compile_definitions(LWTT_ARCH_X86=1)
        message(STATUS "Architecture: x86 (32-bit)")
        
    else()
        add_compile_definitions(LWTT_ARCH_UNKNOWN=1)
        message(STATUS "Architecture: Unknown (${CMAKE_SYSTEM_PROCESSOR})")
    endif()
    
    # ========================================================================
    # Compiler Detection
    # ========================================================================
    
    message(STATUS "Detecting compiler...")
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_definitions(LWTT_COMPILER_GCC=1)
        message(STATUS "Compiler: GNU GCC ${CMAKE_CXX_COMPILER_VERSION}")
        
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_definitions(LWTT_COMPILER_CLANG=1)
        message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
        
        # Apple Clang vs LLVM Clang distinction
        if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            add_compile_definitions(LWTT_COMPILER_APPLE_CLANG=1)
            message(STATUS "  Variant: Apple Clang")
        else()
            add_compile_definitions(LWTT_COMPILER_LLVM_CLANG=1)
            message(STATUS "  Variant: LLVM Clang")
        endif()
        
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_definitions(LWTT_COMPILER_MSVC=1)
        message(STATUS "Compiler: Microsoft Visual C++ ${CMAKE_CXX_COMPILER_VERSION}")
        
    else()
        add_compile_definitions(LWTT_COMPILER_UNKNOWN=1)
        message(STATUS "Compiler: Unknown (${CMAKE_CXX_COMPILER_ID})")
    endif()
    
    # ========================================================================
    # Additional System Information
    # ========================================================================
    
    # Endianness detection
    include(TestBigEndian)
    test_big_endian(IS_BIG_ENDIAN)
    if(IS_BIG_ENDIAN)
        add_compile_definitions(LWTT_BIG_ENDIAN=1)
        message(STATUS "Byte order: Big endian")
    else()
        add_compile_definitions(LWTT_LITTLE_ENDIAN=1)
        message(STATUS "Byte order: Little endian")
    endif()
    
    # Thread support detection
    find_package(Threads QUIET)
    if(Threads_FOUND)
        add_compile_definitions(LWTT_HAS_THREADS=1)
        message(STATUS "Threading: Available")
    else()
        message(STATUS "Threading: Not available")
    endif()
    
    # ========================================================================
    # Platform-Specific Optimizations and Settings
    # ========================================================================
    
    # Apple-specific settings
    if(APPLE)
        # Use libc++ on Apple platforms
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++" PARENT_SCOPE)
        
        # macOS deployment target
        if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
            set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" PARENT_SCOPE)
            message(STATUS "macOS deployment target: 10.15 (default)")
        else()
            message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
    endif()
    
    # Windows-specific settings
    if(WIN32)
        # Unicode support
        add_compile_definitions(UNICODE _UNICODE)
        
        # Windows version targeting
        add_compile_definitions(_WIN32_WINNT=0x0A00)  # Windows 10
        
        # Disable problematic Windows macros
        add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
        
        message(STATUS "Windows: Unicode enabled, targeting Windows 10+")
    endif()
    
    # Linux-specific settings
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Enable GNU extensions
        add_compile_definitions(_GNU_SOURCE)
        message(STATUS "Linux: GNU extensions enabled")
    endif()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    message(STATUS "")
    message(STATUS "Platform Detection Summary:")
    message(STATUS "  System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
    message(STATUS "  Processor: ${CMAKE_SYSTEM_PROCESSOR}")
    message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "")
    
endfunction()

# Function to display platform-specific usage examples
function(lwtt_display_platform_usage_examples)
    message(STATUS "")
    message(STATUS "========================================================================")
    message(STATUS "Platform-Specific C++ Usage Examples")
    message(STATUS "========================================================================")
    message(STATUS "")
    message(STATUS "Use the following patterns in your C++ source code:")
    message(STATUS "")
    message(STATUS "// Platform detection")
    message(STATUS "#ifdef LWTT_PLATFORM_APPLE")
    message(STATUS "    // macOS/iOS specific code")
    message(STATUS "#elif defined(LWTT_PLATFORM_WIN32)")
    message(STATUS "    // Windows specific code")
    message(STATUS "#elif defined(LWTT_PLATFORM_UNIX)")
    message(STATUS "    // Unix/Linux specific code")
    message(STATUS "#endif")
    message(STATUS "")
    message(STATUS "// Architecture detection")
    message(STATUS "#ifdef LWTT_ARCH_X86_64")
    message(STATUS "    // x86_64 optimizations")
    message(STATUS "#elif defined(LWTT_ARCH_ARM64)")
    message(STATUS "    // ARM64 optimizations")
    message(STATUS "#endif")
    message(STATUS "")
    message(STATUS "// Compiler detection")
    message(STATUS "#ifdef LWTT_COMPILER_CLANG")
    message(STATUS "    // Clang-specific pragmas or features")
    message(STATUS "#elif defined(LWTT_COMPILER_GCC)")
    message(STATUS "    // GCC-specific features")
    message(STATUS "#elif defined(LWTT_COMPILER_MSVC)")
    message(STATUS "    // MSVC-specific features")
    message(STATUS "#endif")
    message(STATUS "")
    message(STATUS "========================================================================")
    message(STATUS "")
endfunction()

# Export the main function
macro(lwtt_setup_platform_detection)
    lwtt_detect_platform()
endmacro()
