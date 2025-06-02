cmake_minimum_required(VERSION 3.16)

# プロジェクトがトップレベルかチェック
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LWTT_MAIN_PROJECT TRUE)
else()
    set(LWTT_MAIN_PROJECT FALSE)
endif()

project(crlLwTT
    VERSION 1.0.0
    DESCRIPTION "Lightweight Time-aware Transformer Library"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include custom CMake modules (メインプロジェクトでのみ実行)
if(LWTT_MAIN_PROJECT)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(PlatformDetection)
    include(CompilerOptions)
    include(Dependencies)
    
    # Set up platform detection first (before any other configuration)
lwtt_setup_platform_detection()
else()
    # submodule使用時は簡略化された依存関係検索
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(LWTT_HAS_TORCH TRUE)
    endif()
    
    find_package(Eigen3 QUIET)
    find_package(OpenMP QUIET)
endif()

# ビルドオプション（submoduleの場合はデフォルト調整）
option(LWTT_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(LWTT_ENABLE_OPENMP "Enable OpenMP" ON)
option(LWTT_ENABLE_PROFILING "Enable profiling" OFF)
option(LWTT_ENABLE_QUANTIZATION "Enable quantization support" ON)
option(LWTT_USE_EIGEN "Use Eigen for linear algebra" ON)
option(LWTT_BUILD_SHARED "Build shared library" ${LWTT_MAIN_PROJECT})
option(LWTT_BUILD_STATIC "Build static library" ON)
option(LWTT_ENABLE_COVERAGE "Enable code coverage" OFF)
option(LWTT_BUILD_DOCS "Build documentation" OFF)  # 無効化 - docsディレクトリ構造が不完全
option(LWTT_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)  # 無効化 - bindingsディレクトリが存在しない
option(LWTT_BUILD_TESTS "Build tests, examples, and benchmarks" ${LWTT_MAIN_PROJECT})
option(LWTT_INSTALL "Install LwTT" ${LWTT_MAIN_PROJECT})

# Source files - using automatic file discovery
file(GLOB_RECURSE LWTT_SOURCES 
    "src/core/*.cpp" 
    "src/layers/*.cpp" 
    "src/utils/*.cpp" 
    "src/io/*.cpp"
)

# Add optimization sources if enabled
if(LWTT_ENABLE_QUANTIZATION)
    file(GLOB_RECURSE LWTT_OPTIMIZATION_SOURCES "src/optimization/*.cpp")
    list(APPEND LWTT_SOURCES ${LWTT_OPTIMIZATION_SOURCES})
endif()

# Header files - using automatic file discovery
file(GLOB_RECURSE LWTT_HEADERS 
    "include/LwTT/*.hpp"
)

# ==========================================
# INTERFACEライブラリターゲット（常に作成）
# ==========================================

add_library(LwTT_headers INTERFACE)
target_include_directories(LwTT_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 依存関係をINTERFACEライブラリに設定
if(LWTT_HAS_TORCH AND Torch_FOUND)
    target_include_directories(LwTT_headers INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(LwTT_headers INTERFACE ${TORCH_LIBRARIES})
    target_compile_definitions(LwTT_headers INTERFACE LWTT_HAS_TORCH)
endif()

if(LWTT_USE_EIGEN AND (Eigen3_FOUND OR TARGET Eigen3::Eigen))
    target_link_libraries(LwTT_headers INTERFACE Eigen3::Eigen)
endif()

if(LWTT_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(LwTT_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

target_compile_features(LwTT_headers INTERFACE cxx_std_23)

# Build shared library
if(LWTT_BUILD_SHARED)
    add_library(LwTT_shared SHARED ${LWTT_SOURCES} ${LWTT_HEADERS})
    target_link_libraries(LwTT_shared PUBLIC LwTT_headers)
    set_target_properties(LwTT_shared PROPERTIES
        OUTPUT_NAME LwTT
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
    )

    # Set RPATH for shared library (メインプロジェクトのみ)
    if(LWTT_MAIN_PROJECT)
        if(APPLE)
            set_target_properties(LwTT_shared PROPERTIES
                INSTALL_RPATH "@loader_path;@loader_path/../lib;${LIBTORCH_LIB_DIR};$ENV{HOME}/local/lib"
                BUILD_RPATH "${CMAKE_BINARY_DIR};${LIBTORCH_LIB_DIR};$ENV{HOME}/local/lib"
            )
        elseif(UNIX AND NOT APPLE)
            set_target_properties(LwTT_shared PROPERTIES
                INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${LIBTORCH_LIB_DIR}:$ENV{HOME}/local/lib"
                BUILD_RPATH "${CMAKE_BINARY_DIR}:${LIBTORCH_LIB_DIR}:$ENV{HOME}/local/lib"
            )
        endif()
    endif()
    
    if(LWTT_MAIN_PROJECT)
        message(STATUS "✓ Shared library target: LwTT_shared")
    endif()
endif()

# Build static library (推奨、submoduleでデフォルト)
if(LWTT_BUILD_STATIC)
    add_library(LwTT_static STATIC ${LWTT_SOURCES} ${LWTT_HEADERS})
    target_link_libraries(LwTT_static PUBLIC LwTT_headers)
    set_target_properties(LwTT_static PROPERTIES
        OUTPUT_NAME LwTT
        VERSION ${PROJECT_VERSION}
        POSITION_INDEPENDENT_CODE ON
    )
    
    # メインターゲットエイリアス
    if(NOT TARGET LwTT)
        add_library(LwTT ALIAS LwTT_static)
    endif()
    
    if(LWTT_MAIN_PROJECT)
        message(STATUS "✓ Static library target: LwTT_static")
    endif()
endif()

# Modern CMake エイリアス
if(LWTT_BUILD_SHARED AND NOT TARGET LwTT)
    add_library(LwTT ALIAS LwTT_shared)
endif()

# Python bindings (現在無効化 - bindingsディレクトリが存在しない)
# if(LWTT_BUILD_PYTHON_BINDINGS AND LWTT_MAIN_PROJECT)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings")
#         add_subdirectory(bindings)
#     else()
#         message(STATUS "Bindings directory not found, skipping Python bindings")
#     endif()
# endif()

# Build documentation (現在無効化 - docsの自動生成を無効)
# if(LWTT_BUILD_DOCS AND LWTT_MAIN_PROJECT)
#     find_package(Doxygen)
#     if(DOXYGEN_FOUND)
#         set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
#         set(DOXYGEN_GENERATE_HTML YES)
#         set(DOXYGEN_GENERATE_XML YES)
#         
#         configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
#                       ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
#         
#         add_custom_target(docs
#             COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#             COMMENT "Generating API documentation with Doxygen"
#             VERBATIM)
#     else()
#         message(WARNING "Doxygen not found, documentation will not be built")
#     endif()
# endif()

# Build tests
# Tests (includes examples and benchmarks)
if(LWTT_BUILD_TESTS AND LWTT_MAIN_PROJECT)
    enable_testing()
    
    # Find all .cpp files in tests directory
    file(GLOB TEST_SOURCES "tests/*.cpp")
    
    # Create executable for each test file
    foreach(TEST_SOURCE ${TEST_SOURCES})
        # Get filename without extension
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        
        # Create executable
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        
        # Link with LwTT library
        if(LWTT_BUILD_SHARED AND TARGET LwTT_shared)
            target_link_libraries(${TEST_NAME} PRIVATE LwTT_shared)
        elseif(LWTT_BUILD_STATIC AND TARGET LwTT_static)
            target_link_libraries(${TEST_NAME} PRIVATE LwTT_static)
        endif()
        
        # Link with GoogleTest for test files
        if(TARGET gtest AND TARGET gtest_main)
            target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main)
        endif()
        
        # Include directories
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        # Add compiler definitions if needed
        if(LWTT_HAS_TORCH AND Torch_FOUND)
            target_compile_definitions(${TEST_NAME} PRIVATE LWTT_HAS_TORCH)
        endif()
        
        # Add test to CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        
        # Set test properties
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT 300
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        
        message(STATUS "Added test executable: ${TEST_NAME}")
    endforeach()
    
    # Also add subdirectory if it exists (for compatibility with existing structure)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    endif()
endif()

# Install targets (条件付き)
if(LWTT_INSTALL)

# Install headers interface target first
install(TARGETS LwTT_headers
    EXPORT LwTTTargets
)

if(LWTT_BUILD_SHARED)
    install(TARGETS LwTT_shared
        EXPORT LwTTTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(LWTT_BUILD_STATIC)
    install(TARGETS LwTT_static
        EXPORT LwTTTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

# Install headers
install(DIRECTORY include/LwTT
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Install CMake config files
install(EXPORT LwTTTargets
    FILE LwTTTargets.cmake
    NAMESPACE LwTT::
    DESTINATION lib/cmake/LwTT
)

# Generate and install config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LwTTConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/LwTTConfig.cmake"
    INSTALL_DESTINATION lib/cmake/LwTT
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/LwTTConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LwTTConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LwTTConfigVersion.cmake"
    DESTINATION lib/cmake/LwTT
)

endif() # LWTT_INSTALL

# Package configuration
set(CPACK_PACKAGE_NAME "LwTT")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Lightweight Time-aware Transformer Library")
set(CPACK_PACKAGE_VENDOR "Your Organization")
set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev, libopenblas-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "eigen3-devel, openblas-devel")
include(CPack)

# Print configuration summary
message(STATUS "LwTT Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${LWTT_BUILD_TESTS}")
message(STATUS "  Build Python bindings: ${LWTT_BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build documentation: ${LWTT_BUILD_DOCS}")
message(STATUS "  Enable code coverage: ${LWTT_ENABLE_COVERAGE}")
message(STATUS "  Enable SIMD: ${LWTT_ENABLE_SIMD}")
message(STATUS "  Enable OpenMP: ${LWTT_ENABLE_OPENMP}")
message(STATUS "  Enable profiling: ${LWTT_ENABLE_PROFILING}")
message(STATUS "  Has LibTorch: ${LWTT_HAS_TORCH}")
message(STATUS "  Use Eigen: ${LWTT_USE_EIGEN}")
message(STATUS "  Build shared: ${LWTT_BUILD_SHARED}")
message(STATUS "  Build static: ${LWTT_BUILD_STATIC}")
