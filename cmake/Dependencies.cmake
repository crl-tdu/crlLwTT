# Dependencies.cmake
# External dependencies for LwTT

include(FetchContent)

# Dependencies.cmake
# External dependencies for LwTT

include(FetchContent)

# LibTorch handling - use common detection module
include(FindLibTorch)

# Build options  
option(LWTT_DISABLE_TORCH "Disable LibTorch support" OFF)

# LibTorch detection with guidance
find_libtorch_for_project(LWTT)

if(LIBTORCH_FOUND)
    message(STATUS "Building LwTT with LibTorch support")
    set(LWTT_HAS_TORCH TRUE)
    set(Torch_FOUND TRUE)
    # Set compatibility variables
    set(TORCH_INCLUDE_DIRS ${LIBTORCH_INCLUDE_DIRS})
    set(TORCH_LIBRARIES ${LIBTORCH_LIBRARIES})
    set(TORCH_CXX_FLAGS ${LIBTORCH_CXX_FLAGS})
else()
    message(STATUS "Building LwTT without LibTorch support")
    set(LWTT_HAS_TORCH FALSE)
    set(Torch_FOUND FALSE)
endif()

# Find Eigen (optional)
if(LWTT_USE_EIGEN)
    find_package(Eigen3 3.3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
        add_definitions(-DLWTT_USE_EIGEN)
    else()
        message(STATUS "Eigen3 not found, fetching from source")
        FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(eigen)
        add_definitions(-DLWTT_USE_EIGEN)
    endif()
endif()

# OpenMP
if(LWTT_ENABLE_OPENMP)
    # macOS Homebrew OpenMP handling
    if(APPLE)
        find_program(BREW_COMMAND brew)
        if(BREW_COMMAND)
            execute_process(
                COMMAND ${BREW_COMMAND} --prefix libomp
                OUTPUT_VARIABLE LIBOMP_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}")
                set(OpenMP_ROOT "${LIBOMP_PREFIX}")
                message(STATUS "Setting OpenMP_ROOT to: ${OpenMP_ROOT}")
            endif()
        endif()
    endif()
    
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    else()
        message(STATUS "OpenMP not found")
        if(APPLE)
            message(STATUS "On macOS, you may need to install libomp:")
            message(STATUS "  brew install libomp")
        endif()
        set(LWTT_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP" FORCE)
    endif()
endif()

# GoogleTest (for tests)
if(LWTT_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.13.0
        GIT_SHALLOW TRUE
    )
    # We don't want to install gtest with our project
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Google Benchmark (for benchmarks)
if(LWTT_BUILD_BENCHMARKS)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.0
        GIT_SHALLOW TRUE
    )
    # We don't want to build benchmark tests
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# pybind11 (for Python bindings) - 現在無効化
# if(LWTT_BUILD_PYTHON_BINDINGS)
#     find_package(Python COMPONENTS Interpreter Development QUIET)
#     if(Python_FOUND)
#         FetchContent_Declare(
#             pybind11
#             GIT_REPOSITORY https://github.com/pybind/pybind11.git
#             GIT_TAG v2.11.1
#             GIT_SHALLOW TRUE
#         )
#         FetchContent_MakeAvailable(pybind11)
#     else()
#         message(WARNING "Python not found, Python bindings will be disabled")
#         set(LWTT_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings" FORCE)
#     endif()
# endif()

# Doxygen (for documentation) - 現在無効化
# if(LWTT_BUILD_DOCS)
#     find_package(Doxygen QUIET)
#     if(NOT DOXYGEN_FOUND)
#         message(STATUS "Doxygen not found, documentation will not be built")
#         set(LWTT_BUILD_DOCS OFF CACHE BOOL "Build documentation" FORCE)
#     endif()
# endif()

