# Benchmarks CMakeLists
file(GLOB_RECURSE BENCHMARK_SOURCES "*.cpp")

# Include Google Benchmark
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.0
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Create benchmark executable
add_executable(benchmarks ${BENCHMARK_SOURCES})

# Link with LwTT library and Google Benchmark
target_link_libraries(benchmarks 
    PRIVATE 
    LwTT::LwTT
    benchmark::benchmark
)

# Add include directories
target_include_directories(benchmarks 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add custom target to run benchmarks
add_custom_target(run_benchmarks
    COMMAND benchmarks
    DEPENDS benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
)
