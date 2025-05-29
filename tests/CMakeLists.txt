# LwTT Tests CMakeLists.txt

# Include Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)

# Unit tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests")
    add_subdirectory(unit_tests)
endif()

# Integration tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/integration_tests")
    add_subdirectory(integration_tests)
endif()

# Examples (moved from examples directory)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

# Benchmarks (moved from benchmarks directory)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()
