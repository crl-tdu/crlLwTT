# Unit tests CMakeLists
file(GLOB_RECURSE UNIT_TEST_SOURCES "*.cpp")

# Create test executable
add_executable(unit_tests ${UNIT_TEST_SOURCES})

# Link with LwTT library and GoogleTest
target_link_libraries(unit_tests 
    PRIVATE 
    LwTT::LwTT
    gtest
    gtest_main
)

# Add include directories
target_include_directories(unit_tests 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)
