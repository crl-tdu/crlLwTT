# Integration tests CMakeLists
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "*.cpp")

# Only create test executable if sources exist
if(INTEGRATION_TEST_SOURCES)
    # Create test executable
    add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})

    # Link with LwTT library and GoogleTest
    target_link_libraries(integration_tests 
        PRIVATE 
        LwTT::LwTT
        gtest
        gtest_main
    )

    # Add include directories
    target_include_directories(integration_tests 
        PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    # Add tests to CTest
    include(GoogleTest)
    gtest_discover_tests(integration_tests)
else()
    message(STATUS "No integration test sources found, skipping integration_tests target")
endif()
