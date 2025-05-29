# Advanced Usage Examples CMakeLists.txt

# Use the target defined in the main project
if(TARGET LwTT::LwTT)
    set(LwTT_LIBRARIES LwTT::LwTT)
else()
    # Fallback if the target is not available
    message(WARNING "LwTT::LwTT target not found, examples may not build correctly")
    return()
endif()

# STA Integration Example
add_executable(sta_integration_example 
    sta_integration_example.cpp
)

target_link_libraries(sta_integration_example 
    PRIVATE ${LwTT_LIBRARIES}
)

target_include_directories(sta_integration_example 
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# Set C++17 standard
set_target_properties(sta_integration_example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Enable compiler optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(sta_integration_example PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

# Platform-specific settings
if(UNIX AND NOT APPLE)
    target_link_libraries(sta_integration_example PRIVATE pthread)
endif()

if(APPLE)
    target_link_libraries(sta_integration_example PRIVATE pthread)
endif()

if(WIN32)
    # Windows-specific settings if needed
endif()

# Install example if requested
if(LWTT_INSTALL_EXAMPLES)
    install(TARGETS sta_integration_example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
    )
endif()

# Add to examples target group
set_target_properties(sta_integration_example PROPERTIES FOLDER "Examples/Advanced")

# Copy example to build directory for easy access during development
if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set_target_properties(sta_integration_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples"
    )
endif()
