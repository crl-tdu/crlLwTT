# Basic Usage Examples CMakeLists.txt

# Simple Transformer Example
add_executable(simple_transformer simple_transformer.cpp)

# Link with LwTT library
target_link_libraries(simple_transformer ${LwTT_LIBRARIES})

# Set C++ standard
set_target_properties(simple_transformer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(simple_transformer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set RPATH for finding LibTorch libraries at runtime
if(APPLE)
    # macOS specific settings
    set_target_properties(simple_transformer PROPERTIES
        INSTALL_RPATH "@loader_path;@loader_path/../../;${LIBTORCH_LIB_DIR};$ENV{HOME}/local/lib"
        BUILD_RPATH "${CMAKE_BINARY_DIR};${LIBTORCH_LIB_DIR};$ENV{HOME}/local/lib"
    )
    target_link_libraries(simple_transformer "-framework Accelerate")
elseif(UNIX AND NOT APPLE)
    # Linux specific settings
    set_target_properties(simple_transformer PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/../../:${LIBTORCH_LIB_DIR}:$ENV{HOME}/local/lib"
        BUILD_RPATH "${CMAKE_BINARY_DIR}:${LIBTORCH_LIB_DIR}:$ENV{HOME}/local/lib"
    )
    target_link_libraries(simple_transformer pthread)
elseif(WIN32)
    # Windows specific settings
    target_compile_definitions(simple_transformer PRIVATE WIN32_LEAN_AND_MEAN)
endif()

# Install target
install(TARGETS simple_transformer
    RUNTIME DESTINATION bin/examples
)
