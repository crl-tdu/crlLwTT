# Examples CMakeLists.txt

# Use the target defined in the main project
if(TARGET LwTT::LwTT)
    set(LwTT_LIBRARIES LwTT::LwTT)
else()
    # Fallback if the target is not available
    message(WARNING "LwTT::LwTT target not found, examples may not build correctly")
endif()

# Basic usage examples
add_subdirectory(basic_usage)

# Advanced usage examples (if available)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/advanced_usage/CMakeLists.txt")
    add_subdirectory(advanced_usage)
endif()

# STORM integration examples (if available)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/storm_integration/CMakeLists.txt")
    add_subdirectory(storm_integration)
endif()
