# FindLibTorch.cmake - Common LibTorch detection module for crl* projects
# 
# This module finds LibTorch installation and provides installation guidance
# if LibTorch is not found.
#
# Variables set by this module:
#   LIBTORCH_FOUND          - True if LibTorch is found
#   LIBTORCH_INCLUDE_DIRS   - Include directories for LibTorch
#   LIBTORCH_LIBRARIES      - LibTorch libraries
#   LIBTORCH_CXX_FLAGS      - C++ flags required for LibTorch
#
# Cache variables:
#   LIBTORCH_ROOT_DIR       - Root directory of LibTorch installation

cmake_minimum_required(VERSION 3.16)

# Standard LibTorch installation path
set(LIBTORCH_STANDARD_PATH "$ENV{HOME}/local/libtorch")

# Function to display LibTorch installation instructions
function(display_libtorch_installation_instructions)
    message(STATUS "")
    message(STATUS "========================================================================")
    message(STATUS "LibTorch NOT FOUND")
    message(STATUS "========================================================================")
    message(STATUS "")
    message(STATUS "LibTorch is required for this project but was not found in:")
    message(STATUS "  Expected location: ${LIBTORCH_STANDARD_PATH}")
    message(STATUS "")
    message(STATUS "Please install LibTorch using one of the following methods:")
    message(STATUS "")
    message(STATUS "Method 1: Download pre-built LibTorch (Recommended)")
    message(STATUS "  # For macOS (Apple Silicon)")
    message(STATUS "  cd ~/local")
    message(STATUS "  wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.2.zip")
    message(STATUS "  unzip libtorch-macos-arm64-2.1.2.zip")
    message(STATUS "  rm libtorch-macos-arm64-2.1.2.zip")
    message(STATUS "")
    message(STATUS "  # For macOS (Intel)")
    message(STATUS "  cd ~/local")
    message(STATUS "  wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.1.2.zip")
    message(STATUS "  unzip libtorch-macos-x86_64-2.1.2.zip")
    message(STATUS "  rm libtorch-macos-x86_64-2.1.2.zip")
    message(STATUS "")
    message(STATUS "  # For Linux")
    message(STATUS "  cd ~/local")
    message(STATUS "  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip")
    message(STATUS "  unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu.zip")
    message(STATUS "  rm libtorch-cxx11-abi-shared-with-deps-2.1.2+cpu.zip")
    message(STATUS "")
    message(STATUS "Method 2: Use Homebrew (macOS only)")
    message(STATUS "  brew install pytorch")
    message(STATUS "  ln -s $(brew --prefix)/opt/pytorch/lib ~/local/libtorch")
    message(STATUS "")
    message(STATUS "Method 3: Set environment variables (if installed elsewhere)")
    message(STATUS "  export CMAKE_PREFIX_PATH=/path/to/your/libtorch:\$CMAKE_PREFIX_PATH")
    message(STATUS "")
    message(STATUS "For detailed installation instructions, see:")
    message(STATUS "  docs/LIBTORCH_INSTALLATION_GUIDE.md")
    message(STATUS "")
    message(STATUS "After installation, please re-run cmake.")
    message(STATUS "")
    message(STATUS "To build without LibTorch support, use:")
    message(STATUS "  cmake -D<PROJECT>_DISABLE_TORCH=ON ..")
    message(STATUS "  (Replace <PROJECT> with CRLGRU, CRLSOM, or LWTT)")
    message(STATUS "")
    message(STATUS "========================================================================")
    message(STATUS "")
endfunction()

# Function to check LibTorch installation
function(check_libtorch_installation)
    # Check if LibTorch directory exists
    if(EXISTS "${LIBTORCH_STANDARD_PATH}")
        message(STATUS "Found LibTorch directory: ${LIBTORCH_STANDARD_PATH}")
        
        # Check essential files
        set(LIBTORCH_ESSENTIAL_FILES
            "${LIBTORCH_STANDARD_PATH}/lib"
            "${LIBTORCH_STANDARD_PATH}/include"
            "${LIBTORCH_STANDARD_PATH}/share/cmake/Torch"
        )
        
        set(LIBTORCH_COMPLETE TRUE)
        foreach(file ${LIBTORCH_ESSENTIAL_FILES})
            if(NOT EXISTS "${file}")
                message(STATUS "Missing LibTorch component: ${file}")
                set(LIBTORCH_COMPLETE FALSE)
            endif()
        endforeach()
        
        if(NOT LIBTORCH_COMPLETE)
            message(WARNING "LibTorch installation appears incomplete at ${LIBTORCH_STANDARD_PATH}")
            message(STATUS "Please reinstall LibTorch or check installation.")
        endif()
        
        # Add to CMAKE_PREFIX_PATH
        list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_STANDARD_PATH}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
        
    else()
        message(STATUS "LibTorch not found at standard location: ${LIBTORCH_STANDARD_PATH}")
        
        # Check if CMAKE_PREFIX_PATH contains libtorch
        string(FIND "${CMAKE_PREFIX_PATH}" "libtorch" LIBTORCH_IN_PREFIX)
        if(LIBTORCH_IN_PREFIX GREATER -1)
            message(STATUS "LibTorch may be available via CMAKE_PREFIX_PATH")
        endif()
    endif()
endfunction()

# Main detection logic
function(find_libtorch_with_guidance PROJECT_DISABLE_OPTION)
    # Check installation first
    check_libtorch_installation()
    
    # Try to find LibTorch
    find_package(Torch QUIET)
    
    if(Torch_FOUND)
        message(STATUS "✓ LibTorch found successfully")
        message(STATUS "  Version: ${TORCH_VERSION}")
        message(STATUS "  Libraries: ${TORCH_LIBRARIES}")
        message(STATUS "  Include dirs: ${TORCH_INCLUDE_DIRS}")
        
        # Set common variables
        set(LIBTORCH_FOUND TRUE PARENT_SCOPE)
        set(LIBTORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} PARENT_SCOPE)
        set(LIBTORCH_LIBRARIES ${TORCH_LIBRARIES} PARENT_SCOPE)
        set(LIBTORCH_CXX_FLAGS ${TORCH_CXX_FLAGS} PARENT_SCOPE)
        
    else()
        message(STATUS "✗ LibTorch not found")
        
        # Check if user explicitly disabled LibTorch
        if(${PROJECT_DISABLE_OPTION})
            message(STATUS "LibTorch support is explicitly disabled")
            set(LIBTORCH_FOUND FALSE PARENT_SCOPE)
        else()
            # Display installation instructions
            display_libtorch_installation_instructions()
            
            # Ask user what to do
            message(STATUS "Build options:")
            message(STATUS "  1. Install LibTorch and re-run cmake")
            message(STATUS "  2. Continue without LibTorch (some features will be disabled)")
            message(STATUS "  3. Cancel build")
            message(STATUS "")
            
            # For automated builds, we can continue without LibTorch
            option(CONTINUE_WITHOUT_LIBTORCH "Continue build without LibTorch" OFF)
            
            if(CONTINUE_WITHOUT_LIBTORCH)
                message(STATUS "Continuing build without LibTorch support...")
                set(LIBTORCH_FOUND FALSE PARENT_SCOPE)
            else()
                message(FATAL_ERROR "LibTorch is required. Please install LibTorch or set CONTINUE_WITHOUT_LIBTORCH=ON")
            endif()
        endif()
    endif()
endfunction()

# Export the main function
macro(find_libtorch_for_project PROJECT_PREFIX)
    find_libtorch_with_guidance(${PROJECT_PREFIX}_DISABLE_TORCH)
endmacro()
