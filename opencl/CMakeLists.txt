cmake_minimum_required(VERSION 3.10)
project(DSP_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- 1. Robust OpenCL Discovery ---
# Try standard CMake find_package
find_package(OpenCL)

# If not found, try PkgConfig (without REQUIRED to avoid crashing)
if(NOT OpenCL_FOUND)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(OCL_PKG OpenCL)
        if(OCL_PKG_FOUND)
            set(OpenCL_INCLUDE_DIRS ${OCL_PKG_INCLUDE_DIRS})
            set(OpenCL_LIBRARIES ${OCL_PKG_LIBRARIES})
            set(OpenCL_FOUND TRUE)
        endif()
    endif()
endif()

# Final Fallback: Manual search in ARM/Pi common paths
if(NOT OpenCL_FOUND)
    message(STATUS "[OpenCL] Standard discovery failed. Searching manually...")
    find_path(OpenCL_INCLUDE_DIR CL/cl.h PATHS /usr/include /usr/local/include)
    find_library(OpenCL_LIBRARY NAMES OpenCL PATHS /usr/lib/aarch64-linux-gnu /usr/lib/arm-linux-gnueabihf /usr/lib)
    
    if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
        set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
        set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
        set(OpenCL_FOUND TRUE)
        message(STATUS "[OpenCL] Found manually at ${OpenCL_LIBRARY}")
    endif()
endif()

# Mandatory check
if(NOT OpenCL_FOUND)
    message(FATAL_ERROR "Could NOT find OpenCL. Please run ./setup_pi5.sh first.")
endif()

# --- 2. Vulkan & GLFW ---
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# Add executable
add_executable(inference_app vulkan_opencl_app.cpp)

# Include directories
target_include_directories(inference_app PRIVATE 
    ${OpenCL_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(inference_app PRIVATE 
    ${OpenCL_LIBRARIES}
    Vulkan::Vulkan
    glfw
)
