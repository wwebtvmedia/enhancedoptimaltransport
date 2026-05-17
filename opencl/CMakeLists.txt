cmake_minimum_required(VERSION 3.10)
project(DSP_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCL
find_package(OpenCL)

# Fallback to PkgConfig if find_package fails (common on some ARM distros)
if(NOT OpenCL_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OpenCL REQUIRED OpenCL)
    set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES})
endif()

# Explicitly search in common Pi paths if still not found
if(NOT OpenCL_INCLUDE_DIRS)
    find_path(OpenCL_INCLUDE_DIR CL/cl.h PATHS /usr/include /usr/local/include)
    find_library(OpenCL_LIBRARY NAMES OpenCL PATHS /usr/lib/aarch64-linux-gnu /usr/lib/arm-linux-gnueabihf)
    set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
endif()

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find GLFW (using pkg-config or standard search)
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
