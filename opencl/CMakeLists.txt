cmake_minimum_required(VERSION 3.10)
project(DSP_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCL
find_package(OpenCL REQUIRED)

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
