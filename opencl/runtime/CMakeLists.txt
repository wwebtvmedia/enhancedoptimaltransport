cmake_minimum_required(VERSION 3.16)
project(onnx_opencl_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Search for OpenCL via the standard module first, with several Pi-friendly fallbacks.
find_package(OpenCL QUIET)

if(NOT OpenCL_FOUND)
  find_path(OpenCL_INCLUDE_DIRS CL/cl.h
            HINTS /usr/include /usr/local/include /opt/include)
  find_library(OpenCL_LIBRARIES NAMES OpenCL
               HINTS /usr/lib /usr/lib/aarch64-linux-gnu
                     /usr/lib/arm-linux-gnueabihf /usr/local/lib)
endif()

if(NOT OpenCL_LIBRARIES OR NOT OpenCL_INCLUDE_DIRS)
  message(FATAL_ERROR "OpenCL not found — install ocl-icd-opencl-dev and an ICD")
endif()

add_executable(onnx_opencl_runner
  src/main.cpp
  src/runtime.cpp
)
target_include_directories(onnx_opencl_runner PRIVATE
  src
  third_party
  ${OpenCL_INCLUDE_DIRS}
)
target_link_libraries(onnx_opencl_runner PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(onnx_opencl_runner PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_compile_options(onnx_opencl_runner PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-sign-compare)
