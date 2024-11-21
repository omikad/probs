cmake_minimum_required(VERSION 3.17)

# Project Name
project(ProbsChess)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CUDACXX "/usr/local/cuda-12.1/bin/nvcc")

set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)

set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-12.1/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")
set(CUDA_HOME "/usr/local/cuda-12.1")
set(LD_LIBRARY_PATH "/usr/local/cuda-12.1/lib64")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")

set(Torch_DIR "/home/excellent/Downloads/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the executable
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(ProbsChess ${SOURCES})

target_compile_definitions(ProbsChess PRIVATE NO_PEXT)

target_include_directories(ProbsChess PRIVATE ${CMAKE_SOURCE_DIR}/src)

set_property(TARGET ProbsChess PROPERTY CXX_STANDARD 17)

# Find and link yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(ProbsChess PRIVATE yaml-cpp "${TORCH_LIBRARIES}")
