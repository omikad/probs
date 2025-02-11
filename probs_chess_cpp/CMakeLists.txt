cmake_minimum_required(VERSION 3.17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # produce compile_commands.json

execute_process (COMMAND bash -c "PYTHONPATH=../py_binding/ python3 ../py_binding/gen_py_bindings.py ../src/python/generated_bindings.cpp")

project(ProbsChess)

# set(CMAKE_BUILD_TYPE RelWithDebInfo)

# Conda env
list(APPEND CMAKE_PREFIX_PATH "/home/excellent/miniconda3/envs/ds")
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/yaml-cpp")

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

find_package(yaml-cpp REQUIRED)

find_package(pybind11 REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ProbsChess PRIVATE Threads::Threads yaml-cpp "${TORCH_LIBRARIES}" ${PYTHON_LIBRARIES})

# Python bindings build
find_package(pybind11 CONFIG REQUIRED)

ADD_LIBRARY(probs_chess SHARED ${SOURCES})
target_compile_definitions(probs_chess PRIVATE NO_PEXT)
target_link_libraries(probs_chess PRIVATE yaml-cpp "${TORCH_LIBRARIES}" ${PYTHON_LIBRARIES})
target_include_directories(probs_chess PRIVATE ${PYTHON_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)