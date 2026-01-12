cmake_minimum_required(VERSION 3.16)

project(cxl_p2p_shm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cxl_p2p_shm_cpu p2p_cxl_shm_cpu.cpp)
target_compile_options(cxl_p2p_shm_cpu PRIVATE -O3 -Wall -Wextra -pthread -march=native -mclflushopt)
target_link_libraries(cxl_p2p_shm_cpu PRIVATE pthread)

add_executable(cxl_p2p_shm_gpu p2p_cxl_shm_gpu.cpp)
target_compile_options(cxl_p2p_shm_gpu PRIVATE -O3 -Wall -Wextra -pthread -march=native -mclflushopt)
target_link_libraries(cxl_p2p_shm_gpu PRIVATE pthread CUDA::cudart)
