project(vecxx)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA)

cuda_add_executable(gpu_vecxx gpu_vecxx.cu)
cuda_add_executable(cpu_gpu_vecxx cpu_gpu_vecxx.cu)
