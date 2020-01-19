cmake_minimum_required(VERSION 3.8.0)
project(Voxelyze3 LANGUAGES CXX CUDA)

add_executable(Voxelyze3 src/Voxelyze3.cu)
add_executable(vx_run src/vx_run.cpp)
