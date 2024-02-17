cmake_minimum_required(VERSION 3.10)
project(moptimizer)
enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
add_executable(simple src/simple.cc)
target_include_directories(simple PUBLIC include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(simple Eigen3::Eigen TBB::tbb)