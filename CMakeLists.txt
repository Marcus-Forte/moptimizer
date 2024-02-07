CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(min_nl_opts)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
link_libraries(Eigen3::Eigen)

find_package(matplotlib_cpp)
if(matplotlib_cpp_FOUND)
    link_libraries(matplotlib_cpp::matplotlib_cpp)
    add_definitions(-DUSE_MATPLOTLIB)
endif()

add_executable(simple src/simple.cc)
add_executable(gauss src/gauss.cc)
add_executable(point2 src/point2.cc)