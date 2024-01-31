CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(min_nl_opts)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH /home/marcus/workspace/practical\ software/coding/matplotlib-cpp/build/install)

find_package(Eigen3 REQUIRED)
find_package(matplotlib_cpp REQUIRED )

link_libraries(Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)
add_definitions(-DUSE_MATPLOTLIB)


add_executable(simple src/simple.cc)
add_executable(gauss src/gauss.cc)
add_executable(point2 src/point2.cc)
