CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
project(min_nl_opts)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_PREFIX_PATH /home/marcus/workspace/practical\ software/coding/matplotlib-cpp/build/install)

find_package(Eigen3 REQUIRED)
find_package(matplotlib_cpp REQUIRED )
add_executable(main src/main.cc)
add_executable(gauss src/gauss.cc)

target_link_libraries(main Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)
target_link_libraries(gauss Eigen3::Eigen matplotlib_cpp::matplotlib_cpp)