cmake_minimum_required(VERSION 3.10)
project(moptimizer)
enable_language(CUDA)

option(BUILD_TESTS "Build test programs" OFF)

find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen TBB::tbb)

link_libraries(${PROJECT_NAME})

if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  enable_testing()
  include(GoogleTest)

  file (GLOB tests_srcs ${CMAKE_SOURCE_DIR}/tst/*.cc)
  add_executable(
    ${PROJECT_NAME}_tests
    ${tests_srcs}
    )
  target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME} gtest_main GTest::gtest GTest::gmock)


  gtest_discover_tests(${PROJECT_NAME}_tests)
endif()