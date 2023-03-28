cmake_minimum_required(VERSION 3.24)
project(SpmX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

set(SPMX_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
set(SPMX_SRC_DIRS "${CMAKE_CURRENT_LIST_DIR}/src")
set(SPMX_TESTS_DIRS "${CMAKE_CURRENT_LIST_DIR}/tests")

file(GLOB SPMX_HEADERS ${SPMX_INCLUDE_DIRS}/*)
file(GLOB SPMX_SRC ${SPMX_SRC_DIRS}/*)
add_library(SpmX STATIC ${SPMX_SRC} ${SPMX_HEADERS})
add_executable(testArithmetic ${SPMX_TESTS_DIRS}/testArithmetic.cpp)
target_include_directories(SpmX PUBLIC ${SPMX_INCLUDE_DIRS})
target_link_libraries(testArithmetic SpmX)