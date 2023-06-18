cmake_minimum_required(VERSION 3.24)
project(SpmX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall --verbose")
find_package(OpenMP)

if(OpenMP_FOUND)
    message(STATUS "OpenMP found!")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
    add_definitions(-DOPENMP_ENABLED)
else()
    message(STATUS "OpenMP not found.")
endif()

cmake_host_system_information(RESULT SSE2_SUPPORT QUERY HAS_SSE2)
cmake_host_system_information(RESULT CPU_LOGICAL_CORE_NUM QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT CPU_PHYSICAL_CORE_NUM QUERY NUMBER_OF_PHYSICAL_CORES)
message(STATUS "${CPU_LOGICAL_CORE_NUM} CPU logical cores detected.")
message(STATUS "${CPU_PHYSICAL_CORE_NUM} CPU physical cores detected.")
add_definitions(-DNUMBER_OF_CPU_LOGICAL_CORES=${CPU_LOGICAL_CORE_NUM})
add_definitions(-DNUMBER_OF_CPU_PHYSICAL_CORES=${CPU_PHYSICAL_CORE_NUM})
# the following option is for debugging
# add_definitions(-DMEMORY_TRACING)

# Copied from the User Guide of Google Test: https://google.github.io/googletest/quickstart-cmake.html
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#)
# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)


if(SSE2_SUPPORT)
    message(STATUS "SSE2 is supported by host system. SIMD enabled.")
    add_definitions(-DSIMD_SSE2_ENABLED)
else()
    message(STATUS "SSE2 is not supported by host system. SIMD disabled.")
endif()

set(SPMX_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
set(SPMX_SRC_DIRS "${CMAKE_CURRENT_LIST_DIR}/src")
set(SPMX_TESTS_DIRS "${CMAKE_CURRENT_LIST_DIR}/tests")

file(GLOB SPMX_HEADERS ${SPMX_INCLUDE_DIRS}/*)
file(GLOB SPMX_CHOLESKY ${SPMX_INCLUDE_DIRS}/sparse-cholesky/*)
file(GLOB SPMX_SRC ${SPMX_SRC_DIRS}/*)
include_directories(${SPMX_INCLUDE_DIRS})

# if(benchmark_FOUND)
#    target_link_libraries(benchmark GoogleBenchmark)
//    add_executable(test-arith tests/test-arith.cc ${SPMX_HEADERS})
//    add_executable(test-solver tests/test-solver.cc ${SPMX_HEADERS} ${SPMX_CHOLESKY})
//    add_executable(bench-arith tests/benchmark-arith.cc ${SPMX_HEADERS} ${SPMX_CHOLESKY})
//    add_executable(bench-solver tests/benchmark-solver.cc ${SPMX_HEADERS} ${SPMX_CHOLESKY})
# endif()
