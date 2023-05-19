cmake_minimum_required(VERSION 3.24)
project(SpmX)

set(CMAKE_CXX_STANDARD 17)

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
find_package(GoogleBenchmark)

if(GoogleBenchmark_FOUND)
    message(STATUS "GoogleBenchmark found! Enable benchmarking.")
else()
    message(WARNING "GoogleBenchmark not found. Disable benchmarking.")
endif()

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
file(GLOB SPMX_SRC ${SPMX_SRC_DIRS}/*)
include_directories(${SPMX_INCLUDE_DIRS})

# if(GoogleBenchmark_FOUND)
#    target_link_libraries(benchmark GoogleBenchmark)
    add_executable(benchmark tests/spmx-benchmark.cc ${SPMX_HEADERS})
# endif()