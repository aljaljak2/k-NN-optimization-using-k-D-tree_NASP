cmake_minimum_required(VERSION 3.10)
project(knn_benchmarks)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Benchmark source files
set(BENCHMARK_SOURCES
    src/benchmark_utils.cpp
    src/benchmark_runner.cpp
    src/knn_nanoflann.cpp
)

# Link parent project libraries
set(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(KDTREE_SOURCES
    ${PARENT_DIR}/src/kdtree/kdnode.cpp
    ${PARENT_DIR}/src/kdtree/kdtree.cpp
)

set(KNN_SOURCES
    ${PARENT_DIR}/src/knn/knn_basic.cpp
    ${PARENT_DIR}/src/knn/knn_kdtree.cpp
)

set(UTILS_SOURCES
    ${PARENT_DIR}/src/utils/point.cpp
    ${PARENT_DIR}/src/utils/distance_metrics.cpp
    ${PARENT_DIR}/src/utils/dataset_loader.cpp
    ${PARENT_DIR}/src/utils/metrics.cpp
)

# Create libraries
add_library(kdtree_lib ${KDTREE_SOURCES})
add_library(knn_lib ${KNN_SOURCES})
add_library(utils_lib ${UTILS_SOURCES})
add_library(benchmark_lib ${BENCHMARK_SOURCES})

# Main benchmark executable
add_executable(knn_benchmark benchmark_main.cpp)
target_link_libraries(knn_benchmark
    benchmark_lib
    knn_lib
    kdtree_lib
    utils_lib
)

# Create results directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)

message(STATUS "Benchmark executable: knn_benchmark")
message(STATUS "Output directory: benchmarks/results/")
