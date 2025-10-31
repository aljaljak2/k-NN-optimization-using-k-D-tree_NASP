cmake_minimum_required(VERSION 3.10)
project(knn_kdtree_optimization)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files for libraries
set(KDTREE_SOURCES
    src/kdtree/kdnode.cpp
    src/kdtree/kdtree_original.cpp
)

set(UTILS_SOURCES
    src/utils/point.cpp
    src/utils/distance_metrics.cpp
)

# Libraries
add_library(kdtree ${KDTREE_SOURCES})
add_library(utils ${UTILS_SOURCES})

# Examples
add_executable(example_bentley_1975 examples/example_bentley_1975.cpp)
target_link_libraries(example_bentley_1975 kdtree utils)

# Benchmarks
add_executable(benchmark_basic_vs_kdtree benchmarks/benchmark_basic_vs_kdtree.cpp)
target_link_libraries(benchmark_basic_vs_kdtree kdtree utils)

# Tests (if using a testing framework like Google Test)
# enable_testing()
# add_subdirectory(tests)

# Visualization tools
add_executable(visualize_kdtree visualization/visualize_kdtree.cpp)
target_link_libraries(visualize_kdtree kdtree utils)

# Original kdtree.cpp as standalone executable
add_executable(kdtree_original src/kdtree/kdtree_original.cpp)
