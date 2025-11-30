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
    src/kdtree/kdtree.cpp
)

set(KNN_SOURCES
    src/knn/knn_basic.cpp
    src/knn/knn_kdtree.cpp
)

set(UTILS_SOURCES
    src/utils/point.cpp
    src/utils/distance_metrics.cpp
    src/utils/dataset_loader.cpp
    src/utils/metrics.cpp
)

# Libraries
add_library(kdtree ${KDTREE_SOURCES})
add_library(knn ${KNN_SOURCES})
add_library(utils ${UTILS_SOURCES})

# Tests
add_executable(test_knn_basic tests/test_knn_basic.cpp)
target_link_libraries(test_knn_basic knn utils)

add_executable(test_knn_kdtree tests/test_knn_kdtree.cpp)
target_link_libraries(test_knn_kdtree knn kdtree utils)

# Examples (commented out - not yet implemented)
# add_executable(example_bentley_1975 examples/example_bentley_1975.cpp)
# target_link_libraries(example_bentley_1975 kdtree utils)

# Benchmarks (commented out - not yet implemented)
# add_executable(benchmark_basic_vs_kdtree benchmarks/benchmark_basic_vs_kdtree.cpp)
# target_link_libraries(benchmark_basic_vs_kdtree kdtree utils)

# Visualization tools (commented out - not yet implemented)
# add_executable(visualize_kdtree visualization/visualize_kdtree.cpp)
# target_link_libraries(visualize_kdtree kdtree utils)
