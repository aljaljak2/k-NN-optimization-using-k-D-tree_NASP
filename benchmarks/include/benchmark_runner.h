#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "benchmark_utils.h"
#include "../../include/knn/knn_basic.h"
#include "../../include/knn/knn_kdtree.h"
#include "knn_nanoflann.h"
#include <vector>
#include <string>
#include <map>

/**
 * Dataset configuration for real datasets
 */
struct DatasetConfig {
    std::string filepath;
    int labelColumn;  // -1 = last, 0 = first, -2 = second to last, etc.

    DatasetConfig(const std::string& path, int labelCol = -1)
        : filepath(path), labelColumn(labelCol) {}
};

/**
 * Benchmark runner for comparing KNN implementations
 * Tests: KNNBasic, KNNKDTree, and KNNNanoflann
 */
class BenchmarkRunner {
private:
    std::vector<BenchmarkResult> results;
    std::map<std::string, double> basicQueryTimes; // For speedup calculation
    int totalTests;
    int currentTest;

    // Helper to calculate accuracy (legacy)
    double calculateAccuracy(const std::vector<Point>& train,
                            const std::vector<Point>& test,
                            const std::string& algorithm,
                            int k, int dimensions);

    // Helper to calculate all classification metrics
    ClassificationMetrics calculateClassificationMetrics(const std::vector<Point>& train,
                                                          const std::vector<Point>& test,
                                                          const std::string& algorithm,
                                                          int k, int dimensions,
                                                          long long& total_distance_calcs);

    // Single algorithm benchmark
    BenchmarkResult benchmarkAlgorithm(const std::string& algorithm,
                                        const std::vector<Point>& train,
                                        const std::vector<Point>& queries,
                                        const std::string& dataset_name,
                                        int k, int dimensions);

    // Progress reporting
    void reportProgress(const std::string& message);

public:
    BenchmarkRunner();

    // Test scenarios
    void runCurseOfDimensionality();
    void runScalability();
    void runKParameterImpact();
    void runRealDatasets(const std::vector<DatasetConfig>& datasets);

    // Execute all benchmarks
    void runAllBenchmarks(const std::vector<DatasetConfig>& real_datasets);

    // Save results to JSON
    void saveResults(const std::string& filepath, double total_duration_sec);

    // Save comprehensive CSV results
    void saveCSVResults(const std::string& filepath, double total_duration_sec);

    // Get results
    const std::vector<BenchmarkResult>& getResults() const { return results; }
};

#endif // BENCHMARK_RUNNER_H
