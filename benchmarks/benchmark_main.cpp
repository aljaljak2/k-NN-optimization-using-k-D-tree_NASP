#include "include/benchmark_runner.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// Define real datasets with their label column configurations
// labelColumn: -1 = last column, 0 = first column, -2 = second to last
// Paths are relative to where executable is run from (Implementacija/)
const std::vector<DatasetConfig> REAL_DATASETS = {
    DatasetConfig("../../datasets/letter-recognition.csv", 0),    // Label in first column
    DatasetConfig("../../datasets/WineQT.csv", -2),               // Label in second to last (quality)
    DatasetConfig("../../datasets/covtype.csv", -1)               // Label in last column
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "   KNN Benchmark Suite" << std::endl;
    std::cout << "   Comparing: KNNBasic, KNNKDTree, KNNNanoflann" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create results directory if it doesn't exist
    std::filesystem::create_directories("benchmarks/results");

    // Create benchmark runner
    BenchmarkRunner runner;

    // Start timer
    Timer globalTimer;
    globalTimer.start();

    // Run all benchmarks
    runner.runAllBenchmarks(REAL_DATASETS);

    // Calculate total duration
    double total_duration = globalTimer.elapsed_sec();

    // Save results in both JSON and CSV formats
    std::string json_output = "benchmarks/results/benchmark_results.json";
    std::string csv_output = "benchmarks/results/benchmark_comprehensive.csv";

    runner.saveResults(json_output, total_duration);
    runner.saveCSVResults(csv_output, total_duration);

    std::cout << "\n========================================" << std::endl;
    std::cout << "   Benchmark suite completed!" << std::endl;
    std::cout << "   Results saved to:" << std::endl;
    std::cout << "   - " << json_output << std::endl;
    std::cout << "   - " << csv_output << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
