#include "include/benchmark_runner.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>

#include "../include/utils/dataset_loader.h"
#include "../include/utils/distance_metrics.h"

const std::vector<DatasetConfig> REAL_DATASETS = {
    DatasetConfig("../../datasets/letter-recognition.csv", 0),
    DatasetConfig("../../datasets/WineQT.csv", -2),
    DatasetConfig("../../datasets/covtype.csv", -1)};

void runMemoryAnalysis()
{
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "   [EXTENSION] PRECISE MEMORY FOOTPRINT ANALYSIS" << std::endl;
    std::cout << "===========================================================" << std::endl;

    size_t size_double = sizeof(double);
    size_t size_ptr = sizeof(void *);
    size_t size_int = sizeof(int);
    size_t size_point_struct = sizeof(Point);
    size_t size_node_struct = sizeof(KDNode);

    long long N = 1000000;
    long long D = 16;

    // Memory calculation: raw data vs tree structure
    long long dynamic_data_per_point = D * size_double;
    long long raw_total_bytes = N * (size_point_struct + dynamic_data_per_point);
    long long tree_total_bytes = N * (size_node_struct + dynamic_data_per_point);
    long long structural_overhead_bytes = tree_total_bytes - raw_total_bytes;
    double raw_mb = raw_total_bytes / (1024.0 * 1024.0);
    double tree_mb = tree_total_bytes / (1024.0 * 1024.0);
    double overhead_mb = structural_overhead_bytes / (1024.0 * 1024.0);


    double raw_percent = (static_cast<double>(raw_total_bytes) / tree_total_bytes) * 100.0;
    double overhead_percent = (static_cast<double>(structural_overhead_bytes) / tree_total_bytes) * 100.0;

    std::cout << "Architecture Details:" << std::endl;
    std::cout << "  sizeof(Point):   " << size_point_struct << " bytes (Vector overhead + Label)" << std::endl;
    std::cout << "  sizeof(KDNode):  " << size_node_struct << " bytes (Point + Pointers + Disc)" << std::endl;
    std::cout << "  Dynamic Data:    " << dynamic_data_per_point << " bytes per point (" << D << " doubles)" << std::endl;

    std::cout << "\nProjection for N=1,000,000 points (16 Dimensions):" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  1. Useful Data (Raw):      " << raw_mb << " MB" << std::endl;
    std::cout << "  2. Tree Structure (Total): " << tree_mb << " MB" << std::endl;
    std::cout << "  3. Overhead Cost:          " << overhead_mb << " MB" << std::endl;

    std::cout << "\nMemory Distribution Table:" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << " Component          | Size (MB) | Percentage " << std::endl;
    std::cout << "--------------------|-----------|-----------" << std::endl;
    std::cout << " Raw Data (Content) | " << std::setw(9) << raw_mb << " | " << std::setw(8) << raw_percent << "% " << std::endl;
    std::cout << " Tree Overhead      | " << std::setw(9) << overhead_mb << " | " << std::setw(8) << overhead_percent << "% " << std::endl;
    std::cout << "--------------------|-----------|-----------" << std::endl;
    std::cout << " TOTAL              | " << std::setw(9) << tree_mb << " |   100.00% " << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

void runMetricImpactAnalysis()
{
    std::cout << "\n===========================================================" << std::endl;
    std::cout << "   [EXTENSION] METRIC IMPACT ANALYSIS (Euclidean vs Manhattan)" << std::endl;
    std::cout << "===========================================================" << std::endl;

    int n_samples = 50000;
    int dims = 12;
    int k = 5;

    std::cout << "Generating " << n_samples << " points in " << dims << "D..." << std::endl;
    auto data = SyntheticDataGenerator::generateUniform(n_samples, dims, 123);
    auto queries = SyntheticDataGenerator::generateUniform(500, dims, 456);

    Timer timer;
    std::cout << "Testing Euclidean Metric..." << std::endl;
    KNNKDTree treeEuc(k, dims, DistanceType::EUCLIDEAN);
    treeEuc.fit(data);

    timer.start();
    for (const auto &q : queries)
        treeEuc.predict(q);
    double t_euc = timer.elapsed_ms();
    std::cout << "Testing Manhattan Metric..." << std::endl;
    KNNKDTree treeMan(k, dims, DistanceType::MANHATTAN);
    treeMan.fit(data);

    timer.start();
    for (const auto &q : queries)
        treeMan.predict(q);
    double t_man = timer.elapsed_ms();

    std::cout << "\nResults (500 queries):" << std::endl;
    std::cout << "  Euclidean Time: " << t_euc << " ms" << std::endl;
    std::cout << "  Manhattan Time: " << t_man << " ms" << std::endl;
    std::cout << "  Speedup:        " << (t_euc / t_man) << "x" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
}

int main(int argc, char *argv[])
{
    runMemoryAnalysis();
    runMetricImpactAnalysis();
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "   STARTING STANDARD BENCHMARK SUITE" << std::endl;
    std::cout << "   Comparing: KNNBasic, KNNKDTree, KNNNanoflann" << std::endl;
    std::cout << "========================================" << std::endl;

    std::filesystem::create_directories("benchmarks/results");
    BenchmarkRunner runner;
    Timer globalTimer;
    globalTimer.start();

    runner.runAllBenchmarks(REAL_DATASETS);

    double total_duration = globalTimer.elapsed_sec();
    std::string json_output = "benchmarks/results/benchmark_results.json";
    std::string csv_output = "benchmarks/results/benchmark_comprehensive.csv";

    runner.saveResults(json_output, total_duration);
    runner.saveCSVResults(csv_output, total_duration);

    std::cout << "\nBenchmark suite completed!" << std::endl;
    return 0;
}