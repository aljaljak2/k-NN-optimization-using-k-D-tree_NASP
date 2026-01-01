#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <map>
#include "../../include/utils/point.h"

// Benchmark result for a single test
struct BenchmarkResult {
    std::string algorithm;
    std::string dataset_name;
    int n_samples;
    int n_dimensions;
    int k_neighbors;
    int n_queries;
    double build_time_ms;
    double total_query_time_ms;
    double avg_query_time_ms;
    double speedup_vs_basic;

    // Classification metrics
    double accuracy;      // -1.0 if not applicable
    double precision;     // -1.0 if not applicable
    double recall;        // -1.0 if not applicable
    double f1_score;      // -1.0 if not applicable

    // Distance calculation metrics
    long long total_distance_calculations;
    double avg_distance_calculations_per_query;
};

// Benchmark suite info
struct BenchmarkInfo {
    std::string timestamp;
    int total_tests;
    double total_duration_sec;
};

// CSV loader (simplified version for benchmarks)
class CSVLoader {
public:
    // labelColumn: -1 = last column, 0 = first column, -2 = second to last, N = specific column index
    static std::vector<Point> load(const std::string& filepath, bool hasHeader = true, int labelColumn = -1);
};

// Synthetic data generator
class SyntheticDataGenerator {
public:
    static std::vector<Point> generateUniform(int n_samples, int n_dimensions, int seed = 42);
    static std::vector<Point> generateClustered(int n_clusters, int samples_per_cluster,
                                                  int n_dimensions, int seed = 42);
};

// Train/test split utility
class DataSplitter {
public:
    static void trainTestSplit(const std::vector<Point>& data,
                               std::vector<Point>& train,
                               std::vector<Point>& test,
                               double test_ratio = 0.2,
                               int seed = 42);
};

// JSON output generator
class JSONWriter {
public:
    static void writeBenchmarkResults(const std::string& filepath,
                                       const BenchmarkInfo& info,
                                       const std::vector<BenchmarkResult>& results);

private:
    static std::string escapeJSON(const std::string& str);
};

// CSV output generator for comprehensive metrics
class CSVWriter {
public:
    static void writeComprehensiveResults(const std::string& filepath,
                                          const BenchmarkInfo& info,
                                          const std::vector<BenchmarkResult>& results);

private:
    static void writeSyntheticMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results);
    static void writeRealDatasetMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results);
    static void writeSpeedupTable(std::ofstream& file, const std::vector<BenchmarkResult>& results);
    static void writeDistanceCalculationMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results);
};

// High-resolution timer utility
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start();
    double elapsed_ms() const;
    double elapsed_sec() const;
};

// Classification metrics calculator
struct ClassificationMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1_score;
};

class MetricsCalculator {
public:
    // Calculate accuracy, precision, recall, F1 score
    static ClassificationMetrics calculateMetrics(
        const std::vector<int>& true_labels,
        const std::vector<int>& predicted_labels
    );

    // Calculate per-class metrics
    static std::map<int, ClassificationMetrics> calculatePerClassMetrics(
        const std::vector<int>& true_labels,
        const std::vector<int>& predicted_labels
    );

private:
    static double calculateAccuracy(const std::vector<int>& true_labels,
                                     const std::vector<int>& predicted_labels);
    static double calculatePrecision(const std::vector<int>& true_labels,
                                      const std::vector<int>& predicted_labels,
                                      int target_class);
    static double calculateRecall(const std::vector<int>& true_labels,
                                   const std::vector<int>& predicted_labels,
                                   int target_class);
};

#endif // BENCHMARK_UTILS_H
