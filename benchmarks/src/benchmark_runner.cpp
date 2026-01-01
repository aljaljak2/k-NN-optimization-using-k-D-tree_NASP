#include "../include/benchmark_runner.h"
#include "../../include/utils/distance_metrics.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

BenchmarkRunner::BenchmarkRunner() : totalTests(0), currentTest(0) {}

void BenchmarkRunner::reportProgress(const std::string& message) {
    std::cout << "[" << currentTest << "/" << totalTests << "] " << message << std::endl;
}

double BenchmarkRunner::calculateAccuracy(const std::vector<Point>& train,
                                           const std::vector<Point>& test,
                                           const std::string& algorithm,
                                           int k, int dimensions) {
    int correct = 0;
    int total = test.size();

    if (algorithm == "KNNBasic") {
        KNNBasic knn(k);
        knn.fit(train);
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            if (predicted == query.label) correct++;
        }
    } else if (algorithm == "KNNKDTree") {
        KNNKDTree knn(k, dimensions);
        knn.fit(train);
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            if (predicted == query.label) correct++;
        }
    } else if (algorithm == "KNNNanoflann") {
        KNNNanoflann knn(k, dimensions);
        knn.fit(train);
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            if (predicted == query.label) correct++;
        }
    }

    return (total > 0) ? (100.0 * correct / total) : 0.0;
}

ClassificationMetrics BenchmarkRunner::calculateClassificationMetrics(
    const std::vector<Point>& train,
    const std::vector<Point>& test,
    const std::string& algorithm,
    int k, int dimensions,
    long long& total_distance_calcs) {

    std::vector<int> true_labels;
    std::vector<int> predicted_labels;

    if (algorithm == "KNNBasic") {
        KNNBasic knn(k);
        knn.fit(train);
        DistanceMetrics::resetCounter();
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            predicted_labels.push_back(predicted);
            true_labels.push_back(query.label);
        }
        total_distance_calcs = DistanceMetrics::getCounter();

    } else if (algorithm == "KNNKDTree") {
        KNNKDTree knn(k, dimensions);
        knn.fit(train);
        knn.resetDistanceCount();
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            predicted_labels.push_back(predicted);
            true_labels.push_back(query.label);
        }
        total_distance_calcs = knn.getDistanceCount();

    } else if (algorithm == "KNNNanoflann") {
        KNNNanoflann knn(k, dimensions);
        knn.fit(train);
        knn.resetDistanceCount();
        for (const auto& query : test) {
            int predicted = knn.predict(query);
            predicted_labels.push_back(predicted);
            true_labels.push_back(query.label);
        }
        total_distance_calcs = knn.getDistanceCount();
    }

    // Calculate comprehensive metrics
    ClassificationMetrics metrics = MetricsCalculator::calculateMetrics(true_labels, predicted_labels);

    // Convert accuracy to percentage
    metrics.accuracy *= 100.0;
    metrics.precision *= 100.0;
    metrics.recall *= 100.0;
    metrics.f1_score *= 100.0;

    return metrics;
}

BenchmarkResult BenchmarkRunner::benchmarkAlgorithm(const std::string& algorithm,
                                                      const std::vector<Point>& train,
                                                      const std::vector<Point>& queries,
                                                      const std::string& dataset_name,
                                                      int k, int dimensions) {
    BenchmarkResult result;
    result.algorithm = algorithm;
    result.dataset_name = dataset_name;
    result.n_samples = train.size();
    result.n_dimensions = dimensions;
    result.k_neighbors = k;
    result.n_queries = queries.size();
    result.build_time_ms = 0.0;
    result.total_query_time_ms = 0.0;
    result.avg_query_time_ms = 0.0;
    result.speedup_vs_basic = 1.0;

    // Initialize classification metrics
    result.accuracy = -1.0;
    result.precision = -1.0;
    result.recall = -1.0;
    result.f1_score = -1.0;

    // Initialize distance calculation metrics
    result.total_distance_calculations = 0;
    result.avg_distance_calculations_per_query = 0.0;

    Timer timer;

    if (algorithm == "KNNBasic") {
        KNNBasic knn(k);

        // Build time (minimal for brute force)
        timer.start();
        knn.fit(train);
        result.build_time_ms = timer.elapsed_ms();

        // Warmup
        if (!queries.empty()) {
            knn.predict(queries[0]);
        }

        // Reset counter and measure query time with actual distance calculations
        DistanceMetrics::resetCounter();
        timer.start();
        for (const auto& query : queries) {
            knn.predict(query);
        }
        result.total_query_time_ms = timer.elapsed_ms();
        result.total_distance_calculations = DistanceMetrics::getCounter();

    } else if (algorithm == "KNNKDTree") {
        KNNKDTree knn(k, dimensions);

        // Build time
        timer.start();
        knn.fit(train);
        result.build_time_ms = timer.elapsed_ms();

        // Warmup
        if (!queries.empty()) {
            knn.predict(queries[0]);
        }

        // Reset counter and measure query time with actual distance calculations
        knn.resetDistanceCount();
        timer.start();
        for (const auto& query : queries) {
            knn.predict(query);
        }
        result.total_query_time_ms = timer.elapsed_ms();
        result.total_distance_calculations = knn.getDistanceCount();

    } else if (algorithm == "KNNNanoflann") {
        KNNNanoflann knn(k, dimensions);

        // Build time
        timer.start();
        knn.fit(train);
        result.build_time_ms = timer.elapsed_ms();

        // Warmup
        if (!queries.empty()) {
            knn.predict(queries[0]);
        }

        // Reset counter and measure query time with actual distance calculations
        knn.resetDistanceCount();
        timer.start();
        for (const auto& query : queries) {
            knn.predict(query);
        }
        result.total_query_time_ms = timer.elapsed_ms();
        result.total_distance_calculations = knn.getDistanceCount();
    }

    result.avg_query_time_ms = result.total_query_time_ms / queries.size();

    // Calculate average distance calculations per query
    result.avg_distance_calculations_per_query = (queries.size() > 0) ?
        static_cast<double>(result.total_distance_calculations) / queries.size() : 0.0;

    // Calculate speedup vs basic
    std::string key = dataset_name + "_" + std::to_string(k);
    if (algorithm == "KNNBasic") {
        basicQueryTimes[key] = result.total_query_time_ms;
    } else {
        if (basicQueryTimes.find(key) != basicQueryTimes.end() && basicQueryTimes[key] > 0) {
            result.speedup_vs_basic = basicQueryTimes[key] / result.total_query_time_ms;
        }
    }

    return result;
}

void BenchmarkRunner::runCurseOfDimensionality() {
    std::cout << "\n=== Running Curse of Dimensionality Test ===" << std::endl;

    std::vector<int> dimensions = {2, 4, 8, 16, 32, 64};
    int n_samples = 5000;
    int k = 5;
    int n_queries = 500;

    for (int d : dimensions) {
        std::string dataset_name = "synthetic_" + std::to_string(d) + "d";
        std::cout << "\nTesting dimension: " << d << std::endl;

        // Generate data
        auto data = SyntheticDataGenerator::generateUniform(n_samples, d, 42);
        std::vector<Point> train, test;
        DataSplitter::trainTestSplit(data, train, test, 0.1, 42); // Use 10% for queries

        std::vector<Point> queries(test.begin(), test.begin() + std::min(n_queries, (int)test.size()));

        // Benchmark each algorithm
        for (const auto& algo : {"KNNBasic", "KNNKDTree", "KNNNanoflann"}) {
            currentTest++;
            reportProgress("Testing " + std::string(algo) + " on " + dataset_name);
            results.push_back(benchmarkAlgorithm(algo, train, queries, dataset_name, k, d));
        }
    }
}

void BenchmarkRunner::runScalability() {
    std::cout << "\n=== Running Scalability Test ===" << std::endl;

    std::vector<int> sample_sizes = {100, 500, 1000, 5000, 10000, 20000};
    int d = 8;
    int k = 5;
    int n_queries = 100;

    for (int n : sample_sizes) {
        std::string dataset_name = "synthetic_n" + std::to_string(n);
        std::cout << "\nTesting sample size: " << n << std::endl;

        // Generate data
        auto data = SyntheticDataGenerator::generateUniform(n, d, 42);
        std::vector<Point> train, test;
        DataSplitter::trainTestSplit(data, train, test, 0.1, 42);

        std::vector<Point> queries(test.begin(), test.begin() + std::min(n_queries, (int)test.size()));

        // Benchmark each algorithm
        for (const auto& algo : {"KNNBasic", "KNNKDTree", "KNNNanoflann"}) {
            currentTest++;
            reportProgress("Testing " + std::string(algo) + " on " + dataset_name);
            results.push_back(benchmarkAlgorithm(algo, train, queries, dataset_name, k, d));
        }
    }
}

void BenchmarkRunner::runKParameterImpact() {
    std::cout << "\n=== Running K Parameter Impact Test ===" << std::endl;

    std::vector<int> k_values = {1, 3, 5, 10, 20, 50, 100};
    int n_samples = 5000;
    int d = 8;
    int n_queries = 100;

    // Generate data once
    auto data = SyntheticDataGenerator::generateUniform(n_samples, d, 42);
    std::vector<Point> train, test;
    DataSplitter::trainTestSplit(data, train, test, 0.1, 42);

    std::vector<Point> queries(test.begin(), test.begin() + std::min(n_queries, (int)test.size()));

    for (int k : k_values) {
        std::string dataset_name = "synthetic_k" + std::to_string(k);
        std::cout << "\nTesting k: " << k << std::endl;

        // Benchmark each algorithm
        for (const auto& algo : {"KNNBasic", "KNNKDTree", "KNNNanoflann"}) {
            currentTest++;
            reportProgress("Testing " + std::string(algo) + " with k=" + std::to_string(k));
            results.push_back(benchmarkAlgorithm(algo, train, queries, dataset_name, k, d));
        }
    }
}

void BenchmarkRunner::runRealDatasets(const std::vector<DatasetConfig>& datasets) {
    std::cout << "\n=== Running Real Datasets Test ===" << std::endl;

    std::vector<int> k_values = {1, 5, 10};

    for (const auto& dataset : datasets) {
        std::cout << "\nLoading dataset: " << dataset.filepath << std::endl;

        // Load dataset with specified label column
        auto data = CSVLoader::load(dataset.filepath, true, dataset.labelColumn);

        if (data.empty()) {
            std::cout << "Skipping empty or missing dataset: " << dataset.filepath << std::endl;
            continue;
        }

        // Limit dataset size to 10,000 samples for faster benchmarking
        const size_t MAX_SAMPLES = 10000;
        if (data.size() > MAX_SAMPLES) {
            std::cout << "Limiting dataset from " << data.size() << " to " << MAX_SAMPLES << " samples" << std::endl;
            data.resize(MAX_SAMPLES);
        }

        // Extract dataset name from path
        std::string dataset_name = dataset.filepath;
        size_t last_slash = dataset.filepath.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            dataset_name = dataset.filepath.substr(last_slash + 1);
        }
        size_t last_dot = dataset_name.find_last_of(".");
        if (last_dot != std::string::npos) {
            dataset_name = dataset_name.substr(0, last_dot);
        }

        int dimensions = data[0].dimensions();
        std::cout << "Loaded " << data.size() << " samples with " << dimensions << " dimensions" << std::endl;

        // Train/test split
        std::vector<Point> train, test;
        DataSplitter::trainTestSplit(data, train, test, 0.2, 42);

        for (int k : k_values) {
            std::cout << "\nTesting k=" << k << " on " << dataset_name << std::endl;

            // Benchmark each algorithm
            for (const auto& algo : {"KNNBasic", "KNNKDTree", "KNNNanoflann"}) {
                currentTest++;
                reportProgress("Testing " + std::string(algo) + " on " + dataset_name + " (k=" + std::to_string(k) + ")");

                auto result = benchmarkAlgorithm(algo, train, test, dataset_name + "_k" + std::to_string(k), k, dimensions);

                // Calculate all classification metrics for real datasets
                long long dist_calcs = 0;
                ClassificationMetrics metrics = calculateClassificationMetrics(train, test, algo, k, dimensions, dist_calcs);

                result.accuracy = metrics.accuracy;
                result.precision = metrics.precision;
                result.recall = metrics.recall;
                result.f1_score = metrics.f1_score;
                result.total_distance_calculations = dist_calcs;
                result.avg_distance_calculations_per_query = (test.size() > 0) ?
                    static_cast<double>(dist_calcs) / test.size() : 0.0;

                results.push_back(result);
            }
        }
    }
}

void BenchmarkRunner::runAllBenchmarks(const std::vector<DatasetConfig>& real_datasets) {
    // Calculate total number of tests
    totalTests = 0;
    totalTests += 6 * 3;  // Curse of dimensionality: 6 dimensions * 3 algorithms
    totalTests += 6 * 3;  // Scalability: 6 sample sizes * 3 algorithms
    totalTests += 7 * 3;  // K parameter: 7 k values * 3 algorithms
    totalTests += real_datasets.size() * 3 * 3;  // Real datasets: N datasets * 3 k values * 3 algorithms

    currentTest = 0;

    std::cout << "Starting benchmark suite with " << totalTests << " total tests..." << std::endl;

    runCurseOfDimensionality();
    runScalability();
    runKParameterImpact();
    runRealDatasets(real_datasets);

    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    std::cout << "Total tests run: " << currentTest << std::endl;
}

void BenchmarkRunner::saveResults(const std::string& filepath, double total_duration_sec) {
    // Get current timestamp
    time_t now = time(nullptr);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    BenchmarkInfo info;
    info.timestamp = timestamp;
    info.total_tests = results.size();
    info.total_duration_sec = total_duration_sec;

    JSONWriter::writeBenchmarkResults(filepath, info, results);

    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total tests: " << info.total_tests << std::endl;
    std::cout << "Total duration: " << std::fixed << std::setprecision(2)
              << info.total_duration_sec << " seconds" << std::endl;
}

void BenchmarkRunner::saveCSVResults(const std::string& filepath, double total_duration_sec) {
    // Get current timestamp
    time_t now = time(nullptr);
    char timestamp[100];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));

    BenchmarkInfo info;
    info.timestamp = timestamp;
    info.total_tests = results.size();
    info.total_duration_sec = total_duration_sec;

    CSVWriter::writeComprehensiveResults(filepath, info, results);
    std::cout << "Results saved to: " << filepath << std::endl;
}
