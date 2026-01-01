#include "../include/benchmark_utils.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <set>
#include <map>

// Helper function to check if column name contains "id" (case-insensitive)
static bool isIdColumn(const std::string& columnName) {
    std::string lower = columnName;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Check for "id" in various forms
    return lower == "id" ||
           lower.find("_id") != std::string::npos ||
           lower.find("id_") != std::string::npos ||
           lower.find("-id") != std::string::npos ||
           lower.find("id-") != std::string::npos;
}

// CSV Loader Implementation
std::vector<Point> CSVLoader::load(const std::string& filepath, bool hasHeader, int labelColumn) {
    std::vector<Point> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file: " << filepath << std::endl;
        return data;
    }

    std::string line;
    bool firstLine = true;
    std::vector<std::string> headers;
    std::vector<bool> ignoreColumn;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> values;

        // Parse all values first
        while (std::getline(ss, value, ',')) {
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            if (!value.empty()) {
                values.push_back(value);
            }
        }

        if (values.empty()) continue;

        // Parse header and determine which columns to ignore
        if (firstLine && hasHeader) {
            headers = values;
            ignoreColumn.resize(headers.size(), false);

            // Mark ID columns to ignore
            for (size_t i = 0; i < headers.size(); ++i) {
                if (isIdColumn(headers[i])) {
                    ignoreColumn[i] = true;
                    std::cout << "Ignoring ID column: " << headers[i] << std::endl;
                }
            }

            firstLine = false;
            continue;
        }

        // Determine actual label column index
        int actualLabelCol;
        if (labelColumn == -1) {
            actualLabelCol = values.size() - 1;  // Last column
        } else if (labelColumn == -2) {
            actualLabelCol = values.size() - 2;  // Second to last
        } else if (labelColumn < 0) {
            actualLabelCol = values.size() + labelColumn;  // From end
        } else {
            actualLabelCol = labelColumn;  // Specific index
        }

        // Extract label
        int label = 0;
        if (actualLabelCol >= 0 && actualLabelCol < (int)values.size()) {
            try {
                label = static_cast<int>(std::stod(values[actualLabelCol]));
            } catch (...) {
                // If conversion fails, hash the string
                label = static_cast<int>(std::hash<std::string>{}(values[actualLabelCol]) % 100);
            }
        }

        // Extract features (all columns except label and ignored columns)
        std::vector<double> coords;
        for (int i = 0; i < (int)values.size(); ++i) {
            if (i == actualLabelCol) continue;  // Skip label column

            // Skip ID columns
            if (!ignoreColumn.empty() && i < (int)ignoreColumn.size() && ignoreColumn[i]) {
                continue;
            }

            try {
                coords.push_back(std::stod(values[i]));
            } catch (...) {
                // Skip non-numeric values
            }
        }

        if (!coords.empty()) {
            data.push_back(Point(coords, label));
        }
    }

    file.close();
    return data;
}

// Synthetic Data Generator Implementation
std::vector<Point> SyntheticDataGenerator::generateUniform(int n_samples, int n_dimensions, int seed) {
    std::vector<Point> data;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::uniform_int_distribution<int> label_dist(0, 9);

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> coords(n_dimensions);
        for (int d = 0; d < n_dimensions; ++d) {
            coords[d] = dist(rng);
        }
        int label = label_dist(rng);
        data.push_back(Point(coords, label));
    }

    return data;
}

std::vector<Point> SyntheticDataGenerator::generateClustered(int n_clusters, int samples_per_cluster,
                                                               int n_dimensions, int seed) {
    std::vector<Point> data;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> cluster_center_dist(0.0, 100.0);
    std::normal_distribution<double> point_dist(0.0, 5.0);

    for (int c = 0; c < n_clusters; ++c) {
        // Generate cluster center
        std::vector<double> center(n_dimensions);
        for (int d = 0; d < n_dimensions; ++d) {
            center[d] = cluster_center_dist(rng);
        }

        // Generate points around center
        for (int i = 0; i < samples_per_cluster; ++i) {
            std::vector<double> coords(n_dimensions);
            for (int d = 0; d < n_dimensions; ++d) {
                coords[d] = center[d] + point_dist(rng);
            }
            data.push_back(Point(coords, c));
        }
    }

    return data;
}

// Data Splitter Implementation
void DataSplitter::trainTestSplit(const std::vector<Point>& data,
                                   std::vector<Point>& train,
                                   std::vector<Point>& test,
                                   double test_ratio,
                                   int seed) {
    train.clear();
    test.clear();

    std::vector<Point> shuffled = data;
    std::mt19937 rng(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    size_t test_size = static_cast<size_t>(data.size() * test_ratio);
    size_t train_size = data.size() - test_size;

    train.assign(shuffled.begin(), shuffled.begin() + train_size);
    test.assign(shuffled.begin() + train_size, shuffled.end());
}

// JSON Writer Implementation
std::string JSONWriter::escapeJSON(const std::string& str) {
    std::string escaped;
    for (char c : str) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:   escaped += c; break;
        }
    }
    return escaped;
}

void JSONWriter::writeBenchmarkResults(const std::string& filepath,
                                        const BenchmarkInfo& info,
                                        const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write to file: " << filepath << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(4);
    file << "{\n";
    file << "  \"benchmark_info\": {\n";
    file << "    \"timestamp\": \"" << escapeJSON(info.timestamp) << "\",\n";
    file << "    \"total_tests\": " << info.total_tests << ",\n";
    file << "    \"total_duration_sec\": " << info.total_duration_sec << "\n";
    file << "  },\n";
    file << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"algorithm\": \"" << escapeJSON(r.algorithm) << "\",\n";
        file << "      \"dataset_name\": \"" << escapeJSON(r.dataset_name) << "\",\n";
        file << "      \"n_samples\": " << r.n_samples << ",\n";
        file << "      \"n_dimensions\": " << r.n_dimensions << ",\n";
        file << "      \"k_neighbors\": " << r.k_neighbors << ",\n";
        file << "      \"n_queries\": " << r.n_queries << ",\n";
        file << "      \"build_time_ms\": " << r.build_time_ms << ",\n";
        file << "      \"total_query_time_ms\": " << r.total_query_time_ms << ",\n";
        file << "      \"avg_query_time_ms\": " << r.avg_query_time_ms << ",\n";
        file << "      \"speedup_vs_basic\": " << r.speedup_vs_basic << ",\n";

        // Classification metrics
        if (r.accuracy >= 0.0) {
            file << "      \"accuracy\": " << r.accuracy << ",\n";
        } else {
            file << "      \"accuracy\": null,\n";
        }

        if (r.precision >= 0.0) {
            file << "      \"precision\": " << r.precision << ",\n";
        } else {
            file << "      \"precision\": null,\n";
        }

        if (r.recall >= 0.0) {
            file << "      \"recall\": " << r.recall << ",\n";
        } else {
            file << "      \"recall\": null,\n";
        }

        if (r.f1_score >= 0.0) {
            file << "      \"f1_score\": " << r.f1_score << ",\n";
        } else {
            file << "      \"f1_score\": null,\n";
        }

        // Distance calculation metrics
        file << "      \"total_distance_calculations\": " << r.total_distance_calculations << ",\n";
        file << "      \"avg_distance_calculations_per_query\": " << r.avg_distance_calculations_per_query << "\n";

        file << "    }" << (i < results.size() - 1 ? "," : "") << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Benchmark results saved to: " << filepath << std::endl;
}

// Timer Implementation
void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

double Timer::elapsed_sec() const {
    return elapsed_ms() / 1000.0;
}

// CSV Writer Implementation
void CSVWriter::writeComprehensiveResults(const std::string& filepath,
                                          const BenchmarkInfo& info,
                                          const std::vector<BenchmarkResult>& results) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write to file: " << filepath << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(4);

    // Write header
    file << "# KNN BENCHMARK COMPREHENSIVE RESULTS\n";
    file << "# Timestamp: " << info.timestamp << "\n";
    file << "# Total tests: " << info.total_tests << "\n";
    file << "# Total duration: " << info.total_duration_sec << " seconds\n";
    file << "#\n\n";

    // Write speedup table
    writeSpeedupTable(file, results);
    file << "\n\n";

    // Write synthetic data metrics
    writeSyntheticMetrics(file, results);
    file << "\n\n";

    // Write real dataset metrics
    writeRealDatasetMetrics(file, results);
    file << "\n\n";

    // Write distance calculation metrics
    writeDistanceCalculationMetrics(file, results);

    file.close();
    std::cout << "Comprehensive CSV results saved to: " << filepath << std::endl;
}

void CSVWriter::writeSpeedupTable(std::ofstream& file, const std::vector<BenchmarkResult>& results) {
    file << "# TABLE 1: SPEEDUP COMPARISON (vs KNNBasic baseline)\n";
    file << "Test,Algorithm,Dataset,Dimensions,Samples,K,Query_Time_ms,Speedup\n";

    for (const auto& r : results) {
        file << r.dataset_name << ","
             << r.algorithm << ","
             << r.dataset_name << ","
             << r.n_dimensions << ","
             << r.n_samples << ","
             << r.k_neighbors << ","
             << r.avg_query_time_ms << ","
             << r.speedup_vs_basic << "\n";
    }
}

void CSVWriter::writeSyntheticMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results) {
    file << "# TABLE 2: SYNTHETIC DATA PERFORMANCE METRICS\n";
    file << "Test_Type,Algorithm,Dimensions,Samples,K,Build_Time_ms,Avg_Query_Time_ms,Total_Query_Time_ms,Speedup,Dist_Calc_Per_Query\n";

    for (const auto& r : results) {
        // Filter synthetic results
        if (r.dataset_name.find("synthetic") != std::string::npos) {
            std::string test_type;
            if (r.dataset_name.find("_d") != std::string::npos) {
                test_type = "Curse_of_Dimensionality";
            } else if (r.dataset_name.find("_n") != std::string::npos) {
                test_type = "Scalability";
            } else if (r.dataset_name.find("_k") != std::string::npos) {
                test_type = "K_Parameter";
            }

            file << test_type << ","
                 << r.algorithm << ","
                 << r.n_dimensions << ","
                 << r.n_samples << ","
                 << r.k_neighbors << ","
                 << r.build_time_ms << ","
                 << r.avg_query_time_ms << ","
                 << r.total_query_time_ms << ","
                 << r.speedup_vs_basic << ","
                 << r.avg_distance_calculations_per_query << "\n";
        }
    }
}

void CSVWriter::writeRealDatasetMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results) {
    file << "# TABLE 3: REAL DATASET CLASSIFICATION METRICS\n";
    file << "Dataset,Algorithm,Dimensions,Samples,K,Accuracy,Precision,Recall,F1_Score,Avg_Query_Time_ms,Speedup,Dist_Calc_Per_Query\n";

    for (const auto& r : results) {
        // Filter real dataset results
        if (r.dataset_name.find("synthetic") == std::string::npos &&
            r.dataset_name.find("_k") != std::string::npos) {

            // Extract dataset name without _k suffix
            std::string dataset = r.dataset_name;
            size_t pos = dataset.find("_k");
            if (pos != std::string::npos) {
                dataset = dataset.substr(0, pos);
            }

            file << dataset << ","
                 << r.algorithm << ","
                 << r.n_dimensions << ","
                 << r.n_samples << ","
                 << r.k_neighbors << ",";

            // Write classification metrics
            if (r.accuracy >= 0.0) {
                file << r.accuracy;
            } else {
                file << "N/A";
            }
            file << ",";

            if (r.precision >= 0.0) {
                file << r.precision;
            } else {
                file << "N/A";
            }
            file << ",";

            if (r.recall >= 0.0) {
                file << r.recall;
            } else {
                file << "N/A";
            }
            file << ",";

            if (r.f1_score >= 0.0) {
                file << r.f1_score;
            } else {
                file << "N/A";
            }
            file << ",";

            file << r.avg_query_time_ms << ","
                 << r.speedup_vs_basic << ","
                 << r.avg_distance_calculations_per_query << "\n";
        }
    }
}

void CSVWriter::writeDistanceCalculationMetrics(std::ofstream& file, const std::vector<BenchmarkResult>& results) {
    file << "# TABLE 4: DISTANCE CALCULATION EFFICIENCY\n";
    file << "Algorithm,Dataset,Dimensions,Samples,K,Total_Dist_Calc,Avg_Dist_Calc_Per_Query,Theoretical_Max,Efficiency_Percent\n";

    for (const auto& r : results) {
        long long theoretical_max = static_cast<long long>(r.n_samples) * r.n_queries;
        double efficiency = (theoretical_max > 0) ?
            (1.0 - (static_cast<double>(r.total_distance_calculations) / theoretical_max)) * 100.0 : 0.0;

        file << r.algorithm << ","
             << r.dataset_name << ","
             << r.n_dimensions << ","
             << r.n_samples << ","
             << r.k_neighbors << ","
             << r.total_distance_calculations << ","
             << r.avg_distance_calculations_per_query << ","
             << theoretical_max << ","
             << efficiency << "\n";
    }
}

// MetricsCalculator Implementation
double MetricsCalculator::calculateAccuracy(const std::vector<int>& true_labels,
                                            const std::vector<int>& predicted_labels) {
    if (true_labels.size() != predicted_labels.size() || true_labels.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / true_labels.size();
}

double MetricsCalculator::calculatePrecision(const std::vector<int>& true_labels,
                                              const std::vector<int>& predicted_labels,
                                              int target_class) {
    int true_positives = 0;
    int false_positives = 0;

    for (size_t i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == target_class) {
            if (true_labels[i] == target_class) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }

    int total_predicted_positive = true_positives + false_positives;
    if (total_predicted_positive == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positives) / total_predicted_positive;
}

double MetricsCalculator::calculateRecall(const std::vector<int>& true_labels,
                                          const std::vector<int>& predicted_labels,
                                          int target_class) {
    int true_positives = 0;
    int false_negatives = 0;

    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == target_class) {
            if (predicted_labels[i] == target_class) {
                true_positives++;
            } else {
                false_negatives++;
            }
        }
    }

    int total_actual_positive = true_positives + false_negatives;
    if (total_actual_positive == 0) {
        return 0.0;
    }

    return static_cast<double>(true_positives) / total_actual_positive;
}

ClassificationMetrics MetricsCalculator::calculateMetrics(
    const std::vector<int>& true_labels,
    const std::vector<int>& predicted_labels) {

    ClassificationMetrics metrics;

    // Calculate accuracy
    metrics.accuracy = calculateAccuracy(true_labels, predicted_labels);

    // Get unique classes
    std::set<int> unique_classes;
    for (int label : true_labels) {
        unique_classes.insert(label);
    }

    // Calculate macro-averaged precision, recall, F1
    double total_precision = 0.0;
    double total_recall = 0.0;
    int num_classes = 0;

    for (int cls : unique_classes) {
        double precision = calculatePrecision(true_labels, predicted_labels, cls);
        double recall = calculateRecall(true_labels, predicted_labels, cls);

        total_precision += precision;
        total_recall += recall;
        num_classes++;
    }

    if (num_classes > 0) {
        metrics.precision = total_precision / num_classes;
        metrics.recall = total_recall / num_classes;

        // Calculate F1 score
        if (metrics.precision + metrics.recall > 0) {
            metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
        } else {
            metrics.f1_score = 0.0;
        }
    } else {
        metrics.precision = 0.0;
        metrics.recall = 0.0;
        metrics.f1_score = 0.0;
    }

    return metrics;
}

std::map<int, ClassificationMetrics> MetricsCalculator::calculatePerClassMetrics(
    const std::vector<int>& true_labels,
    const std::vector<int>& predicted_labels) {

    std::map<int, ClassificationMetrics> per_class_metrics;

    // Get unique classes
    std::set<int> unique_classes;
    for (int label : true_labels) {
        unique_classes.insert(label);
    }

    for (int cls : unique_classes) {
        ClassificationMetrics metrics;
        metrics.accuracy = calculateAccuracy(true_labels, predicted_labels);
        metrics.precision = calculatePrecision(true_labels, predicted_labels, cls);
        metrics.recall = calculateRecall(true_labels, predicted_labels, cls);

        if (metrics.precision + metrics.recall > 0) {
            metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall);
        } else {
            metrics.f1_score = 0.0;
        }

        per_class_metrics[cls] = metrics;
    }

    return per_class_metrics;
}
