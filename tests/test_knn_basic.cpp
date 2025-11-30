#include <iostream>
#include <string>
#include <chrono>
#include "../include/knn/knn_basic.h"
#include "../include/utils/dataset_loader.h"
#include "../include/utils/metrics.h"

void printUsage() {
    std::cout << "Usage: test_knn_basic <csv_file> <k> [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --no-header         CSV file has no header row\n";
    std::cout << "  --auto-encode       Automatically detect and one-hot encode categorical columns\n";
    std::cout << "  --distance <type>   Distance metric: euclidean, manhattan, hamming, minkowski\n";
    std::cout << "  --minkowski-p <p>   Parameter p for Minkowski distance (default: 2.0)\n";
    std::cout << "  --test-ratio <r>    Test set ratio (default: 0.2)\n";
    std::cout << "  --output <file>     Output JSON file for metrics (default: metrics.json)\n";
    std::cout << "\nExample:\n";
    std::cout << "  test_knn_basic iris.csv 5 --auto-encode --distance manhattan\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage();
        return 1;
    }

    // Parse arguments
    std::string csvFile = argv[1];
    int k = std::stoi(argv[2]);

    bool hasHeader = true;
    bool autoEncode = false;
    DistanceType distMetric = DistanceType::EUCLIDEAN;
    double minkowskiP = 2.0;
    double testRatio = 0.2;
    std::string outputFile = "metrics.json";

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-header") {
            hasHeader = false;
        } else if (arg == "--auto-encode") {
            autoEncode = true;
        } else if (arg == "--distance" && i + 1 < argc) {
            std::string dist = argv[++i];
            if (dist == "euclidean") distMetric = DistanceType::EUCLIDEAN;
            else if (dist == "manhattan") distMetric = DistanceType::MANHATTAN;
            else if (dist == "hamming") distMetric = DistanceType::HAMMING;
            else if (dist == "minkowski") distMetric = DistanceType::MINKOWSKI;
        } else if (arg == "--minkowski-p" && i + 1 < argc) {
            minkowskiP = std::stod(argv[++i]);
        } else if (arg == "--test-ratio" && i + 1 < argc) {
            testRatio = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            outputFile = argv[++i];
        }
    }

    std::cout << "=== KNN Basic Classifier Test ===" << std::endl;
    std::cout << "Dataset: " << csvFile << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "Auto-encode: " << (autoEncode ? "Yes" : "No") << std::endl;
    std::cout << "Test ratio: " << (testRatio * 100) << "%" << std::endl;

    try {
        // Load dataset
        std::cout << "\nLoading dataset..." << std::endl;
        std::vector<Point> data;

        if (autoEncode) {
            data = DatasetLoader::loadCSVWithEncoding(csvFile, hasHeader);
            std::cout << "Loaded with automatic categorical encoding" << std::endl;
        } else {
            data = DatasetLoader::loadCSV(csvFile, hasHeader);
            std::cout << "Loaded as numeric data" << std::endl;
        }

        std::cout << "Total samples: " << data.size() << std::endl;
        if (!data.empty()) {
            std::cout << "Dimensions: " << data[0].dimensions() << std::endl;
        }

        // Split into train/test
        std::cout << "\nSplitting dataset..." << std::endl;
        std::vector<Point> train, test;
        DatasetLoader::trainTestSplit(data, train, test, testRatio);

        std::cout << "Training samples: " << train.size() << std::endl;
        std::cout << "Test samples: " << test.size() << std::endl;

        // Train KNN
        std::cout << "\nTraining KNN..." << std::endl;
        auto startTrain = std::chrono::high_resolution_clock::now();

        KNNBasic knn(k, distMetric, minkowskiP);
        knn.fit(train);

        auto endTrain = std::chrono::high_resolution_clock::now();
        auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTrain - startTrain);
        std::cout << "Training time: " << trainTime.count() << " ms" << std::endl;

        // Predict on test set
        std::cout << "\nTesting KNN..." << std::endl;
        auto startTest = std::chrono::high_resolution_clock::now();

        std::vector<int> true_labels;
        std::vector<int> predicted_labels;

        for (const auto& point : test) {
            int prediction = knn.predict(point);
            true_labels.push_back(point.label);
            predicted_labels.push_back(prediction);
        }

        auto endTest = std::chrono::high_resolution_clock::now();
        auto testTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTest - startTest);

        std::cout << "Testing time: " << testTime.count() << " ms" << std::endl;
        std::cout << "Average prediction time: "
                  << (static_cast<double>(testTime.count()) / test.size())
                  << " ms/sample" << std::endl;

        // Evaluate metrics
        std::cout << "\nEvaluating metrics..." << std::endl;
        Metrics::printMetrics(true_labels, predicted_labels);

        // Save to JSON
        Metrics::saveMetricsJSON(true_labels, predicted_labels, outputFile, "KNN_Basic");

        std::cout << "\n=== Test Complete ===" << std::endl;
        std::cout << "Results saved to: " << outputFile << std::endl;
        std::cout << "\nTo visualize results, run:" << std::endl;
        std::cout << "  python visualization/visualize_metrics.py " << outputFile << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
