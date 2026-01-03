#include <iostream>
#include <string>
#include <vector>
#include "../include/knn/knn_kdtree.h"
#include "../include/utils/dataset_loader.h"

void printUsage() {
    std::cout << "Usage: predict_knn_kdtree <csv_file> <k> [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --no-header                    CSV file has no header row\n";
    std::cout << "  --auto-encode                  Automatically detect and one-hot encode categorical columns\n";
    std::cout << "  --distance <type>              Distance metric: euclidean, manhattan, hamming, minkowski\n";
    std::cout << "  --minkowski-p <p>              Parameter p for Minkowski distance (default: 2.0)\n";
    std::cout << "  --label-column <idx>           Index of label column (default: -1 for last column)\n";
    std::cout << "  --predict-instance-index <idx> Index of instance to predict (0-based, within data rows)\n";
    std::cout << "\nExample:\n";
    std::cout << "  predict_knn_kdtree dataset.csv 5 --predict-instance-index 10 --auto-encode --distance manhattan\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage();
        return 1;
    }

    // Parse arguments
    std::string csvFile = argv[1];
    int k = std::stoi(argv[2]);

    // Parse options
    bool hasHeader = true;
    bool autoEncode = false;
    DistanceType distMetric = DistanceType::EUCLIDEAN;
    double minkowskiP = 2.0;
    int labelColumn = -1;
    int predictInstanceIndex = -1;  // Index of instance to predict

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
        } else if (arg == "--label-column" && i + 1 < argc) {
            labelColumn = std::stoi(argv[++i]);
        } else if (arg == "--predict-instance-index" && i + 1 < argc) {
            predictInstanceIndex = std::stoi(argv[++i]);
        }
    }

    if (predictInstanceIndex < 0) {
        std::cerr << "Error: --predict-instance-index is required\n";
        printUsage();
        return 1;
    }

    try {
        // Load full dataset (including the instance to predict)
        std::vector<Point> allData;

        if (autoEncode) {
            allData = DatasetLoader::loadCSVWithEncoding(csvFile, hasHeader, {}, labelColumn);
        } else {
            allData = DatasetLoader::loadCSV(csvFile, hasHeader, labelColumn);
        }

        if (allData.empty()) {
            std::cerr << "Error: Dataset is empty\n";
            return 1;
        }

        if (predictInstanceIndex >= static_cast<int>(allData.size())) {
            std::cerr << "Error: Predict instance index " << predictInstanceIndex
                      << " is out of range (dataset has " << allData.size() << " instances)\n";
            return 1;
        }

        // Extract the instance to predict
        Point queryPoint = allData[predictInstanceIndex];

        // Create training data (all instances EXCEPT the one to predict)
        std::vector<Point> trainingData;
        trainingData.reserve(allData.size() - 1);

        for (size_t i = 0; i < allData.size(); i++) {
            if (static_cast<int>(i) != predictInstanceIndex) {
                trainingData.push_back(allData[i]);
            }
        }

        if (trainingData.empty()) {
            std::cerr << "Error: No training data available\n";
            return 1;
        }

        // Train KNN on training data (excluding the query instance)
        int dims = trainingData[0].dimensions();
        KNNKDTree knn(k, dims, distMetric, minkowskiP);
        knn.fit(trainingData);

        // Predict
        auto result = knn.predictWithMetrics(queryPoint);

        // Output results as JSON
        std::cout << "{\n";
        std::cout << "  \"predicted_label\": " << result.predicted_label << ",\n";
        std::cout << "  \"distance_calculations\": " << result.distance_calculations << ",\n";
        std::cout << "  \"prediction_time_ms\": " << result.prediction_time_ms << "\n";
        std::cout << "}\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
