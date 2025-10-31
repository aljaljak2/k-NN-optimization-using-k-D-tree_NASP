#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <vector>
#include <string>
#include "point.h"

/**
 * Dataset loading utilities
 * Supports various formats for benchmark datasets
 */
class DatasetLoader {
public:
    // Load CSV format datasets
    static std::vector<Point> loadCSV(const std::string& filepath, bool hasHeader = true);

    // Generate synthetic datasets for testing
    static std::vector<Point> generateRandom(int numPoints, int dimensions, int seed = 42);
    static std::vector<Point> generateClustered(int numClusters, int pointsPerCluster,
                                                 int dimensions, int seed = 42);

    // Split dataset into train/test
    static void trainTestSplit(const std::vector<Point>& data,
                               std::vector<Point>& train,
                               std::vector<Point>& test,
                               double testRatio = 0.2,
                               int seed = 42);
};

#endif // DATASET_LOADER_H
