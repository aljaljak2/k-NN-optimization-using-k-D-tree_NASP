#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <vector>
#include <string>
#include <map>
#include "point.h"

/**
 * Dataset loading utilities
 * Supports various formats for benchmark datasets
 */
class DatasetLoader {
public:
    // Load CSV format datasets (numeric only)
    // labelColumn: index of the column containing the label (-1 means last column)
    static std::vector<Point> loadCSV(const std::string& filepath,
                                      bool hasHeader = true,
                                      int labelColumn = -1);

    // Load CSV with automatic one-hot encoding for categorical columns
    // categoricalColumns: indices of columns to one-hot encode (empty = auto-detect)
    // labelColumn: index of the column containing the label (-1 means last column)
    static std::vector<Point> loadCSVWithEncoding(const std::string& filepath,
                                                   bool hasHeader = true,
                                                   const std::vector<int>& categoricalColumns = {},
                                                   int labelColumn = -1);

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

private:
    // Helper: Check if string is numeric
    static bool isNumeric(const std::string& str);

    // Helper: Detect categorical columns automatically
    static std::vector<int> detectCategoricalColumns(const std::string& filepath,
                                                      bool hasHeader,
                                                      int labelColumn = -1);
};

#endif // DATASET_LOADER_H
