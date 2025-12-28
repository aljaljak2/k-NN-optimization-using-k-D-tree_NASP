#include "../../include/utils/dataset_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <map>
#include <set>
#include <cctype>

std::vector<Point> DatasetLoader::loadCSV(const std::string& filepath, bool hasHeader) {
    std::vector<Point> data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        // Skip header if present
        if (firstLine && hasHeader) {
            firstLine = false;
            continue;
        }

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> coords;
        int label = -1;

        while (std::getline(ss, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

            if (cell.empty()) {
                continue;
            }

            try {
                double value = std::stod(cell);
                coords.push_back(value);
            } catch (...) {
                // If conversion fails, skip this cell
                continue;
            }
        }

        if (!coords.empty()) {
            // Last column is the label
            label = static_cast<int>(coords.back());
            coords.pop_back();

            // Create point with remaining coordinates
            if (!coords.empty()) {
                data.emplace_back(coords, label);
            }
        }

        firstLine = false;
    }

    file.close();

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filepath);
    }

    return data;
}

void DatasetLoader::trainTestSplit(const std::vector<Point>& data,
                                   std::vector<Point>& train,
                                   std::vector<Point>& test,
                                   double testRatio,
                                   int seed) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot split empty dataset");
    }

    if (testRatio <= 0.0 || testRatio >= 1.0) {
        throw std::invalid_argument("testRatio must be between 0 and 1");
    }

    // Create shuffled indices
    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        indices[i] = i;
    }

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Calculate split point
    size_t testSize = static_cast<size_t>(data.size() * testRatio);
    size_t trainSize = data.size() - testSize;

    // Clear output vectors
    train.clear();
    test.clear();
    train.reserve(trainSize);
    test.reserve(testSize);

    // Split data
    for (size_t i = 0; i < trainSize; i++) {
        train.push_back(data[indices[i]]);
    }

    for (size_t i = trainSize; i < data.size(); i++) {
        test.push_back(data[indices[i]]);
    }
}

std::vector<Point> DatasetLoader::generateRandom(int numPoints, int dimensions, int seed) {
    std::vector<Point> data;
    data.reserve(numPoints);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::uniform_int_distribution<int> labelDist(0, 2);

    for (int i = 0; i < numPoints; i++) {
        std::vector<double> coords(dimensions);
        for (int j = 0; j < dimensions; j++) {
            coords[j] = dist(rng);
        }
        data.emplace_back(coords, labelDist(rng));
    }

    return data;
}

std::vector<Point> DatasetLoader::generateClustered(int numClusters, int pointsPerCluster,
                                                     int dimensions, int seed) {
    std::vector<Point> data;
    data.reserve(numClusters * pointsPerCluster);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> centerDist(0.0, 100.0);
    std::normal_distribution<double> pointDist(0.0, 5.0);

    // Generate cluster centers
    for (int cluster = 0; cluster < numClusters; cluster++) {
        std::vector<double> center(dimensions);
        for (int d = 0; d < dimensions; d++) {
            center[d] = centerDist(rng);
        }

        // Generate points around this center
        for (int p = 0; p < pointsPerCluster; p++) {
            std::vector<double> coords(dimensions);
            for (int d = 0; d < dimensions; d++) {
                coords[d] = center[d] + pointDist(rng);
            }
            data.emplace_back(coords, cluster);
        }
    }

    return data;
}

// Helper: Check if string is numeric
bool DatasetLoader::isNumeric(const std::string& str) {
    if (str.empty()) return false;

    size_t start = 0;
    // Handle negative numbers
    if (str[0] == '-' || str[0] == '+') {
        if (str.length() == 1) return false;
        start = 1;
    }

    bool hasDecimal = false;
    bool hasDigit = false;

    for (size_t i = start; i < str.length(); i++) {
        if (std::isdigit(str[i])) {
            hasDigit = true;
        } else if (str[i] == '.' && !hasDecimal) {
            hasDecimal = true;
        } else {
            return false;
        }
    }

    return hasDigit;
}

// Helper: Detect categorical columns automatically
std::vector<int> DatasetLoader::detectCategoricalColumns(const std::string& filepath, bool hasHeader) {
    std::vector<int> categoricalCols;
    std::ifstream file(filepath);
    std::cout << "Detecting categorical columns in file: " << filepath << std::endl;
    if (!file.is_open()) {
        return categoricalCols;
    }

    std::cout << "Analyzing data types of columns..." << std::endl;
    std::string line;
    bool firstDataLine = true;  // Track first DATA line (not header)
    std::vector<bool> isNumericColumn;
    std::map<int, std::set<std::string>> uniqueValues;  // Track unique values per column
    int rowCount = 0;  // Track rows checked (NOT static!)

    // Skip header if present
    if (hasHeader) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        size_t colIndex = 0;

        while (std::getline(ss, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

            if (firstDataLine) {
                // Initialize on first data line
                isNumericColumn.push_back(isNumeric(cell));
            } else {
                // Update: if any cell is non-numeric, mark column as categorical
                if (colIndex < isNumericColumn.size() && !isNumeric(cell)) {
                    isNumericColumn[colIndex] = false;
                }
            }

            // Track unique values for categorical columns
            if (colIndex < isNumericColumn.size() && !isNumericColumn[colIndex]) {
                uniqueValues[static_cast<int>(colIndex)].insert(cell);
            }

            colIndex++;
        }

        firstDataLine = false;

        // Only check first 100 rows for efficiency
        if (++rowCount > 100) break;
    }

    file.close();

    // Collect categorical column indices (exclude last column - label)
    // Skip columns with too many unique values (likely IDs or other non-categorical data)
    const int MAX_CATEGORIES = 50;  // Maximum number of categories to one-hot encode

    for (size_t i = 0; i < isNumericColumn.size() - 1; i++) {
        if (!isNumericColumn[i]) {
            int numUnique = uniqueValues[i].size();
            if (numUnique <= MAX_CATEGORIES) {
                categoricalCols.push_back(static_cast<int>(i));
            } else {
                std::cout << "Skipping column " << i << " with " << numUnique
                         << " unique values (exceeds max " << MAX_CATEGORIES << ")" << std::endl;
            }
        }
    }
    std::cout << "Detected " << categoricalCols.size() << " categorical columns" << std::endl;
    return categoricalCols;
}

// Load CSV with one-hot encoding
std::vector<Point> DatasetLoader::loadCSVWithEncoding(const std::string& filepath,
                                                       bool hasHeader,
                                                       const std::vector<int>& categoricalColumns) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Auto-detect categorical columns if not specified
    std::vector<int> catCols = categoricalColumns;
    if (catCols.empty()) {
        std::cout << "Auto-detecting categorical columns..." << std::endl;
        catCols = detectCategoricalColumns(filepath, hasHeader);
        std::cout << "Found " << catCols.size() << " categorical columns to encode" << std::endl;
    }

    std::set<int> catColSet(catCols.begin(), catCols.end());

    // First pass: collect all unique values for categorical columns
    std::map<int, std::set<std::string>> categoryValues;
    std::map<int, std::map<std::string, int>> categoryEncoding;

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine && hasHeader) {
            firstLine = false;
            continue;
        }

        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        int colIndex = 0;

        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

            if (catColSet.count(colIndex)) {
                categoryValues[colIndex].insert(cell);
            }
            colIndex++;
        }
        firstLine = false;
    }

    // Create encoding maps
    int totalEncodedDimensions = 0;
    for (const auto& [colIdx, values] : categoryValues) {
        int encoding = 0;
        std::cout << "Column " << colIdx << ": " << values.size() << " categories" << std::endl;
        totalEncodedDimensions += values.size();
        for (const auto& value : values) {
            categoryEncoding[colIdx][value] = encoding++;
        }
    }
    std::cout << "Total dimensions from categorical encoding: " << totalEncodedDimensions << std::endl;

    // Second pass: load data with one-hot encoding
    file.clear();
    file.seekg(0);

    std::vector<Point> data;
    firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine && hasHeader) {
            firstLine = false;
            continue;
        }

        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> coords;
        int label = -1;
        std::vector<std::string> cells;

        // Parse all cells
        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            cells.push_back(cell);
        }

        if (cells.empty()) continue;

        // Process all columns except last (label)
        for (size_t i = 0; i < cells.size() - 1; i++) {
            if (catColSet.count(i)) {
                // One-hot encode categorical column
                int numCategories = categoryValues[i].size();
                int encodedValue = categoryEncoding[i][cells[i]];

                for (int j = 0; j < numCategories; j++) {
                    coords.push_back(j == encodedValue ? 1.0 : 0.0);
                }
            } else {
                // Numeric column
                try {
                    coords.push_back(std::stod(cells[i]));
                } catch (...) {
                    coords.push_back(0.0);
                }
            }
        }

        // Last column is label
        if (!cells.empty()) {
            try {
                label = static_cast<int>(std::stod(cells.back()));
            } catch (...) {
                // If label is categorical, encode it
                if (categoryEncoding.count(cells.size() - 1)) {
                    label = categoryEncoding[cells.size() - 1][cells.back()];
                }
            }
        }

        if (!coords.empty()) {
            data.emplace_back(coords, label);
        }

        firstLine = false;
    }

    file.close();

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filepath);
    }

    return data;
}
