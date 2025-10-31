#ifndef KNN_BASIC_H
#define KNN_BASIC_H

#include <vector>
#include "../utils/point.h"

/**
 * Classic k-NN implementation (brute force)
 * Baseline for comparison with optimized versions
 *
 * Reference: Uddin et al. (2022) - Classic k-NN variant
 */
class KNNBasic {
private:
    std::vector<Point> trainingData;
    int k;

public:
    KNNBasic(int k_neighbors);

    void fit(const std::vector<Point>& data);
    std::vector<Point> findKNearest(const Point& query);
    int predict(const Point& query);  // For classification
};

#endif // KNN_BASIC_H
