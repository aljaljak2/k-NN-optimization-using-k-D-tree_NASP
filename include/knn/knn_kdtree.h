#ifndef KNN_KDTREE_H
#define KNN_KDTREE_H

#include <vector>
#include "../kdtree/kdtree.h"
#include "../utils/point.h"

/**
 * k-NN implementation using k-d tree optimization
 * Based on: Bentley (1975) k-d tree structure for efficient nearest neighbor search
 *
 * This combines the k-d tree spatial indexing with k-NN algorithm
 */
class KNNKDTree {
private:
    KDTree* tree;
    std::vector<Point> trainingData;
    int k;
    int dimensions;

public:
    KNNKDTree(int k_neighbors, int dims);
    ~KNNKDTree();

    void fit(const std::vector<Point>& data);
    std::vector<Point> findKNearest(const Point& query);
    int predict(const Point& query);  // For classification

    // New: Single instance prediction with metrics
    struct PredictionResult {
        int predicted_label;
        int distance_calculations;
        double prediction_time_ms;
    };

    PredictionResult predictWithMetrics(const Point& query);
};

#endif // KNN_KDTREE_H
