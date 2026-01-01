#ifndef KNN_NANOFLANN_H
#define KNN_NANOFLANN_H

#include <vector>
#include "../../include/utils/point.h"

// Include nanoflann (header-only library)
// Download from: https://github.com/jlblancoc/nanoflann
// Place nanoflann.hpp in this directory or add to include path
#include "nanoflann.hpp"

/**
 * k-NN implementation using nanoflann library
 * Wrapper to match the interface of KNNBasic and KNNKDTree
 */
class KNNNanoflann {
private:
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    int k;
    int dimensions;

    // Adapter for nanoflann
    struct PointCloudAdapter {
        const std::vector<std::vector<double>>& points;
        PointCloudAdapter(const std::vector<std::vector<double>>& pts) : points(pts) {}
        inline size_t kdtree_get_point_count() const { return points.size(); }
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx][dim];
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
    };

    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloudAdapter>,
        PointCloudAdapter,
        -1
    > KDTreeType;

    PointCloudAdapter* adapter;
    KDTreeType* kdtree;
    mutable long long distance_count;  // Manual counter for nanoflann

public:
    KNNNanoflann(int k_neighbors, int dims);
    ~KNNNanoflann();

    void fit(const std::vector<Point>& data);
    int predict(const Point& query);

    struct PredictionResult {
        int predicted_label;
        int distance_calculations;
        double prediction_time_ms;
    };

    PredictionResult predictWithMetrics(const Point& query);

    // Distance calculation counter methods
    void resetDistanceCount();
    int getDistanceCount() const;
};

#endif // KNN_NANOFLANN_H
