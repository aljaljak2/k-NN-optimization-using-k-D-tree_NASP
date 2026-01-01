#include "../include/knn_nanoflann.h"
#include <chrono>
#include <map>
#include <cmath>

KNNNanoflann::KNNNanoflann(int k_neighbors, int dims)
    : k(k_neighbors), dimensions(dims), adapter(nullptr), kdtree(nullptr), distance_count(0) {}

KNNNanoflann::~KNNNanoflann() {
    if (kdtree) delete kdtree;
    if (adapter) delete adapter;
}

void KNNNanoflann::fit(const std::vector<Point>& data) {
    trainingData.clear();
    trainingLabels.clear();

    for (const auto& point : data) {
        trainingData.push_back(point.coordinates);
        trainingLabels.push_back(point.label);
    }

    // Build k-d tree
    if (adapter) delete adapter;
    if (kdtree) delete kdtree;

    adapter = new PointCloudAdapter(trainingData);
    kdtree = new KDTreeType(dimensions, *adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree->buildIndex();
}

int KNNNanoflann::predict(const Point& query) {
    if (!kdtree) return -1;

    std::vector<size_t> indices(k);
    std::vector<double> distances(k);

    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(&indices[0], &distances[0]);

    // Use SearchParameters instead of SearchParams (nanoflann API change)
    kdtree->findNeighbors(resultSet, &query.coordinates[0], nanoflann::SearchParameters(10));

    // Approximate distance calculations with dimensionality factor
    // Formula: log(n) * k * dimension_factor * overhead
    // In high dimensions, KD-tree degrades toward O(n) due to curse of dimensionality
    if (trainingData.size() > 0) {
        double log_n = std::log2(trainingData.size());

        // Dimension factor: higher dimensions = more nodes to visit
        // Linear approximation: 1 + (dim / 20) to reflect degradation
        double dim_factor = 1.0 + (dimensions / 20.0);

        // Base overhead for tree traversal
        double overhead = 1.5;

        long long estimated_calcs = static_cast<long long>(log_n * k * dim_factor * overhead);
        distance_count += estimated_calcs;
    }

    // Vote for most common label
    std::map<int, int> votes;
    for (size_t i = 0; i < static_cast<size_t>(k) && i < indices.size(); ++i) {
        votes[trainingLabels[indices[i]]]++;
    }

    int maxVotes = 0;
    int predictedLabel = -1;
    for (const auto& vote : votes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            predictedLabel = vote.first;
        }
    }

    return predictedLabel;
}

KNNNanoflann::PredictionResult KNNNanoflann::predictWithMetrics(const Point& query) {
    auto start = std::chrono::high_resolution_clock::now();

    int predicted = predict(query);

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {predicted, static_cast<int>(distance_count), time_ms};
}

void KNNNanoflann::resetDistanceCount() {
    distance_count = 0;
}

int KNNNanoflann::getDistanceCount() const {
    return static_cast<int>(distance_count);
}
