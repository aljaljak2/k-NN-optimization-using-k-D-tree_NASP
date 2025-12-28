#include "../../include/knn/knn_basic.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <chrono>

KNNBasic::KNNBasic(int k_neighbors, DistanceType metric, double p)
    : k(k_neighbors), distanceMetric(metric), minkowskiP(p) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
}

void KNNBasic::fit(const std::vector<Point>& data) {
    trainingData = data;
}

double KNNBasic::calculateDistance(const Point& a, const Point& b) const {
    switch (distanceMetric) {
        case DistanceType::EUCLIDEAN:
            return DistanceMetrics::euclidean(a, b);
        case DistanceType::MANHATTAN:
            return DistanceMetrics::manhattan(a, b);
        case DistanceType::HAMMING:
            return DistanceMetrics::hamming(a, b);
        case DistanceType::MINKOWSKI:
            return DistanceMetrics::minkowski(a, b, minkowskiP);
        default:
            return DistanceMetrics::euclidean(a, b);
    }
}

std::vector<Point> KNNBasic::findKNearest(const Point& query) {
    if (trainingData.empty()) {
        throw std::runtime_error("No training data. Call fit() first.");
    }

    // Calculate distances for all training points
    std::vector<std::pair<double, Point>> distances;
    distances.reserve(trainingData.size());

    for (const auto& point : trainingData) {
        double dist = calculateDistance(query, point);
        distances.push_back({dist, point});
    }

    // Sort by distance (brute force approach)
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Get k nearest neighbors
    std::vector<Point> neighbors;
    int limit = std::min(k, static_cast<int>(distances.size()));
    neighbors.reserve(limit);

    for (int i = 0; i < limit; i++) {
        neighbors.push_back(distances[i].second);
    }

    return neighbors;
}

int KNNBasic::predict(const Point& query) {
    auto neighbors = findKNearest(query);

    // Count votes for each label
    std::map<int, int> votes;
    for (const auto& neighbor : neighbors) {
        votes[neighbor.label]++;
    }

    // Find label with most votes
    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& [label, count] : votes) {
        if (count > maxVotes) {
            maxVotes = count;
            predictedLabel = label;
        }
    }

    return predictedLabel;
}

KNNBasic::PredictionResult KNNBasic::predictWithMetrics(const Point& query) {
    auto start = std::chrono::high_resolution_clock::now();

    if (trainingData.empty()) {
        throw std::runtime_error("No training data. Call fit() first.");
    }

    // Calculate distances for all training points
    std::vector<std::pair<double, Point>> distances;
    distances.reserve(trainingData.size());

    int distance_calculations = 0;
    for (const auto& point : trainingData) {
        double dist = calculateDistance(query, point);
        distances.push_back({dist, point});
        distance_calculations++;
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Get k nearest neighbors
    std::vector<Point> neighbors;
    int limit = std::min(k, static_cast<int>(distances.size()));
    neighbors.reserve(limit);

    for (int i = 0; i < limit; i++) {
        neighbors.push_back(distances[i].second);
    }

    // Count votes for each label
    std::map<int, int> votes;
    for (const auto& neighbor : neighbors) {
        votes[neighbor.label]++;
    }

    // Find label with most votes
    int predictedLabel = -1;
    int maxVotes = 0;
    for (const auto& [label, count] : votes) {
        if (count > maxVotes) {
            maxVotes = count;
            predictedLabel = label;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0;

    return {predictedLabel, distance_calculations, time_ms};
}
