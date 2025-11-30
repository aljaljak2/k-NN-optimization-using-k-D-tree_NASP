#include "../../include/knn/knn_kdtree.h"
#include <map>
#include <stdexcept>

KNNKDTree::KNNKDTree(int k_neighbors, int dims)
    : tree(nullptr), k(k_neighbors), dimensions(dims) {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (dims <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }

    tree = new KDTree(dims);
}

KNNKDTree::~KNNKDTree() {
    delete tree;
}

void KNNKDTree::fit(const std::vector<Point>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }

    trainingData = data;

    // Build k-d tree from training data
    for (const auto& point : data) {
        tree->insert(point);
    }
}

std::vector<Point> KNNKDTree::findKNearest(const Point& query) {
    if (trainingData.empty()) {
        throw std::runtime_error("No training data. Call fit() first.");
    }

    // Use k-d tree's k-nearest neighbors search
    return tree->kNearestNeighbors(query, k);
}

int KNNKDTree::predict(const Point& query) {
    auto neighbors = findKNearest(query);

    if (neighbors.empty()) {
        return -1;
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

    return predictedLabel;
}
