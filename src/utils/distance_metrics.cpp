#include "../../include/utils/distance_metrics.h"
#include <cmath>
#include <algorithm>

namespace DistanceMetrics {

// Initialize the global counter
std::atomic<long long> distance_calculation_counter{0};

void resetCounter() {
    distance_calculation_counter.store(0);
}

long long getCounter() {
    return distance_calculation_counter.load();
}

double euclidean(const Point& a, const Point& b) {
    distance_calculation_counter.fetch_add(1);
    return euclidean(a.coordinates, b.coordinates);
}

double euclidean(const std::vector<double>& a, const std::vector<double>& b) {
    // Note: Don't increment here, already counted in Point version
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double manhattan(const Point& a, const Point& b) {
    double sum = 0;
    for (size_t i = 0; i < a.coordinates.size(); i++) {
        sum += std::abs(a.coordinates[i] - b.coordinates[i]);
    }
    return sum;
}

double chebyshev(const Point& a, const Point& b) {
    double maxDiff = 0;
    for (size_t i = 0; i < a.coordinates.size(); i++) {
        double diff = std::abs(a.coordinates[i] - b.coordinates[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

double minkowski(const Point& a, const Point& b, double p) {
    double sum = 0;
    for (size_t i = 0; i < a.coordinates.size(); i++) {
        sum += std::pow(std::abs(a.coordinates[i] - b.coordinates[i]), p);
    }
    return std::pow(sum, 1.0 / p);
}

double hamming(const Point& a, const Point& b) {
    double count = 0;
    for (size_t i = 0; i < a.coordinates.size(); i++) {
        if (a.coordinates[i] != b.coordinates[i]) {
            count++;
        }
    }
    return count;
}

} // namespace DistanceMetrics
