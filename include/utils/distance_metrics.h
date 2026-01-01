#ifndef DISTANCE_METRICS_H
#define DISTANCE_METRICS_H

#include <vector>
#include <atomic>
#include "point.h"

/**
 * Various distance metrics for k-NN
 * Reference: Uddin et al. (2022) discusses different distance measures
 */

namespace DistanceMetrics {
    // Global counter for distance calculations (thread-safe)
    extern std::atomic<long long> distance_calculation_counter;

    // Counter management
    void resetCounter();
    long long getCounter();

    // Euclidean distance (L2 norm)
    double euclidean(const Point& a, const Point& b);
    double euclidean(const std::vector<double>& a, const std::vector<double>& b);

    // Manhattan distance (L1 norm)
    double manhattan(const Point& a, const Point& b);

    // Chebyshev distance (L-infinity norm)
    double chebyshev(const Point& a, const Point& b);

    // Minkowski distance (generalized)
    double minkowski(const Point& a, const Point& b, double p);

    // Hamming distance (for discrete/binary features)
    double hamming(const Point& a, const Point& b);
}

// Distance metric types
enum class DistanceType {
    EUCLIDEAN,
    MANHATTAN,
    HAMMING,
    MINKOWSKI
};

#endif // DISTANCE_METRICS_H
