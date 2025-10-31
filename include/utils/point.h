#ifndef POINT_H
#define POINT_H

#include <vector>
#include <iostream>

/**
 * Point structure for k-dimensional data
 * Supports both features and optional labels for classification
 */
struct Point {
    std::vector<double> coordinates;
    int label;  // For classification tasks (-1 if unlabeled)

    Point() : label(-1) {}
    Point(const std::vector<double>& coords, int lbl = -1)
        : coordinates(coords), label(lbl) {}

    size_t dimensions() const { return coordinates.size(); }

    double& operator[](size_t i) { return coordinates[i]; }
    const double& operator[](size_t i) const { return coordinates[i]; }
};

std::ostream& operator<<(std::ostream& os, const Point& p);

#endif // POINT_H
