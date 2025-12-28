#ifndef KDTREE_H
#define KDTREE_H

#include "kdnode.h"
#include "../utils/point.h"
#include <vector>

/**
 * KDTree - k-dimensional tree implementation
 * Based on: Bentley, J. L. (1975) "Multidimensional binary search trees
 * used for associative searching"
 *
 * Implements algorithms:
 * - INSERT (Algorithm I)
 * - DELETE (Algorithm D)
 * - SEARCH
 * - Nearest neighbor search
 */
class KDTree {
private:
    int k;              // number of dimensions
    KDNode* root;
    mutable int distance_calc_count;  // Track distance calculations

    // Algorithm functions from Bentley 1975
    int nextdisc(int disc);
    std::vector<double> superkey(const Point& point, int j);

    enum SuccessorResult { LOSON, HISON, EQUAL };
    SuccessorResult successor(KDNode* node, const Point& point);

    // Helper functions
    KDNode* findMin(KDNode* node, int dim, int currentDisc);
    KDNode* findMax(KDNode* node, int dim, int currentDisc);
    KDNode* deleteNode(KDNode* node, const Point& point);
    KDNode* searchRec(KDNode* node, const Point& point);
    void inorderRec(KDNode* node);

    // Nearest neighbor search
    double distance(const Point& a, const Point& b);
    void nearestNeighborRec(KDNode* node, const Point& target,
                           Point& best, double& bestDist);

    // k-NN search helper
    struct NeighborCandidate {
        Point point;
        double distance;

        bool operator<(const NeighborCandidate& other) const {
            return distance < other.distance;
        }
    };

    void kNearestRec(KDNode* node, const Point& target,
                    std::vector<NeighborCandidate>& candidates, int k);

public:
    KDTree(int dimensions);
    ~KDTree();

    // Main operations
    bool insert(const Point& point);
    bool search(const Point& point);
    void remove(const Point& point);
    void inorder();

    // Nearest neighbor search
    Point nearestNeighbor(const Point& target);

    // k-NN search - find k nearest neighbors
    std::vector<Point> kNearestNeighbors(const Point& target, int k);

    // Get distance calculations count (for metrics)
    void resetDistanceCount() { distance_calc_count = 0; }
    int getDistanceCount() const { return distance_calc_count; }
};

#endif // KDTREE_H
