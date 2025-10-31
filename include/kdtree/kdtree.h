#ifndef KDTREE_H
#define KDTREE_H

#include "kdnode.h"
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

    // Algorithm functions from Bentley 1975
    int nextdisc(int disc);
    std::vector<double> superkey(const std::vector<double>& point, int j);

    enum SuccessorResult { LOSON, HISON, EQUAL };
    SuccessorResult successor(KDNode* node, const std::vector<double>& point);

    // Helper functions
    KDNode* findMin(KDNode* node, int dim, int currentDisc);
    KDNode* findMax(KDNode* node, int dim, int currentDisc);
    KDNode* deleteNode(KDNode* node, const std::vector<double>& point);
    KDNode* searchRec(KDNode* node, const std::vector<double>& point);
    void inorderRec(KDNode* node);

    // Nearest neighbor search
    double distance(const std::vector<double>& a, const std::vector<double>& b);
    void nearestNeighborRec(KDNode* node, const std::vector<double>& target,
                           std::vector<double>& best, double& bestDist);

public:
    KDTree(int dimensions);
    ~KDTree();

    // Main operations
    bool insert(const std::vector<double>& point);
    bool search(const std::vector<double>& point);
    void remove(const std::vector<double>& point);
    void inorder();

    // k-NN search
    std::vector<double> nearestNeighbor(const std::vector<double>& target);
};

#endif // KDTREE_H
