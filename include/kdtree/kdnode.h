#ifndef KDNODE_H
#define KDNODE_H

#include <vector>

/**
 * KDNode - Node structure for k-d tree
 * Based on: Bentley, J. L. (1975) "Multidimensional binary search trees
 * used for associative searching"
 */
class KDNode {
public:
    std::vector<double> point;  // k-dimensional point
    int disc;                   // discriminator (0 to k-1)
    KDNode* loson;             // left subtree (lesser values)
    KDNode* hison;             // right subtree (greater values)

    KDNode(const std::vector<double>& p, int d);
    ~KDNode();
};

#endif // KDNODE_H
