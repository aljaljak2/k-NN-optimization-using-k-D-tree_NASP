#ifndef KDNODE_H
#define KDNODE_H

#include "../utils/point.h"

/**
 * KDNode - Node structure for k-d tree
 * Based on: Bentley, J. L. (1975) "Multidimensional binary search trees
 * used for associative searching"
 */
class KDNode {
public:
    Point point;       // k-dimensional point with label
    int disc;          // discriminator (0 to k-1)
    KDNode* loson;     // left subtree (lesser values)
    KDNode* hison;     // right subtree (greater values)

    KDNode(const Point& p, int d);
    ~KDNode();
};

#endif // KDNODE_H
