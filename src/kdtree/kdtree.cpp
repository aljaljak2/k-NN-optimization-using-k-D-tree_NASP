#include "../../include/kdtree/kdtree.h"
#include <iostream>
#include <cmath>
#include <algorithm>

KDTree::KDTree(int dimensions, DistanceType metric, double p)
    : k(dimensions), root(nullptr), distance_calc_count(0),
      distanceMetric(metric), minkowskiP(p) {
}

KDTree::~KDTree() {
    delete root;
}

// NEXTDISC function from Bentley 1975
int KDTree::nextdisc(int disc) {
    return (disc + 1) % k;
}

// Creates superkey for comparison (cyclic concatenation)
std::vector<double> KDTree::superkey(const Point& point, int j) {
    std::vector<double> sk;
    // Cyclic concatenation: Kj, Kj+1, ..., Kk-1, K0, ..., Kj-1
    for (int i = j; i < k; i++) {
        sk.push_back(point[i]);
    }
    for (int i = 0; i < j; i++) {
        sk.push_back(point[i]);
    }
    return sk;
}

// SUCCESSOR function from Bentley 1975
KDTree::SuccessorResult KDTree::successor(KDNode* node, const Point& point) {
    int j = node->disc;

    if (point[j] < node->point[j]) {
        return LOSON;
    } else if (point[j] > node->point[j]) {
        return HISON;
    } else {
        // If Kj are equal, compare superkeys
        std::vector<double> s_point = superkey(point, j);
        std::vector<double> s_node = superkey(node->point, j);

        if (s_point < s_node) {
            return LOSON;
        } else if (s_point > s_node) {
            return HISON;
        } else {
            return EQUAL;
        }
    }
}

// Algorithm INSERT from Bentley 1975
bool KDTree::insert(const Point& point) {
    if (point.dimensions() != static_cast<size_t>(k)) {
        std::cerr << "Point dimension does not match!" << std::endl;
        return false;
    }

    // I1: Check if tree is empty
    if (root == nullptr) {
        root = new KDNode(point, 0);
        return true;
    }

    KDNode* Q = root;

    while (true) {
        // I2: Compare
        if (Q->point.coordinates == point.coordinates) {
            return false;  // Point already exists
        }

        SuccessorResult succ = successor(Q, point);

        if (succ == EQUAL) {
            return false;  // All keys equal
        }

        // Determine which son
        KDNode** nextSon = (succ == LOSON) ? &(Q->loson) : &(Q->hison);

        if (*nextSon == nullptr) {
            // I4: Insert new node into tree
            *nextSon = new KDNode(point, nextdisc(Q->disc));
            return true;
        }

        // I3: Move down
        Q = *nextSon;
    }
}

// Recursive search for deletion
KDNode* KDTree::findMin(KDNode* node, int dim, int currentDisc) {
    if (node == nullptr) return nullptr;

    int cd = currentDisc;

    if (cd == dim) {
        if (node->loson == nullptr) {
            return node;
        }
        return findMin(node->loson, dim, nextdisc(cd));
    }

    KDNode* left = findMin(node->loson, dim, nextdisc(cd));
    KDNode* right = findMin(node->hison, dim, nextdisc(cd));

    KDNode* minNode = node;
    if (left != nullptr && left->point[dim] < minNode->point[dim]) {
        minNode = left;
    }
    if (right != nullptr && right->point[dim] < minNode->point[dim]) {
        minNode = right;
    }

    return minNode;
}

KDNode* KDTree::findMax(KDNode* node, int dim, int currentDisc) {
    if (node == nullptr) return nullptr;

    int cd = currentDisc;

    if (cd == dim) {
        if (node->hison == nullptr) {
            return node;
        }
        return findMax(node->hison, dim, nextdisc(cd));
    }

    KDNode* left = findMax(node->loson, dim, nextdisc(cd));
    KDNode* right = findMax(node->hison, dim, nextdisc(cd));

    KDNode* maxNode = node;
    if (left != nullptr && left->point[dim] > maxNode->point[dim]) {
        maxNode = left;
    }
    if (right != nullptr && right->point[dim] > maxNode->point[dim]) {
        maxNode = right;
    }

    return maxNode;
}

// DELETE algorithm from Bentley 1975
KDNode* KDTree::deleteNode(KDNode* node, const Point& point) {
    if (node == nullptr) return nullptr;

    int j = node->disc;

    // If we found the node to delete
    if (node->point.coordinates == point.coordinates) {
        // D1: Is P a leaf?
        if (node->hison == nullptr && node->loson == nullptr) {
            delete node;
            return nullptr;
        }

        // D2: Decide where to get P's successor
        KDNode* replacement;
        if (node->hison != nullptr) {
            // D3: Get next root from HISON(P)
            replacement = findMin(node->hison, j, nextdisc(j));
            node->point = replacement->point;
            node->hison = deleteNode(node->hison, replacement->point);
        } else {
            // D4: Get next root from LOSON(P)
            replacement = findMax(node->loson, j, nextdisc(j));
            node->point = replacement->point;
            node->hison = deleteNode(node->loson, replacement->point);
            node->loson = nullptr;
        }

        return node;
    }

    // Continue search
    SuccessorResult succ = successor(node, point);
    if (succ == LOSON) {
        node->loson = deleteNode(node->loson, point);
    } else if (succ == HISON) {
        node->hison = deleteNode(node->hison, point);
    }

    return node;
}

// Point search
KDNode* KDTree::searchRec(KDNode* node, const Point& point) {
    if (node == nullptr) return nullptr;

    if (node->point.coordinates == point.coordinates) {
        return node;
    }

    SuccessorResult succ = successor(node, point);
    if (succ == LOSON) {
        return searchRec(node->loson, point);
    } else if (succ == HISON) {
        return searchRec(node->hison, point);
    } else {
        return node;  // EQUAL
    }
}

bool KDTree::search(const Point& point) {
    return searchRec(root, point) != nullptr;
}

void KDTree::remove(const Point& point) {
    root = deleteNode(root, point);
}

// In-order traversal
void KDTree::inorderRec(KDNode* node) {
    if (node != nullptr) {
        inorderRec(node->loson);
        std::cout << "(";
        for (size_t i = 0; i < node->point.dimensions(); i++) {
            std::cout << node->point[i];
            if (i < node->point.dimensions() - 1) std::cout << ",";
        }
        std::cout << ") label=" << node->point.label
                  << " disc=" << node->disc << std::endl;
        inorderRec(node->hison);
    }
}

void KDTree::inorder() {
    inorderRec(root);
}

// Distance calculation (supports multiple metrics)
double KDTree::distance(const Point& a, const Point& b) {
    distance_calc_count++;  // Track distance calculations

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

// Nearest neighbor search (recursive)
void KDTree::nearestNeighborRec(KDNode* node, const Point& target,
                                Point& best, double& bestDist) {
    if (node == nullptr) return;

    double d = distance(target, node->point);
    if (d < bestDist) {
        bestDist = d;
        best = node->point;
    }

    int j = node->disc;
    double diff = target[j] - node->point[j];

    KDNode* near = (diff < 0) ? node->loson : node->hison;
    KDNode* far = (diff < 0) ? node->hison : node->loson;

    nearestNeighborRec(near, target, best, bestDist);

    if (std::abs(diff) < bestDist) {
        nearestNeighborRec(far, target, best, bestDist);
    }
}

Point KDTree::nearestNeighbor(const Point& target) {
    if (root == nullptr) {
        return Point();  // Return empty point
    }

    Point best = root->point;
    double bestDist = distance(target, root->point);

    nearestNeighborRec(root, target, best, bestDist);

    return best;
}

// k-NN search - recursive helper
void KDTree::kNearestRec(KDNode* node, const Point& target,
                        std::vector<NeighborCandidate>& candidates, int k) {
    if (node == nullptr) return;

    // Calculate distance to current node
    double dist = distance(target, node->point);

    // Add to candidates if we have less than k, or if this is closer than the worst candidate
    if (candidates.size() < static_cast<size_t>(k)) {
        candidates.push_back({node->point, dist});
        std::sort(candidates.begin(), candidates.end());
    } else if (dist < candidates.back().distance) {
        // Replace worst candidate
        candidates.back() = {node->point, dist};
        std::sort(candidates.begin(), candidates.end());
    }

    // Determine which subtree to search first
    int j = node->disc;
    double diff = target[j] - node->point[j];

    KDNode* near = (diff < 0) ? node->loson : node->hison;
    KDNode* far = (diff < 0) ? node->hison : node->loson;

    // Search near subtree first
    kNearestRec(near, target, candidates, k);

    // Check if we need to search far subtree
    // If we don't have k neighbors yet, or if the splitting plane is close enough
    if (candidates.size() < static_cast<size_t>(k) ||
        std::abs(diff) < candidates.back().distance) {
        kNearestRec(far, target, candidates, k);
    }
}

// k-NN search - public interface
std::vector<Point> KDTree::kNearestNeighbors(const Point& target, int k) {
    if (root == nullptr || k <= 0) {
        return {};
    }

    std::vector<NeighborCandidate> candidates;
    kNearestRec(root, target, candidates, k);

    // Extract points from candidates
    std::vector<Point> result;
    result.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        result.push_back(candidate.point);
    }

    return result;
}
