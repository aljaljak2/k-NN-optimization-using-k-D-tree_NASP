#include "../../include/kdtree/kdnode.h"

KDNode::KDNode(const std::vector<double>& p, int d)
    : point(p), disc(d), loson(nullptr), hison(nullptr) {}

KDNode::~KDNode() {
    delete loson;
    delete hison;
}
