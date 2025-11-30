#include "../../include/kdtree/kdnode.h"

KDNode::KDNode(const Point& p, int d)
    : point(p), disc(d), loson(nullptr), hison(nullptr) {
}

KDNode::~KDNode() {
    delete loson;
    delete hison;
}
