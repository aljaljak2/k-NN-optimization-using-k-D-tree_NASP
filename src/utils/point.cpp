#include "../../include/utils/point.h"

std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << "(";
    for (size_t i = 0; i < p.coordinates.size(); i++) {
        os << p.coordinates[i];
        if (i < p.coordinates.size() - 1) os << ",";
    }
    if (p.label != -1) {
        os << " | label=" << p.label;
    }
    os << ")";
    return os;
}
