#include <iostream>
#include <cassert>
#include <cmath>
#include "../include/kdtree/kdtree.h"
#include "../include/utils/point.h"

void testInsertAndSearch() {
    std::cout << "\n=== Test 1: Insert and Search ===" << std::endl;

    KDTree tree(2);

    // Insert some 2D points
    Point p1({3.0, 6.0}, 0);
    Point p2({17.0, 15.0}, 1);
    Point p3({13.0, 15.0}, 1);
    Point p4({6.0, 12.0}, 0);
    Point p5({9.0, 1.0}, 1);
    Point p6({2.0, 7.0}, 0);
    Point p7({10.0, 19.0}, 1);

    // Test insertion
    assert(tree.insert(p1) == true);
    assert(tree.insert(p2) == true);
    assert(tree.insert(p3) == true);
    assert(tree.insert(p4) == true);
    assert(tree.insert(p5) == true);
    assert(tree.insert(p6) == true);
    assert(tree.insert(p7) == true);

    std::cout << " Inserted 7 points successfully" << std::endl;

    // Test duplicate insertion (should fail)
    assert(tree.insert(p1) == false);
    std::cout << " Duplicate insertion correctly rejected" << std::endl;

    // Test search - points that exist
    assert(tree.search(p1) == true);
    assert(tree.search(p2) == true);
    assert(tree.search(p5) == true);
    std::cout << " Search found all inserted points" << std::endl;

    // Test search - point that doesn't exist
    Point p_not_exist({100.0, 100.0}, 0);
    assert(tree.search(p_not_exist) == false);
    std::cout << " Search correctly returned false for non-existent point" << std::endl;

    std::cout << "\nTree structure (inorder):" << std::endl;
    tree.inorder();
}

void testDelete() {
    std::cout << "\n=== Test 2: Delete Operation ===" << std::endl;

    KDTree tree(2);

    // Build a tree
    Point p1({7.0, 2.0}, 0);
    Point p2({5.0, 4.0}, 1);
    Point p3({9.0, 6.0}, 1);
    Point p4({2.0, 3.0}, 0);
    Point p5({4.0, 7.0}, 1);
    Point p6({8.0, 1.0}, 0);

    tree.insert(p1);
    tree.insert(p2);
    tree.insert(p3);
    tree.insert(p4);
    tree.insert(p5);
    tree.insert(p6);

    std::cout << "Inserted 6 points" << std::endl;

    // Verify all points exist
    assert(tree.search(p1) == true);
    assert(tree.search(p2) == true);
    assert(tree.search(p3) == true);

    // Delete a leaf node
    tree.remove(p4);
    assert(tree.search(p4) == false);
    assert(tree.search(p1) == true);  // Others still exist
    std::cout << " Deleted leaf node successfully" << std::endl;

    // Delete an internal node
    tree.remove(p2);
    assert(tree.search(p2) == false);
    assert(tree.search(p1) == true);  // Others still exist
    assert(tree.search(p3) == true);
    std::cout << " Deleted internal node successfully" << std::endl;

    std::cout << "\nTree after deletions:" << std::endl;
    tree.inorder();
}

void testNearestNeighbor() {
    std::cout << "\n=== Test 3: Nearest Neighbor Search ===" << std::endl;

    KDTree tree(2);

    // Create a simple dataset
    Point p1({1.0, 2.0}, 0);
    Point p2({3.0, 5.0}, 1);
    Point p3({8.0, 7.0}, 1);
    Point p4({10.0, 2.0}, 0);
    Point p5({5.0, 4.0}, 1);

    tree.insert(p1);
    tree.insert(p2);
    tree.insert(p3);
    tree.insert(p4);
    tree.insert(p5);

    // Test 1: Query point close to p1
    Point query1({2.0, 3.0}, -1);
    Point nearest1 = tree.nearestNeighbor(query1);

    // Calculate distances manually
    auto dist = [](const Point& a, const Point& b) {
        double sum = 0;
        for (size_t i = 0; i < a.dimensions(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    };

    double d1 = dist(query1, p1);
    double d_nearest1 = dist(query1, nearest1);

    assert(std::abs(d1 - d_nearest1) < 1e-6);  // Should be p1
    std::cout << " Found correct nearest neighbor for query (2,3): "
              << "(" << nearest1[0] << "," << nearest1[1] << ")" << std::endl;

    // Test 2: Query point close to p4
    Point query2({9.0, 1.0}, -1);
    Point nearest2 = tree.nearestNeighbor(query2);

    double d4 = dist(query2, p4);
    double d_nearest2 = dist(query2, nearest2);

    assert(std::abs(d4 - d_nearest2) < 1e-6);  // Should be p4
    std::cout << " Found correct nearest neighbor for query (9,1): "
              << "(" << nearest2[0] << "," << nearest2[1] << ")" << std::endl;
}

void testKNearestNeighbors() {
    std::cout << "\n=== Test 4: k-Nearest Neighbors Search ===" << std::endl;

    KDTree tree(2);

    // Create a dataset
    Point p1({1.0, 3.0}, 0);
    Point p2({2.0, 2.0}, 0);
    Point p3({5.0, 4.0}, 1);
    Point p4({6.0, 1.0}, 1);
    Point p5({7.0, 5.0}, 1);
    Point p6({8.0, 3.0}, 1);

    tree.insert(p1);
    tree.insert(p2);
    tree.insert(p3);
    tree.insert(p4);
    tree.insert(p5);
    tree.insert(p6);

    // Test k=3 nearest neighbors
    Point query({4.0, 3.0}, -1);
    std::vector<Point> neighbors = tree.kNearestNeighbors(query, 3);

    assert(neighbors.size() == 3);
    std::cout << " Found k=3 neighbors for query (4,3):" << std::endl;

    for (size_t i = 0; i < neighbors.size(); i++) {
        std::cout << "  Neighbor " << (i+1) << ": ("
                  << neighbors[i][0] << "," << neighbors[i][1] << ")"
                  << " label=" << neighbors[i].label << std::endl;
    }

    // Test k > number of points
    std::vector<Point> all_neighbors = tree.kNearestNeighbors(query, 10);
    assert(all_neighbors.size() == 6);  // Should return all 6 points
    std::cout << " Correctly returned all points when k > tree size" << std::endl;
}

void test3DTree() {
    std::cout << "\n=== Test 5: 3D Tree Operations ===" << std::endl;

    KDTree tree(3);

    // Insert 3D points
    Point p1({1.0, 2.0, 3.0}, 0);
    Point p2({4.0, 5.0, 6.0}, 1);
    Point p3({7.0, 8.0, 9.0}, 1);
    Point p4({2.0, 3.0, 1.0}, 0);

    assert(tree.insert(p1) == true);
    assert(tree.insert(p2) == true);
    assert(tree.insert(p3) == true);
    assert(tree.insert(p4) == true);

    std::cout << " Inserted 4 points into 3D tree" << std::endl;

    // Test search
    assert(tree.search(p1) == true);
    assert(tree.search(p3) == true);
    std::cout << " Search works in 3D tree" << std::endl;

    // Test nearest neighbor
    Point query({3.0, 4.0, 5.0}, -1);
    Point nearest = tree.nearestNeighbor(query);
    std::cout << " Nearest neighbor to (3,4,5): ("
              << nearest[0] << "," << nearest[1] << "," << nearest[2] << ")" << std::endl;

    // Test deletion
    tree.remove(p2);
    assert(tree.search(p2) == false);
    assert(tree.search(p1) == true);
    std::cout << " Deletion works in 3D tree" << std::endl;
}

void testEdgeCases() {
    std::cout << "\n=== Test 6: Edge Cases ===" << std::endl;

    // Test empty tree
    KDTree tree1(2);
    Point query({1.0, 2.0}, 0);

    assert(tree1.search(query) == false);
    std::cout << " Search on empty tree returns false" << std::endl;

    std::vector<Point> empty_neighbors = tree1.kNearestNeighbors(query, 5);
    assert(empty_neighbors.empty());
    std::cout << " k-NN on empty tree returns empty vector" << std::endl;

    // Test single point tree
    KDTree tree2(2);
    Point p({5.0, 5.0}, 1);
    tree2.insert(p);

    Point nearest = tree2.nearestNeighbor(query);
    assert(nearest.coordinates == p.coordinates);
    std::cout << " Nearest neighbor in single-point tree works" << std::endl;

    std::vector<Point> one_neighbor = tree2.kNearestNeighbors(query, 1);
    assert(one_neighbor.size() == 1);
    assert(one_neighbor[0].coordinates == p.coordinates);
    std::cout << " k-NN in single-point tree works" << std::endl;

    // Test invalid k
    std::vector<Point> invalid_k = tree2.kNearestNeighbors(query, 0);
    assert(invalid_k.empty());
    std::cout << " k-NN with k=0 returns empty vector" << std::endl;

    // Test wrong dimension
    Point wrong_dim({1.0, 2.0, 3.0}, 0);
    assert(tree2.insert(wrong_dim) == false);
    std::cout << " Insert with wrong dimension correctly rejected" << std::endl;
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   KD-TREE COMPREHENSIVE TEST SUITE    " << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testInsertAndSearch();
        testDelete();
        testNearestNeighbor();
        testKNearestNeighbors();
        test3DTree();
        testEdgeCases();

        std::cout << "\n========================================" << std::endl;
        std::cout << "    ALL TESTS PASSED SUCCESSFULLY!    Q" << std::endl;
        std::cout << "========================================\n" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\nTEST FAILED: Unknown error" << std::endl;
        return 1;
    }
}
