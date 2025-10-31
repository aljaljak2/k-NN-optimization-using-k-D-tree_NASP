#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

class KDNode {
public:
    std::vector<double> point;  // k-dimenzionalna tačka
    int disc;                   // diskriminator (0 do k-1)
    KDNode* loson;             // lijevo podstablo (manje vrijednosti)
    KDNode* hison;             // desno podstablo (veće vrijednosti)
    
    KDNode(const std::vector<double>& p, int d) 
        : point(p), disc(d), loson(nullptr), hison(nullptr) {}
    
    ~KDNode() {
        delete loson;
        delete hison;
    }
};

class KDTree {
private:
    int k;              // broj dimenzija
    KDNode* root;
    
    // NEXTDISC funkcija iz rada
    int nextdisc(int disc) {
        return (disc + 1) % k;
    }
    
    // Kreira superkey za poređenje
    std::vector<double> superkey(const std::vector<double>& point, int j) {
        std::vector<double> sk;
        // Ciklična konkatenacija: Kj, Kj+1, ..., Kk-1, K0, ..., Kj-1
        for (int i = j; i < k; i++) {
            sk.push_back(point[i]);
        }
        for (int i = 0; i < j; i++) {
            sk.push_back(point[i]);
        }
        return sk;
    }
    
    // SUCCESSOR funkcija iz rada
    enum SuccessorResult { LOSON, HISON, EQUAL };
    
    SuccessorResult successor(KDNode* node, const std::vector<double>& point) {
        int j = node->disc;
        
        if (point[j] < node->point[j]) {
            return LOSON;
        } else if (point[j] > node->point[j]) {
            return HISON;
        } else {
            // Ako su Kj jednaki, uporedi superkeys
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
    
    // Rekurzivna pretraga za brisanje
    KDNode* findMin(KDNode* node, int dim, int currentDisc) {
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
    
    KDNode* findMax(KDNode* node, int dim, int currentDisc) {
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
    
    // DELETE algoritam iz rada
    KDNode* deleteNode(KDNode* node, const std::vector<double>& point) {
        if (node == nullptr) return nullptr;
        
        int j = node->disc;
        
        // Ako smo našli čvor za brisanje
        if (node->point == point) {
            // D1: Da li je P list?
            if (node->hison == nullptr && node->loson == nullptr) {
                delete node;
                return nullptr;
            }
            
            // D2: Odluči odakle uzeti P-ov nasljednika
            KDNode* replacement;
            if (node->hison != nullptr) {
                // D3: Uzmi sljedeći korijen iz HISON(P)
                replacement = findMin(node->hison, j, nextdisc(j));
                node->point = replacement->point;
                node->hison = deleteNode(node->hison, replacement->point);
            } else {
                // D4: Uzmi sljedeći korijen iz LOSON(P)
                replacement = findMax(node->loson, j, nextdisc(j));
                node->point = replacement->point;
                node->hison = deleteNode(node->loson, replacement->point);
                node->loson = nullptr;
            }
            
            return node;
        }
        
        // Nastavi pretragu
        SuccessorResult succ = successor(node, point);
        if (succ == LOSON) {
            node->loson = deleteNode(node->loson, point);
        } else if (succ == HISON) {
            node->hison = deleteNode(node->hison, point);
        }
        
        return node;
    }
    
    // Rekurzivna pretraga
    KDNode* searchRec(KDNode* node, const std::vector<double>& point) {
        if (node == nullptr) return nullptr;
        
        if (node->point == point) {
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
    
    // In-order obilazak
    void inorderRec(KDNode* node) {
        if (node != nullptr) {
            inorderRec(node->loson);
            std::cout << "(";
            for (size_t i = 0; i < node->point.size(); i++) {
                std::cout << node->point[i];
                if (i < node->point.size() - 1) std::cout << ",";
            }
            std::cout << ") disc=" << node->disc << std::endl;
            inorderRec(node->hison);
        }
    }

public:
    KDTree(int dimensions) : k(dimensions), root(nullptr) {}
    
    ~KDTree() {
        delete root;
    }
    
    // Algorithm INSERT iz rada
    bool insert(const std::vector<double>& point) {
        if (point.size() != k) {
            std::cerr << "Dimenzija tačke ne odgovara!" << std::endl;
            return false;
        }
        
        // I1: Provjeri da li je stablo prazno
        if (root == nullptr) {
            root = new KDNode(point, 0);
            return true;
        }
        
        KDNode* Q = root;
        
        while (true) {
            // I2: Uporedi
            if (Q->point == point) {
                return false;  // Tačka već postoji
            }
            
            SuccessorResult succ = successor(Q, point);
            
            if (succ == EQUAL) {
                return false;  // Svi ključevi jednaki
            }
            
            // Odredi koji sin
            KDNode** nextSon = (succ == LOSON) ? &(Q->loson) : &(Q->hison);
            
            if (*nextSon == nullptr) {
                // I4: Ubaci novi čvor u stablo
                *nextSon = new KDNode(point, nextdisc(Q->disc));
                return true;
            }
            
            // I3: Pomjeri se nadole
            Q = *nextSon;
        }
    }
    
    // Pretraga tačke
    bool search(const std::vector<double>& point) {
        return searchRec(root, point) != nullptr;
    }
    
    // Brisanje tačke
    void remove(const std::vector<double>& point) {
        root = deleteNode(root, point);
    }
    
    // Ispis stabla
    void inorder() {
        inorderRec(root);
    }
    
    // Nearest neighbor search (osnovna verzija)
    std::vector<double> nearestNeighbor(const std::vector<double>& target) {
        if (root == nullptr) return {};
        
        std::vector<double> best = root->point;
        double bestDist = distance(target, root->point);
        
        nearestNeighborRec(root, target, best, bestDist);
        
        return best;
    }
    
private:
    double distance(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    void nearestNeighborRec(KDNode* node, const std::vector<double>& target,
                           std::vector<double>& best, double& bestDist) {
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
};

// Test program
int main() {
    std::cout << "=== K-D Tree implementacija (Bentley 1975) ===" << std::endl;
    
    // Primjer iz rada: 2-d stablo
    KDTree tree(2);
    
    std::cout << "\nUbacivanje tačaka iz Figure 1:" << std::endl;
    tree.insert({50, 50});  // A
    tree.insert({10, 70});  // B
    tree.insert({80, 85});  // C
    tree.insert({25, 20});  // D
    tree.insert({40, 85});  // E
    tree.insert({70, 85});  // F
    
    std::cout << "\nIn-order obilazak:" << std::endl;
    tree.inorder();
    
    std::cout << "\nPretraga tačke (50,50): " 
              << (tree.search({50, 50}) ? "Pronađena" : "Nije pronađena") << std::endl;
    std::cout << "Pretraga tačke (99,99): " 
              << (tree.search({99, 99}) ? "Pronađena" : "Nije pronađena") << std::endl;
    
    std::cout << "\nNajbliži susjed tački (45, 45):" << std::endl;
    auto nearest = tree.nearestNeighbor({45, 45});
    std::cout << "(" << nearest[0] << "," << nearest[1] << ")" << std::endl;
    
    std::cout << "\nBrisanje tačke (25,20)..." << std::endl;
    tree.remove({25, 20});
    
    std::cout << "\nStablo nakon brisanja:" << std::endl;
    tree.inorder();
    
    return 0;
}