# k-NN Algorithm Optimization using k-d Trees

Research project implementing and comparing various k-NN algorithm optimizations using k-dimensional trees.

## Scientific References

This implementation follows algorithms and techniques from the following peer-reviewed papers:

### 1. Bentley, J. L. (1975) - Original k-d tree paper
**"Multidimensional binary search trees used for associative searching"**
- Source: Communications of the ACM, Vol. 18, Issue 9, pp. 509-517
- Link: https://dl.acm.org/doi/10.1145/361002.361007
- **Implementation**: Core k-d tree structure, INSERT, DELETE, SEARCH algorithms

### 2. Pinkham, R., Zeng, S., Zhang, Z. (2020) - QuickNN
**"QuickNN: Memory and Performance Optimization of k-d Tree Based Nearest Neighbor Search for 3D Point Clouds"**
- Source: IEEE HPCA 2020
- Link: https://ieeexplore.ieee.org/document/9065602/
- PDF: https://web.eecs.umich.edu/~zhengya/papers/pinkham_hpca20.pdf
- **Dataset**: KITTI (30k 3D points), Ford Campus Vision
- **Implementation**: Memory layout optimization, cache-friendly structures

### 3. Jiang, K., et al. (2018) - Revised k-d tree
**"Fast neighbor search by using revised k-d tree"**
- Source: Information Sciences, Vol. 472
- Link: https://www.sciencedirect.com/science/article/abs/pii/S0020025518307126
- **Implementation**: Techniques for reducing unnecessary distance calculations

### 4. Uddin, S. et al. (2022) - k-NN variants comparison
**"Comparative performance analysis of K-nearest neighbour (KNN) algorithm and its different variants for disease prediction"**
- Source: Scientific Reports, Vol. 12
- Link: https://www.nature.com/articles/s41598-022-10358-x
- **Datasets**: 8 benchmark datasets (Kaggle, UCI ML Repository, OpenML)
- **Implementation**: Comparison of 9 k-NN variants

## Project Structure

```
.
├── include/              # Header files
│   ├── kdtree/          # Bentley (1975) k-d tree
│   ├── knn/             # k-NN implementations
│   ├── optimizations/   # QuickNN, Revised k-d tree
│   └── utils/           # Utilities (distance metrics, data loading)
├── src/                 # Implementation files
├── benchmarks/          # Performance testing
├── tests/               # Unit tests
├── datasets/            # Benchmark datasets
├── visualization/       # Visualization tools
├── examples/            # Usage examples
├── docs/                # Documentation and paper notes
└── results/             # Experimental results
```

## Implementations

### Completed
- ✓ Project structure

### In Progress
- Basic k-d tree (Bentley 1975)

### Planned
1. Classic k-NN (brute force baseline)
2. k-NN with k-d tree optimization
3. Benchmarking framework
4. 2D/3D visualization
5. QuickNN optimizations (Pinkham et al. 2020)
6. Revised k-d tree (Jiang et al. 2018)
7. k-NN variants (Uddin et al. 2022)

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Running Examples

```bash
# Bentley (1975) k-d tree examples
./examples/example_bentley_1975

# Basic k-NN vs k-d tree k-NN comparison
./benchmarks/benchmark_basic_vs_kdtree
```

## Datasets

Datasets are organized in `datasets/` directory:
- `synthetic/` - Generated test data
- `kitti/` - KITTI 3D point cloud dataset
- `uci/` - UCI Machine Learning Repository datasets

See [datasets/README.md](datasets/README.md) for dataset sources and citations.

## Documentation

Detailed notes on each paper implementation are in the `docs/` directory:
- [Bentley 1975 Implementation Notes](docs/bentley_1975_notes.md)
- [QuickNN Implementation Notes](docs/quicknn_notes.md)
- [Revised k-d tree Notes](docs/revised_kdtree_notes.md)
- [k-NN Variants Notes](docs/uddin_2022_notes.md)

## License

Academic/Research use only. Please cite the original papers when using this code.
