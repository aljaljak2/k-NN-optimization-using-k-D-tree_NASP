# Datasets

## Directory Structure

- `synthetic/` - Synthetically generated datasets for testing
- `kitti/` - KITTI Vision Benchmark Suite (3D point clouds)
- `uci/` - UCI Machine Learning Repository datasets

## Dataset Sources

### KITTI Dataset
- **Source**: http://www.cvlibs.net/datasets/kitti/
- **Used in**: Pinkham et al. (2020) - QuickNN paper
- **Description**: 3D point cloud data from autonomous driving scenarios
- **Size**: ~30,000 points per scene
- **Citation**:
  ```
  @inproceedings{Geiger2012CVPR,
    author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2012}
  }
  ```

### UCI Machine Learning Repository
- **Source**: https://archive.ics.uci.edu/ml/index.php
- **Used in**: Uddin et al. (2022) - k-NN variants comparison
- **Common datasets**:
  - Iris Dataset
  - Wine Dataset
  - Breast Cancer Wisconsin
  - Diabetes Dataset

### Kaggle Datasets
- **Source**: https://www.kaggle.com/datasets
- **Used in**: Uddin et al. (2022)
- Various medical and classification datasets

## Synthetic Data Generation

Synthetic datasets can be generated using the `DatasetLoader` utility:

```cpp
#include "utils/dataset_loader.h"

// Generate random points
auto data = DatasetLoader::generateRandom(1000, 3, 42);

// Generate clustered data
auto clustered = DatasetLoader::generateClustered(5, 200, 3, 42);
```

## Adding New Datasets

1. Place dataset files in the appropriate subdirectory
2. Document the source and citation
3. Create a loader function if needed
4. Update this README with dataset information
