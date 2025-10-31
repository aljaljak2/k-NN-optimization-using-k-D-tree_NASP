# Implementation Notes: k-NN Variants Comparison (2022)

**Paper**: "Comparative performance analysis of K-nearest neighbour (KNN) algorithm and its different variants for disease prediction"
**Authors**: Uddin, S. et al.
**Source**: Scientific Reports, Vol. 12
**Link**: https://www.nature.com/articles/s41598-022-10358-x

## k-NN Variants Compared

### 1. Classic k-NN
- Standard Euclidean distance
- Majority voting for classification
- Our baseline implementation

### 2. Adaptive k-NN
- Dynamically adjust k based on local density
- Different k for different query points

### 3. Locally Adaptive k-NN
- Adapt both k and distance metric locally
- Context-dependent neighborhoods

### 4. k-means Clustering k-NN
- Pre-cluster training data
- Search within relevant clusters only

### 5. Fuzzy k-NN
- Fuzzy membership degrees
- Weighted voting based on membership

### 6. Mutual k-NN
- Consider reciprocal nearest neighbors
- Both points must be in each other's k-NN

### 7. Ensemble k-NN
- Combine multiple k-NN classifiers
- Vote aggregation

### 8. Hassanat k-NN
- Modified distance metric
- Better handling of different scales

### 9. Generalized Mean Distance k-NN
- Use generalized mean for distance
- Parameter controls mean type

## Datasets Used in Paper

8 medical datasets from:
- Kaggle
- UCI Machine Learning Repository
- OpenML

Topics: Various disease prediction tasks

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Training/prediction time

## Implementation Priority

1. **Phase 1**: Classic k-NN (baseline)
2. **Phase 2**: Distance-weighted k-NN
3. **Phase 3**: Adaptive k-NN
4. **Phase 4**: Other variants as needed

## Status

**Current**: Design phase
**Next**: Implement classic k-NN first
