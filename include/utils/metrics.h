#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <map>
#include <string>
#include "point.h"

/**
 * Evaluation metrics for classification
 * Calculates accuracy, precision, recall, F1-score, confusion matrix, ROC curve
 */
class Metrics {
public:
    // Calculate all metrics and store results
    static void evaluate(const std::vector<int>& true_labels,
                        const std::vector<int>& predicted_labels,
                        const std::string& outputFile = "");

    // Individual metrics
    static double accuracy(const std::vector<int>& true_labels,
                          const std::vector<int>& predicted_labels);

    static std::map<int, double> precision(const std::vector<int>& true_labels,
                                           const std::vector<int>& predicted_labels);

    static std::map<int, double> recall(const std::vector<int>& true_labels,
                                       const std::vector<int>& predicted_labels);

    static std::map<int, double> f1Score(const std::vector<int>& true_labels,
                                         const std::vector<int>& predicted_labels);

    // Confusion matrix: confusionMatrix[true_label][predicted_label] = count
    static std::map<int, std::map<int, int>> confusionMatrix(
        const std::vector<int>& true_labels,
        const std::vector<int>& predicted_labels);

    // ROC curve data: for each class, return (FPR, TPR) pairs at different thresholds
    // For multi-class: uses One-vs-Rest approach
    struct ROCPoint {
        double fpr;  // False Positive Rate
        double tpr;  // True Positive Rate
        double threshold;
    };

    static std::map<int, std::vector<ROCPoint>> rocCurve(
        const std::vector<int>& true_labels,
        const std::vector<int>& predicted_labels,
        const std::vector<std::vector<double>>& prediction_scores);

    // Print metrics to console
    static void printMetrics(const std::vector<int>& true_labels,
                            const std::vector<int>& predicted_labels);

    // Save metrics to JSON file for Python visualization
    // Supports multiple runs for comparison (e.g., different algorithms)
    static void saveMetricsJSON(const std::vector<int>& true_labels,
                               const std::vector<int>& predicted_labels,
                               const std::string& outputFile,
                               const std::string& algorithmName = "KNN",
                               const std::vector<std::vector<double>>& prediction_scores = {});
};

#endif // METRICS_H
