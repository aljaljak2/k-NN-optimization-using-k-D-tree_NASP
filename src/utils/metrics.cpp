#include "../../include/utils/metrics.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <set>

double Metrics::accuracy(const std::vector<int>& true_labels,
                        const std::vector<int>& predicted_labels) {
    if (true_labels.size() != predicted_labels.size() || true_labels.empty()) {
        return 0.0;
    }

    int correct = 0;
    for (size_t i = 0; i < true_labels.size(); i++) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / true_labels.size();
}

std::map<int, std::map<int, int>> Metrics::confusionMatrix(
    const std::vector<int>& true_labels,
    const std::vector<int>& predicted_labels) {

    std::map<int, std::map<int, int>> cm;

    for (size_t i = 0; i < true_labels.size(); i++) {
        cm[true_labels[i]][predicted_labels[i]]++;
    }

    return cm;
}

std::map<int, double> Metrics::precision(const std::vector<int>& true_labels,
                                        const std::vector<int>& predicted_labels) {
    std::map<int, double> precisions;
    auto cm = confusionMatrix(true_labels, predicted_labels);

    // Get all unique labels
    std::set<int> labels;
    for (int label : true_labels) labels.insert(label);
    for (int label : predicted_labels) labels.insert(label);

    for (int label : labels) {
        int tp = cm[label][label];  // True Positives
        int fp = 0;  // False Positives

        // Count false positives: predicted as 'label' but actually something else
        for (int other : labels) {
            if (other != label) {
                fp += cm[other][label];
            }
        }

        if (tp + fp > 0) {
            precisions[label] = static_cast<double>(tp) / (tp + fp);
        } else {
            precisions[label] = 0.0;
        }
    }

    return precisions;
}

std::map<int, double> Metrics::recall(const std::vector<int>& true_labels,
                                     const std::vector<int>& predicted_labels) {
    std::map<int, double> recalls;
    auto cm = confusionMatrix(true_labels, predicted_labels);

    std::set<int> labels;
    for (int label : true_labels) labels.insert(label);
    for (int label : predicted_labels) labels.insert(label);

    for (int label : labels) {
        int tp = cm[label][label];  // True Positives
        int fn = 0;  // False Negatives

        // Count false negatives: actually 'label' but predicted as something else
        for (int other : labels) {
            if (other != label) {
                fn += cm[label][other];
            }
        }

        if (tp + fn > 0) {
            recalls[label] = static_cast<double>(tp) / (tp + fn);
        } else {
            recalls[label] = 0.0;
        }
    }

    return recalls;
}

std::map<int, double> Metrics::f1Score(const std::vector<int>& true_labels,
                                      const std::vector<int>& predicted_labels) {
    std::map<int, double> f1scores;
    auto prec = precision(true_labels, predicted_labels);
    auto rec = recall(true_labels, predicted_labels);

    std::set<int> labels;
    for (int label : true_labels) labels.insert(label);

    for (int label : labels) {
        double p = prec[label];
        double r = rec[label];

        if (p + r > 0) {
            f1scores[label] = 2 * (p * r) / (p + r);
        } else {
            f1scores[label] = 0.0;
        }
    }

    return f1scores;
}

std::map<int, std::vector<Metrics::ROCPoint>> Metrics::rocCurve(
    const std::vector<int>& true_labels,
    const std::vector<int>& predicted_labels,
    const std::vector<std::vector<double>>& prediction_scores) {

    std::map<int, std::vector<ROCPoint>> roc_curves;

    // If no prediction scores provided, use simple binary predictions
    if (prediction_scores.empty()) {
        std::set<int> labels;
        for (int label : true_labels) labels.insert(label);

        // For each class, calculate single ROC point
        for (int target_class : labels) {
            std::vector<ROCPoint> points;

            int tp = 0, fp = 0, tn = 0, fn = 0;

            for (size_t i = 0; i < true_labels.size(); i++) {
                bool actual_positive = (true_labels[i] == target_class);
                bool predicted_positive = (predicted_labels[i] == target_class);

                if (actual_positive && predicted_positive) tp++;
                else if (!actual_positive && predicted_positive) fp++;
                else if (!actual_positive && !predicted_positive) tn++;
                else fn++;
            }

            double tpr = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
            double fpr = (fp + tn > 0) ? static_cast<double>(fp) / (fp + tn) : 0.0;

            points.push_back({fpr, tpr, 0.5});
            roc_curves[target_class] = points;
        }
    }

    return roc_curves;
}

void Metrics::printMetrics(const std::vector<int>& true_labels,
                          const std::vector<int>& predicted_labels) {
    std::cout << "\n=== Classification Metrics ===" << std::endl;

    // Accuracy
    double acc = accuracy(true_labels, predicted_labels);
    std::cout << "\nAccuracy: " << std::fixed << std::setprecision(4)
              << (acc * 100) << "%" << std::endl;

    // Per-class metrics
    auto prec = precision(true_labels, predicted_labels);
    auto rec = recall(true_labels, predicted_labels);
    auto f1 = f1Score(true_labels, predicted_labels);

    std::cout << "\nPer-class metrics:" << std::endl;
    std::cout << std::setw(10) << "Class"
              << std::setw(15) << "Precision"
              << std::setw(15) << "Recall"
              << std::setw(15) << "F1-Score" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    std::set<int> labels;
    for (int label : true_labels) labels.insert(label);

    for (int label : labels) {
        std::cout << std::setw(10) << label
                  << std::setw(15) << std::fixed << std::setprecision(4) << prec[label]
                  << std::setw(15) << rec[label]
                  << std::setw(15) << f1[label] << std::endl;
    }

    // Confusion Matrix
    auto cm = confusionMatrix(true_labels, predicted_labels);
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << std::setw(10) << "True\\Pred";
    for (int label : labels) {
        std::cout << std::setw(10) << label;
    }
    std::cout << std::endl;

    for (int true_label : labels) {
        std::cout << std::setw(10) << true_label;
        for (int pred_label : labels) {
            std::cout << std::setw(10) << cm[true_label][pred_label];
        }
        std::cout << std::endl;
    }
}

void Metrics::saveMetricsJSON(const std::vector<int>& true_labels,
                             const std::vector<int>& predicted_labels,
                             const std::string& outputFile,
                             const std::string& algorithmName,
                             const std::vector<std::vector<double>>& prediction_scores) {
    std::ofstream file(outputFile);
    if (!file.is_open()) {
        std::cerr << "Could not open output file: " << outputFile << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"algorithm\": \"" << algorithmName << "\",\n";

    // Accuracy
    file << "  \"accuracy\": " << accuracy(true_labels, predicted_labels) << ",\n";

    // Per-class metrics
    auto prec = precision(true_labels, predicted_labels);
    auto rec = recall(true_labels, predicted_labels);
    auto f1 = f1Score(true_labels, predicted_labels);

    std::set<int> labels;
    for (int label : true_labels) labels.insert(label);

    file << "  \"precision\": {";
    bool first = true;
    for (int label : labels) {
        if (!first) file << ", ";
        file << "\"" << label << "\": " << prec[label];
        first = false;
    }
    file << "},\n";

    file << "  \"recall\": {";
    first = true;
    for (int label : labels) {
        if (!first) file << ", ";
        file << "\"" << label << "\": " << rec[label];
        first = false;
    }
    file << "},\n";

    file << "  \"f1_score\": {";
    first = true;
    for (int label : labels) {
        if (!first) file << ", ";
        file << "\"" << label << "\": " << f1[label];
        first = false;
    }
    file << "},\n";

    // Confusion Matrix
    auto cm = confusionMatrix(true_labels, predicted_labels);
    file << "  \"confusion_matrix\": {\n";
    first = true;
    for (int true_label : labels) {
        if (!first) file << ",\n";
        file << "    \"" << true_label << "\": {";
        bool first_pred = true;
        for (int pred_label : labels) {
            if (!first_pred) file << ", ";
            file << "\"" << pred_label << "\": " << cm[true_label][pred_label];
            first_pred = false;
        }
        file << "}";
        first = false;
    }
    file << "\n  },\n";

    // ROC curve data
    auto roc = rocCurve(true_labels, predicted_labels, prediction_scores);
    file << "  \"roc_curve\": {\n";
    first = true;
    for (const auto& [class_label, points] : roc) {
        if (!first) file << ",\n";
        file << "    \"" << class_label << "\": [\n";
        for (size_t i = 0; i < points.size(); i++) {
            file << "      {\"fpr\": " << points[i].fpr
                 << ", \"tpr\": " << points[i].tpr
                 << ", \"threshold\": " << points[i].threshold << "}";
            if (i < points.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ]";
        first = false;
    }
    file << "\n  }\n";

    file << "}\n";
    file.close();

    std::cout << "Metrics saved to: " << outputFile << std::endl;
}

void Metrics::evaluate(const std::vector<int>& true_labels,
                      const std::vector<int>& predicted_labels,
                      const std::string& outputFile) {
    printMetrics(true_labels, predicted_labels);

    if (!outputFile.empty()) {
        saveMetricsJSON(true_labels, predicted_labels, outputFile);
    }
}
