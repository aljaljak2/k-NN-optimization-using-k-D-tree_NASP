#!/usr/bin/env python3
"""
Visualization script for KNN classification metrics
Supports comparison of multiple algorithms (e.g., Basic KNN vs KD-Tree KNN)
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_confusion_matrix(cm_dict, algorithm_name, ax=None):
    """Plot confusion matrix as heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert dict to numpy array
    labels = sorted(cm_dict.keys(), key=int)
    cm = np.zeros((len(labels), len(labels)))

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = cm_dict[true_label].get(pred_label, 0)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {algorithm_name}')

    return ax

def plot_metrics_bar(metrics_dict, metric_name, ax=None):
    """Plot per-class metrics as bar chart"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    classes = sorted(metrics_dict.keys(), key=int)
    values = [metrics_dict[c] for c in classes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), values, color=colors, alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'Per-Class {metric_name.capitalize()}')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    return ax

def plot_roc_curves(metrics_list, algorithm_names, ax=None):
    """
    Plot ROC curves for multiple algorithms on the same plot
    metrics_list: list of metric dictionaries
    algorithm_names: list of algorithm names
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

    for idx, (metrics, alg_name) in enumerate(zip(metrics_list, algorithm_names)):
        roc_data = metrics.get('roc_curve', {})

        for class_label, points in roc_data.items():
            if not points:
                continue

            fpr = [p['fpr'] for p in points]
            tpr = [p['tpr'] for p in points]

            # Sort by fpr for proper curve
            sorted_pairs = sorted(zip(fpr, tpr))
            fpr, tpr = zip(*sorted_pairs) if sorted_pairs else ([], [])

            # Add (0,0) and (1,1) for complete curve
            fpr = [0] + list(fpr) + [1]
            tpr = [0] + list(tpr) + [1]

            label = f'{alg_name} - Class {class_label}'
            ax.plot(fpr, tpr, label=label, color=colors[idx],
                   linewidth=2, marker='o', markersize=4)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    return ax

def create_summary_report(metrics_list, algorithm_names):
    """Create a text summary of metrics"""
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS SUMMARY")
    print("="*60)

    for metrics, alg_name in zip(metrics_list, algorithm_names):
        print(f"\n{alg_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

        print(f"\n  Per-Class Metrics:")
        classes = sorted(metrics['precision'].keys(), key=int)
        print(f"    {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"    {'-'*44}")

        for c in classes:
            prec = metrics['precision'][c]
            rec = metrics['recall'][c]
            f1 = metrics['f1_score'][c]
            print(f"    {c:<8} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_metrics.py <metrics_file1.json> [metrics_file2.json ...]")
        print("\nExample:")
        print("  python visualize_metrics.py metrics_basic.json")
        print("  python visualize_metrics.py metrics_basic.json metrics_kdtree.json")
        sys.exit(1)

    # Load all metrics files
    metrics_files = sys.argv[1:]
    metrics_list = []
    algorithm_names = []

    for filepath in metrics_files:
        metrics = load_metrics(filepath)
        metrics_list.append(metrics)
        algorithm_names.append(metrics.get('algorithm', Path(filepath).stem))

    # Print summary
    create_summary_report(metrics_list, algorithm_names)

    # Create visualizations
    if len(metrics_list) == 1:
        # Single algorithm - detailed view
        metrics = metrics_list[0]
        alg_name = algorithm_names[0]

        fig = plt.figure(figsize=(16, 12))

        # Confusion matrix
        ax1 = plt.subplot(2, 3, 1)
        plot_confusion_matrix(metrics['confusion_matrix'], alg_name, ax1)

        # Precision
        ax2 = plt.subplot(2, 3, 2)
        plot_metrics_bar(metrics['precision'], 'Precision', ax2)

        # Recall
        ax3 = plt.subplot(2, 3, 3)
        plot_metrics_bar(metrics['recall'], 'Recall', ax3)

        # F1-Score
        ax4 = plt.subplot(2, 3, 4)
        plot_metrics_bar(metrics['f1_score'], 'F1-Score', ax4)

        # ROC Curve
        ax5 = plt.subplot(2, 3, (5, 6))
        plot_roc_curves([metrics], [alg_name], ax5)

        plt.suptitle(f'Classification Metrics - {alg_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

    else:
        # Multiple algorithms - comparison view
        fig = plt.figure(figsize=(18, 10))

        # ROC curves comparison (large)
        ax1 = plt.subplot(2, 3, (1, 4))
        plot_roc_curves(metrics_list, algorithm_names, ax1)

        # Accuracy comparison
        ax2 = plt.subplot(2, 3, 2)
        accuracies = [m['accuracy'] for m in metrics_list]
        colors = plt.cm.viridis(np.linspace(0, 1, len(accuracies)))
        bars = ax2.bar(range(len(algorithm_names)), accuracies, color=colors, alpha=0.8)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Comparison')
        ax2.set_xticks(range(len(algorithm_names)))
        ax2.set_xticklabels(algorithm_names, rotation=15)
        ax2.set_ylim([0, 1.05])
        ax2.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        # Confusion matrices side by side
        for idx, (metrics, alg_name) in enumerate(zip(metrics_list[:2], algorithm_names[:2])):
            ax = plt.subplot(2, 3, 3 + idx * 3)
            plot_confusion_matrix(metrics['confusion_matrix'], alg_name, ax)

        plt.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

    # Save figure
    output_file = 'metrics_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
