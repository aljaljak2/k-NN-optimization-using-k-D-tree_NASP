#!/usr/bin/env python3
"""
Visualization script for KNN benchmark results
Generates plots for curse of dimensionality, scalability, and K parameter impact
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(filepath='build/benchmarks/results/benchmark_results.json'):
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_curse_of_dimensionality(results, output_dir='build/benchmarks/results/plots'):
    """Plot how performance degrades with increasing dimensions"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter results for curse of dimensionality test
    cod_results = [r for r in results if 'synthetic_' in r['dataset_name']
                   and r['dataset_name'].endswith('d')]

    algorithms = ['KNNBasic', 'KNNKDTree', 'KNNNanoflann']
    dimensions = sorted(list(set([r['n_dimensions'] for r in cod_results])))

    plt.figure(figsize=(12, 6))

    # Plot 1: Average query time
    plt.subplot(1, 2, 1)
    for algo in algorithms:
        times = [r['avg_query_time_ms'] for r in cod_results
                if r['algorithm'] == algo]
        times = sorted(times, key=lambda x: dimensions)
        plt.plot(dimensions, times, label=algo, marker='o', linewidth=2)

    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Average Query Time (ms)', fontsize=12)
    plt.title('Curse of Dimensionality - Query Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Plot 2: Speedup vs Basic
    plt.subplot(1, 2, 2)
    for algo in ['KNNKDTree', 'KNNNanoflann']:
        speedups = [r['speedup_vs_basic'] for r in cod_results
                   if r['algorithm'] == algo]
        speedups = sorted(speedups, key=lambda x: dimensions)
        plt.plot(dimensions, speedups, label=algo, marker='s', linewidth=2)

    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Speedup vs KNNBasic', fontsize=12)
    plt.title('Speedup Comparison', fontsize=14, fontweight='bold')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/curse_of_dimensionality.png', dpi=300)
    print(f"Saved: {output_dir}/curse_of_dimensionality.png")
    plt.close()

def plot_scalability(results, output_dir='build/benchmarks/results/plots'):
    """Plot how performance scales with dataset size"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter results for scalability test
    scal_results = [r for r in results if 'synthetic_n' in r['dataset_name']]

    algorithms = ['KNNBasic', 'KNNKDTree', 'KNNNanoflann']
    sample_sizes = sorted(list(set([r['n_samples'] for r in scal_results])))

    plt.figure(figsize=(12, 6))

    # Plot 1: Query time vs dataset size
    plt.subplot(1, 2, 1)
    for algo in algorithms:
        times = []
        for n in sample_sizes:
            res = [r for r in scal_results if r['algorithm'] == algo and r['n_samples'] == n]
            if res:
                times.append(res[0]['avg_query_time_ms'])
        plt.plot(sample_sizes, times, label=algo, marker='o', linewidth=2)

    plt.xlabel('Dataset Size (n_samples)', fontsize=12)
    plt.ylabel('Average Query Time (ms)', fontsize=12)
    plt.title('Scalability - Query Time vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    # Plot 2: Build time vs dataset size
    plt.subplot(1, 2, 2)
    for algo in ['KNNKDTree', 'KNNNanoflann']:
        build_times = []
        for n in sample_sizes:
            res = [r for r in scal_results if r['algorithm'] == algo and r['n_samples'] == n]
            if res:
                build_times.append(res[0]['build_time_ms'])
        plt.plot(sample_sizes, build_times, label=algo, marker='s', linewidth=2)

    plt.xlabel('Dataset Size (n_samples)', fontsize=12)
    plt.ylabel('Build Time (ms)', fontsize=12)
    plt.title('Build Time vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/scalability.png', dpi=300)
    print(f"Saved: {output_dir}/scalability.png")
    plt.close()

def plot_k_parameter_impact(results, output_dir='build/benchmarks/results/plots'):
    """Plot how K parameter affects performance"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter results for K parameter test
    k_results = [r for r in results if 'synthetic_k' in r['dataset_name']]

    algorithms = ['KNNBasic', 'KNNKDTree', 'KNNNanoflann']
    k_values = sorted(list(set([r['k_neighbors'] for r in k_results])))

    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        times = []
        for k in k_values:
            res = [r for r in k_results if r['algorithm'] == algo and r['k_neighbors'] == k]
            if res:
                times.append(res[0]['avg_query_time_ms'])
        plt.plot(k_values, times, label=algo, marker='o', linewidth=2)

    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Average Query Time (ms)', fontsize=12)
    plt.title('K Parameter Impact on Query Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/k_parameter_impact.png', dpi=300)
    print(f"Saved: {output_dir}/k_parameter_impact.png")
    plt.close()

def plot_distance_calculations_real_datasets(results, output_dir='build/benchmarks/results/plots'):
    """Plot average distance calculations per algorithm for real datasets"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Filter results for real datasets (non-synthetic)
    real_results = [r for r in results if 'synthetic' not in r['dataset_name'] and
                    'avg_distance_calculations_per_query' in r and
                    r.get('avg_distance_calculations_per_query', 0) > 0]

    if not real_results:
        print("Warning: No real dataset results with distance calculations found")
        return

    # Extract unique datasets
    datasets = {}
    for r in real_results:
        # Extract dataset name (remove _kX suffix)
        dataset_name = r['dataset_name']
        if '_k' in dataset_name:
            dataset_name = dataset_name.split('_k')[0]

        if dataset_name not in datasets:
            datasets[dataset_name] = {
                'dimensions': r['n_dimensions'],
                'KNNBasic': [],
                'KNNKDTree': [],
                'KNNNanoflann': []
            }

        algo = r['algorithm']
        if algo in datasets[dataset_name]:
            datasets[dataset_name][algo].append(r['avg_distance_calculations_per_query'])

    # Calculate averages for each algorithm per dataset
    dataset_names = []
    dataset_dims = []
    basic_avgs = []
    kdtree_avgs = []
    nano_avgs = []

    for name, data in sorted(datasets.items()):
        dataset_names.append(f"{name}\n({data['dimensions']}D)")
        dataset_dims.append(data['dimensions'])

        basic_avgs.append(np.mean(data['KNNBasic']) if data['KNNBasic'] else 0)
        kdtree_avgs.append(np.mean(data['KNNKDTree']) if data['KNNKDTree'] else 0)
        nano_avgs.append(np.mean(data['KNNNanoflann']) if data['KNNNanoflann'] else 0)

    # Create bar plot
    x = np.arange(len(dataset_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width, basic_avgs, width, label='KNNBasic', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, kdtree_avgs, width, label='KNNKDTree', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, nano_avgs, width, label='KNNNanoflann', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Dataset (Dimensions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Distance Calculations per Query', fontsize=13, fontweight='bold')
    ax.set_title('Distance Calculation Efficiency on Real Datasets', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Use linear scale to better show differences
    # Format y-axis with comma thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'))

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, rotation=0)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/distance_calculations_real_datasets.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/distance_calculations_real_datasets.png")
    plt.close()

def generate_latex_table(results, output_dir='build/benchmarks/results'):
    """Generate LaTeX table for academic paper"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Summary table for curse of dimensionality
    cod_results = [r for r in results if 'synthetic_' in r['dataset_name']
                   and r['dataset_name'].endswith('d')]

    latex = []
    latex.append("% KNN Benchmark Results - Curse of Dimensionality")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{|c|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("Dimensions & KNNBasic (ms) & KNNKDTree (ms) & KNNNanoflann (ms) & Best Speedup \\\\")
    latex.append("\\hline")

    dimensions = sorted(list(set([r['n_dimensions'] for r in cod_results])))
    for d in dimensions:
        basic_time = [r['avg_query_time_ms'] for r in cod_results
                     if r['algorithm'] == 'KNNBasic' and r['n_dimensions'] == d][0]
        kdtree_time = [r['avg_query_time_ms'] for r in cod_results
                      if r['algorithm'] == 'KNNKDTree' and r['n_dimensions'] == d][0]
        nano_time = [r['avg_query_time_ms'] for r in cod_results
                    if r['algorithm'] == 'KNNNanoflann' and r['n_dimensions'] == d][0]

        speedup_kdtree = basic_time / kdtree_time
        speedup_nano = basic_time / nano_time
        best_speedup = max(speedup_kdtree, speedup_nano)

        latex.append(f"{d} & {basic_time:.3f} & {kdtree_time:.3f} & {nano_time:.3f} & {best_speedup:.2f}x \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Query time comparison across different dimensionalities}")
    latex.append("\\label{tab:curse_of_dimensionality}")
    latex.append("\\end{table}")

    output_file = f'{output_dir}/benchmark_table.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"Saved: {output_file}")

def main():
    """Main function to generate all visualizations"""
    print("Loading benchmark results...")
    data = load_results()
    results = data['results']

    print(f"\nTotal tests: {data['benchmark_info']['total_tests']}")
    print(f"Total duration: {data['benchmark_info']['total_duration_sec']:.2f} seconds")
    print(f"Timestamp: {data['benchmark_info']['timestamp']}")

    print("\nGenerating visualizations...")
    plot_curse_of_dimensionality(results)
    plot_scalability(results)
    plot_k_parameter_impact(results)
    plot_distance_calculations_real_datasets(results)

    print("\nGenerating LaTeX table...")
    generate_latex_table(results)

    print("\n✓ All visualizations complete!")
    print("Check build/benchmarks/results/plots/ directory for PNG files")
    print("  - curse_of_dimensionality.png")
    print("  - scalability.png")
    print("  - k_parameter_impact.png")
    print("  - distance_calculations_real_datasets.png")
    print("Check build/benchmarks/results/ directory for LaTeX table")

if __name__ == '__main__':
    main()
