#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_flops(n, time):
    """
    Compute FLOP/s for a GEMM operation of size m=n=k=n
    GEMM: C = A * B where A, B, C are n x n matrices
    FLOPs = 2 * n^3 (multiply-add operations)
    """
    flops = 2 * n**3
    return flops / time

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Read the CSV file
    df = pd.read_csv(csv_file, sep=':')

    # Compute FLOP/s for each row
    df['flops'] = df.apply(lambda row: compute_flops(row['n'], row['time']), axis=1)

    # Convert to TFLOP/s
    df['tflops'] = df['flops'] / 1e12

    # Group by ngpus and n, compute mean and std
    grouped = df.groupby(['ngpus', 'n'])['tflops'].agg(['mean', 'std']).reset_index()

    # Get unique number of GPUs
    ngpu_values = sorted(grouped['ngpus'].unique())

    # Define colors, markers, and linestyles for each curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot for each GPU count
    for i, ngpus in enumerate(ngpu_values):
        data = grouped[grouped['ngpus'] == ngpus]
        ax.errorbar(data['n'], data['mean'], yerr=data['std'],
                    marker=markers[i], label=f'{ngpus} Xe Stack', capsize=5,
                    linewidth=2.5, markersize=8, color=colors[i],
                    linestyle=linestyles[i])

    ax.set_xlabel('Matrix Size (n)', fontsize=20)
    ax.set_ylabel('Performance (TFLOP/s)', fontsize=20)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', labelsize=18)

    # Set x-axis tick labels to actual values instead of powers of 2
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Format the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
