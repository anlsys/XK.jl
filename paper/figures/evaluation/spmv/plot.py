import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def calculate_gflops(n, density, time):
    """
    Calculate GFLOP/s for SpMV operation.
    SpMV performs approximately 2*nnz operations (one multiply, one add per non-zero)
    where nnz = n*n*density
    """
    nnz = n * n * density
    flops = 2 * nnz
    gflops = (flops / time) / 1e9
    return gflops

def calculate_memory(n, density):
    """
    Calculate memory consumption in GB.
    memory = 1 matrix + 2 vectors
    sizeof(double) = 8 bytes
    """
    nnz=n*n*density
    rows   = n+1
    cols   = nnz*8
    values = nnz*8
    vectors = 2*n*8
    memory_bytes = rows+cols+values+vectors
    memory_gb = memory_bytes / (1024**3)
    return memory_gb

def plot_spmv_performance(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file, sep=':')

    # Calculate GFLOP/s
    df['gflops'] = df.apply(lambda row: calculate_gflops(row['n'], row['density'], row['time']), axis=1)

    # Group by ngpus and n, calculate mean and std GFLOP/s
    grouped = df.groupby(['ngpus', 'n'])['gflops'].agg(['mean', 'std']).reset_index()

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ax3 = ax1.twiny()  # Second x-axis for nnz

    # Define styles for different ngpus
    styles = {
        1: {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'label': '1 H100'},
        4: {'color': 'red', 'linestyle': '--', 'marker': 's', 'label': '4 H100s'}
    }

    # Plot performance curves with error bars
    for ngpus in sorted(grouped['ngpus'].unique()):
        data = grouped[grouped['ngpus'] == ngpus]
        style = styles[ngpus]
        ax1.errorbar(data['n'], data['mean'], yerr=data['std'],
                    color=style['color'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=8,
                    linewidth=2.5,
                    capsize=5,
                    capthick=2,
                    label=style['label'])

    # Calculate and plot memory consumption
    unique_n = sorted(grouped['n'].unique())
    density = df['density'].iloc[0]  # Assuming constant density
    memory = [calculate_memory(n, density) for n in unique_n]

    ax2.plot(unique_n, memory,
            color='green',
            linestyle='-.',
            marker='^',
            markersize=8,
            linewidth=2.5,
            label='Memory (GB)')

    # Set labels and title
    ax1.set_xlabel('Matrix Size (n)', fontsize=20)
    ax1.set_ylabel('Performance (GFLOP/s)', fontsize=20, color='black')
    ax2.set_ylabel('Memory Consumption (GB)', fontsize=20, color='green')

    # plt.title(f'SpMV Performance vs Matrix Size (Density = {density})',
    #          fontsize=14, fontweight='bold', pad=20)

    # Set log scale for x-axis
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax2.set_yscale('log')

    # Color the y-axis labels and set tick label size
    ax1.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=18)

    # Add grids
    ax1.grid(True, which='both', alpha=0.3)

    # Add horizontal line for H100 memory capacity
    ax2.axhline(y=80, color='green', linestyle=':', linewidth=2.5)

    # Add text label on the line
    ax2.text(ax1.get_xlim()[1] * 0.95, 80 * 1.15, '1x H100 Memory Capacity (80 GB)',
             fontsize=14, color='green', ha='right', va='bottom')

    ax1.set_xlim(right=2**18)
    ax2.set_ylim(top=500)

    # Set up the second x-axis for nnz (after setting scales and limits)
    unique_n = sorted(grouped['n'].unique())
    nnz_values = [n * n * density for n in unique_n]

    ax3.set_xlabel('Number of Non-zeros (nnz)', fontsize=20)
    ax3.set_xscale('log', base=2)
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(unique_n)
    ax3.set_xticklabels([f'{int(nnz):,}' for nnz in nnz_values])
    ax3.tick_params(axis='x', labelsize=14, labelrotation=15)

    # Set x-axis tick labels to actual values instead of powers of 2
    from matplotlib.ticker import FuncFormatter
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=16)

    # Tight layout
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    plot_spmv_performance(csv_file)
