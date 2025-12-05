"""
Plot bound width comparison for different CROWN methods across epsilon values
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_bound_comparison(output_path='bound_comparison.png'):
    """Plot bound width comparison for different methods across epsilon values"""
    
    # Data from the tables provided
    # Format: {method: [avg_bound_width for eps=0.05, 0.1, 0.3]}
    epsilon_values = [0.05, 0.1, 0.3]
    
    methods_data = {
        'IBP': [23.831, 44.3, 247.2],
        'CROWN-IBP': [6.704, 14.7, 142.6],
        'CROWN': [3.214, 2.2, 30.2],
        'alpha-CROWN': [3.215, 2.0, 22.7]
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Color palette for different methods
    colors = {
        'IBP': '#d62728',           # Red
        'CROWN-IBP': '#ff7f0e',      # Orange
        'CROWN': '#2ca02c',          # Green
        'alpha-CROWN': '#1f77b4'     # Blue
    }
    
    # Marker styles for different methods
    markers = {
        'IBP': 'o',
        'CROWN-IBP': 's',
        'CROWN': '^',
        'alpha-CROWN': 'D'
    }
    
    # Plot lines for each method
    for method, values in methods_data.items():
        color = colors.get(method, None)
        marker = markers.get(method, 'o')
        ax.plot(epsilon_values, values, 
                marker=marker, 
                label=method,
                color=color, 
                linewidth=2.5, 
                markersize=10,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Bound Width', fontsize=14, fontweight='bold')
    ax.set_title('Bound Width Comparison: ResNet on MNIST', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks to match epsilon values
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
    
    # Use log scale for y-axis since values span a large range
    ax.set_yscale('log')
    
    # Add minor grid lines for log scale
    ax.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bound comparison plot saved to {output_path}")
    plt.close()

def plot_bound_comparison_linear(output_path='bound_comparison_linear.png'):
    """Plot bound width comparison with linear y-axis scale"""
    
    # Data from the tables provided
    epsilon_values = [0.05, 0.1, 0.3]
    
    methods_data = {
        'IBP': [23.831, 44.3, 247.2],
        'CROWN-IBP': [6.704, 14.7, 142.6],
        'CROWN': [3.214, 2.2, 30.2],
        'alpha-CROWN': [3.215, 2.0, 22.7]
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Color palette for different methods
    colors = {
        'IBP': '#d62728',           # Red
        'CROWN-IBP': '#ff7f0e',      # Orange
        'CROWN': '#2ca02c',          # Green
        'alpha-CROWN': '#1f77b4'     # Blue
    }
    
    # Marker styles for different methods
    markers = {
        'IBP': 'o',
        'CROWN-IBP': 's',
        'CROWN': '^',
        'alpha-CROWN': 'D'
    }
    
    # Plot lines for each method
    for method, values in methods_data.items():
        color = colors.get(method, None)
        marker = markers.get(method, 'o')
        ax.plot(epsilon_values, values, 
                marker=marker, 
                label=method,
                color=color, 
                linewidth=2.5, 
                markersize=10,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Bound Width', fontsize=14, fontweight='bold')
    ax.set_title('Bound Width Comparison: ResNet on MNIST', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks to match epsilon values
    ax.set_xticks(epsilon_values)
    ax.set_xticklabels([f'{eps:.2f}' for eps in epsilon_values])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bound comparison plot (linear scale) saved to {output_path}")
    plt.close()

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create plots
    output_path1 = os.path.join(script_dir, 'bound_comparison.png')
    plot_bound_comparison(output_path1)
    
    output_path2 = os.path.join(script_dir, 'bound_comparison_linear.png')
    plot_bound_comparison_linear(output_path2)
    
    print("\nVisualization complete!")
    print(f"Log scale plot: {output_path1}")
    print(f"Linear scale plot: {output_path2}")

if __name__ == "__main__":
    main()

