"""
Plot training curves from training_curves.json
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_training_curves(json_path):
    """Load training curves from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_training_curves(curves_data, output_path='training_curves.png'):
    """Plot training curves for all models"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color palette for different models
    colors = {
        'natural': '#1f77b4',
        'IBP_eps01': '#ff7f0e',
        'IBP_eps03': '#2ca02c',
        'CROWN_IBP_eps01': '#d62728',
        'CROWN_IBP_eps03': '#9467bd',
        'CROWN_eps01': '#8c564b',
        'CROWN_eps03': '#e377c2',
    }
    
    # Plot 1: Training Loss (epoch level)
    ax1 = axes[0]
    for model_name, curves in curves_data.items():
        if 'train_loss' in curves and 'epoch' in curves:
            epochs = curves['epoch']
            train_loss = curves['train_loss']
            color = colors.get(model_name, None)
            ax1.plot(epochs, train_loss, marker='o', label=model_name, 
                    color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss (Epoch Level)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0.5)
    
    # Plot 2: Test Accuracy (epoch level)
    ax2 = axes[1]
    for model_name, curves in curves_data.items():
        if 'test_accuracy' in curves and 'epoch' in curves:
            epochs = curves['epoch']
            test_acc = curves['test_accuracy']
            color = colors.get(model_name, None)
            ax2.plot(epochs, test_acc, marker='s', label=model_name,
                    color=color, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy (Epoch Level)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()

def plot_detailed_training_curves(curves_data, output_path='training_curves_detailed.png'):
    """Plot detailed training curves including batch-level loss"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Color palette for different models
    colors = {
        'natural': '#1f77b4',
        'IBP_eps01': '#ff7f0e',
        'IBP_eps03': '#2ca02c',
        'CROWN_IBP_eps01': '#d62728',
        'CROWN_IBP_eps03': '#9467bd',
        'CROWN_eps01': '#8c564b',
        'CROWN_eps03': '#e377c2',
    }
    
    # Plot 1: Training Loss (epoch level)
    ax1 = axes[0]
    for model_name, curves in curves_data.items():
        if 'train_loss' in curves and 'epoch' in curves:
            epochs = curves['epoch']
            train_loss = curves['train_loss']
            color = colors.get(model_name, None)
            ax1.plot(epochs, train_loss, marker='o', label=model_name, 
                    color=color, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss (Epoch Level)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0.5)
    
    # Plot 2: Training Loss (batch level) - smoothed
    ax2 = axes[1]
    for model_name, curves in curves_data.items():
        if 'train_loss_batch' in curves and 'batch' in curves:
            batches = curves['batch']
            train_loss_batch = curves['train_loss_batch']
            
            # Smooth the batch-level loss using moving average (window=10)
            if len(train_loss_batch) > 10:
                window = 10
                smoothed_loss = []
                for i in range(len(train_loss_batch)):
                    start = max(0, i - window // 2)
                    end = min(len(train_loss_batch), i + window // 2 + 1)
                    smoothed_loss.append(np.mean(train_loss_batch[start:end]))
            else:
                smoothed_loss = train_loss_batch
            
            color = colors.get(model_name, None)
            ax2.plot(batches, smoothed_loss, label=model_name,
                    color=color, linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Batch', fontsize=12)
    ax2.set_ylabel('Training Loss (Smoothed)', fontsize=12)
    ax2.set_title('Training Loss (Batch Level, Smoothed)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Accuracy (epoch level)
    ax3 = axes[2]
    for model_name, curves in curves_data.items():
        if 'test_accuracy' in curves and 'epoch' in curves:
            epochs = curves['epoch']
            test_acc = curves['test_accuracy']
            color = colors.get(model_name, None)
            ax3.plot(epochs, test_acc, marker='s', label=model_name,
                    color=color, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Test Accuracy (Epoch Level)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed training curves saved to {output_path}")
    plt.close()

def main():
    # Path to training curves JSON
    json_path = '/Users/ikuhiruryou/Downloads/formal_veri/auto_LiRPA/examples/vision/training_curves.json'
    
    # Load data
    curves_data = load_training_curves(json_path)
    
    # Plot basic curves
    output_path1 = '/Users/ikuhiruryou/Downloads/formal_veri/auto_LiRPA/examples/vision/training_curves.png'
    plot_training_curves(curves_data, output_path1)
    
    # Plot detailed curves
    output_path2 = '/Users/ikuhiruryou/Downloads/formal_veri/auto_LiRPA/examples/vision/training_curves_detailed.png'
    plot_detailed_training_curves(curves_data, output_path2)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

