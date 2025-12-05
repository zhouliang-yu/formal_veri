#!/usr/bin/env python3
"""
绘制不同epsilon值下的鲁棒准确率
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

def plot_epsilon_results():
    """绘制epsilon vs 鲁棒准确率的折线图"""
    # 数据
    epsilons = [0.05, 0.1, 0.2, 0.5, 1.0]
    robust_accs = [0.360, 0.121, 0.029, 0.010, 0.006]
    robust_losses = [2.05, 3.75, 5.03, 5.47, 5.52]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 鲁棒准确率 vs Epsilon
    ax1 = axes[0]
    ax1.plot(epsilons, robust_accs, 'o-', linewidth=2.5, markersize=10, color='#1f77b4')
    ax1.set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Robust Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('Robust Accuracy vs Epsilon', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.4])
    
    # 添加数值标签
    for i, (eps, acc) in enumerate(zip(epsilons, robust_accs)):
        ax1.annotate(f'{acc:.3f}', (eps, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # 2. 鲁棒损失 vs Epsilon
    ax2 = axes[1]
    ax2.plot(epsilons, robust_losses, 's-', linewidth=2.5, markersize=10, color='#d62728')
    ax2.set_xlabel('Epsilon (ε)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Robust Loss', fontsize=14, fontweight='bold')
    ax2.set_title('Robust Loss vs Epsilon', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (eps, loss) in enumerate(zip(epsilons, robust_losses)):
        ax2.annotate(f'{loss:.2f}', (eps, loss), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('epsilon_robustness_results.png', dpi=300, bbox_inches='tight')
    print("图表已保存为: epsilon_robustness_results.png")
    
    # 打印数据摘要
    print("\n数据摘要:")
    print(f"Epsilon范围: {min(epsilons)} - {max(epsilons)}")
    print(f"鲁棒准确率范围: {min(robust_accs):.3f} - {max(robust_accs):.3f}")
    print(f"鲁棒损失范围: {min(robust_losses):.2f} - {max(robust_losses):.2f}")

if __name__ == '__main__':
    plot_epsilon_results()

