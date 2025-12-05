#!/usr/bin/env python3
"""
从训练日志中提取loss数据并绘制折线图
"""
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def parse_log_file(log_file_path):
    """解析日志文件，提取loss数据"""
    epochs = []
    train_loss = []
    dev_loss = []
    test_loss = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # 从后往前解析，只保留最新的每个epoch的数据
    epoch_data = {}  # {epoch: {'train': loss, 'dev': loss, 'test': loss}}
    
    # 从后往前遍历，这样后面的数据会覆盖前面的
    for line in reversed(lines):
        # 提取训练loss（最后一个step）
        train_match = re.search(r'Epoch (\d+), train step (\d+)/(\d+):.*loss=([\d.]+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            step = int(train_match.group(2))
            total = int(train_match.group(3))
            loss = float(train_match.group(4))
            
            if step == total:  # 只保存最后一个step的数据
                if epoch not in epoch_data:
                    epoch_data[epoch] = {}
                epoch_data[epoch]['train'] = loss
        
        # 提取dev loss
        dev_match = re.search(r'Epoch (\d+), dev step.*loss=([\d.]+)', line)
        if dev_match:
            epoch = int(dev_match.group(1))
            loss = float(dev_match.group(2))
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            epoch_data[epoch]['dev'] = loss
        
        # 提取test loss
        test_match = re.search(r'Epoch (\d+), test step.*loss=([\d.]+)', line)
        if test_match:
            epoch = int(test_match.group(1))
            loss = float(test_match.group(2))
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            epoch_data[epoch]['test'] = loss
    
    # 按epoch排序并提取数据
    sorted_epochs = sorted(epoch_data.keys())
    for epoch in sorted_epochs:
        epochs.append(epoch)
        train_loss.append(epoch_data[epoch].get('train'))
        dev_loss.append(epoch_data[epoch].get('dev'))
        test_loss.append(epoch_data[epoch].get('test'))
    
    return epochs, train_loss, dev_loss, test_loss

def plot_loss(epochs, train_loss, dev_loss, test_loss, save_path='loss_curve.png'):
    """绘制loss折线图"""
    plt.figure(figsize=(12, 6))
    
    # 过滤None值并绘制训练loss
    valid_train = [(e, l) for e, l in zip(epochs, train_loss) if l is not None]
    if valid_train:
        epochs_train, losses_train = zip(*valid_train)
        plt.plot(epochs_train, losses_train, 'o-', label='Train Loss', linewidth=2.5, markersize=8, color='#1f77b4')
    
    # 绘制dev loss（如果有数据）
    # valid_dev = [(e, l) for e, l in zip(epochs, dev_loss) if l is not None]
    # if valid_dev:
    #     epochs_dev, losses_dev = zip(*valid_dev)
    #     plt.plot(epochs_dev, losses_dev, 's-', label='Dev Loss', linewidth=2.5, markersize=8, color='#ff7f0e')
    
    # 绘制test loss（如果有数据）
    # valid_test = [(e, l) for e, l in zip(epochs, test_loss) if l is not None]
    # if valid_test:
    #     epochs_test, losses_test = zip(*valid_test)
    #     plt.plot(epochs_test, losses_test, '^-', label='Test Loss', linewidth=2.5, markersize=8, color='#2ca02c')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度为整数
    plt.xticks(epochs)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss曲线图已保存为: {save_path}")
    
    # 打印数据摘要
    print(f"\n数据摘要:")
    print(f"Epochs: {len(epochs)}")
    valid_train = [x for x in train_loss if x is not None]
    if valid_train:
        print(f"Train Loss范围: {min(valid_train):.4f} - {max(valid_train):.4f}")
    if dev_loss:
        valid_dev = [x for x in dev_loss if x is not None]
        if valid_dev:
            print(f"Dev Loss范围: {min(valid_dev):.4f} - {max(valid_dev):.4f}")
    if test_loss:
        valid_test = [x for x in test_loss if x is not None]
        if valid_test:
            print(f"Test Loss范围: {min(valid_test):.4f} - {max(valid_test):.4f}")

if __name__ == '__main__':
    import sys
    
    # 默认日志文件路径
    log_file = 'trained_model/log/train.log'
    
    # 如果提供了命令行参数，使用该路径
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    # 输出文件路径
    output_file = 'loss_curve.png'
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"正在解析日志文件: {log_file}")
    epochs, train_loss, dev_loss, test_loss = parse_log_file(log_file)
    
    print(f"提取到 {len(epochs)} 个epoch的数据")
    
    plot_loss(epochs, train_loss, dev_loss, test_loss, output_file)

