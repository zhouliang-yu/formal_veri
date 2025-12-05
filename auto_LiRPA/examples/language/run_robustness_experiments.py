"""
Language Model Robustness Experiment Script
运行多种鲁棒性实验并生成可视化结果
"""
import argparse
import subprocess
import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import re

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class RobustnessExperimentRunner:
    def __init__(self, base_dir='./experiments', train_script='train.py', 
                 pretrained_model=None, train_model_first=False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.train_script = train_script
        self.pretrained_model = pretrained_model
        self.train_model_first = train_model_first
        self.trained_model_path = None
        self.results = defaultdict(dict)
        self.experiment_log = []
        
    def train_model_if_needed(self, model_type='transformer', num_epochs=5):
        """如果需要，先训练模型"""
        if self.trained_model_path and Path(self.trained_model_path).exists():
            print(f"Using existing trained model: {self.trained_model_path}")
            return self.trained_model_path
        
        if self.pretrained_model and Path(self.pretrained_model).exists():
            print(f"Using pretrained model: {self.pretrained_model}")
            return self.pretrained_model
        
        if not self.train_model_first:
            print("\n" + "="*60)
            print("WARNING: No pretrained model provided and train_model_first=False")
            print("Experiments will use randomly initialized model (poor results expected)")
            print("="*60 + "\n")
            return None
        
        print("\n" + "="*60)
        print("Training model first...")
        print("="*60)
        
        train_dir = self.base_dir / 'trained_model'
        train_dir.mkdir(exist_ok=True)
        
        cmd = ['python', self.train_script, '--train', '--device', 'cpu',
               '--model', model_type, '--num_epochs', str(num_epochs),
               '--dir', str(train_dir)]
        
        print(f"Training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 查找最新的checkpoint
            checkpoints = list(train_dir.glob('ckpt_*'))
            if checkpoints:
                # 按epoch排序，取最新的
                checkpoints.sort(key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)
                self.trained_model_path = str(checkpoints[-1])
                print(f"Model trained and saved to: {self.trained_model_path}")
                return self.trained_model_path
            else:
                print("Warning: Training completed but no checkpoint found")
                return None
        else:
            print(f"Training failed: {result.stderr}")
            return None
    
    def run_experiment(self, name, **kwargs):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")
        
        # 构建命令
        cmd = ['python', self.train_script]
        # 默认使用CPU（如果没有指定device）
        if 'device' not in kwargs:
            cmd.extend(['--device', 'cpu'])
        
        # 如果提供了训练好的模型，使用它
        # 注意：需要在创建BoundedModule之前加载，所以load参数应该在model参数之后
        model_path = self.trained_model_path or self.pretrained_model
        if model_path and 'load' not in kwargs and Path(model_path).exists():
            # 确保load参数在model参数之后
            if 'model' in kwargs:
                # 找到model参数的位置，在它之后插入load
                cmd_list = list(cmd)
                model_idx = None
                for i, arg in enumerate(cmd_list):
                    if arg == '--model':
                        model_idx = i + 2  # model参数值之后
                        break
                if model_idx:
                    cmd_list.insert(model_idx, '--load')
                    cmd_list.insert(model_idx + 1, model_path)
                    cmd = cmd_list
                else:
                    cmd.extend(['--load', model_path])
            else:
                cmd.extend(['--load', model_path])
        
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                cmd.append(f'--{key}')
            elif not isinstance(value, bool):
                cmd.append(f'--{key}')
                cmd.append(str(value))
        
        # 设置实验目录
        exp_dir = self.base_dir / name
        exp_dir.mkdir(exist_ok=True)
        cmd.extend(['--dir', str(exp_dir)])
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Output directory: {exp_dir}")
        
        # 运行实验
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            elapsed_time = time.time() - start_time
            
            # 解析结果
            output = result.stdout + result.stderr
            metrics = self.parse_output(output)
            
            self.results[name] = {
                'metrics': metrics,
                'elapsed_time': elapsed_time,
                'success': result.returncode == 0,
                'output': output[:1000]  # 保存前1000字符
            }
            
            # 保存日志
            log_file = exp_dir / 'experiment_log.txt'
            with open(log_file, 'w') as f:
                f.write(output)
            
            # 如果从输出中没有解析到结果，尝试从日志文件读取
            if not metrics and result.returncode == 0:
                # 尝试读取train.py生成的日志
                train_log = exp_dir / 'log' / 'train.log'
                if train_log.exists():
                    log_metrics = self.parse_log_file(train_log)
                    if log_metrics:
                        metrics.update(log_metrics)
            
            print(f"Experiment completed in {elapsed_time:.2f} seconds")
            if metrics:
                print(f"Results: {metrics}")
            else:
                print("Warning: Could not parse metrics from output")
            
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"Experiment timed out after 1 hour")
            self.results[name] = {
                'metrics': {},
                'elapsed_time': 3600,
                'success': False,
                'error': 'Timeout'
            }
            return {}
        except Exception as e:
            print(f"Error running experiment: {e}")
            self.results[name] = {
                'metrics': {},
                'elapsed_time': 0,
                'success': False,
                'error': str(e)
            }
            return {}
    
    def parse_output(self, output):
        """从输出中解析指标"""
        metrics = {}
        
        # 解析验证结果（budget相关的准确率）- 格式: "budget 1 acc_rob 0.xxx"
        budget_pattern = r'budget\s+(\d+)\s+acc_rob\s+([\d.]+)'
        budget_matches = re.findall(budget_pattern, output)
        if budget_matches:
            metrics['budget_results'] = {
                int(budget): float(acc) 
                for budget, acc in budget_matches
            }
            # 使用最后一个budget的结果作为总体robust accuracy
            if budget_matches:
                metrics['robust_accuracy'] = float(budget_matches[-1][1])
        
        # 解析最终准确率 - 多种可能的格式
        acc_patterns = [
            r'acc[_\s]+rob[_\s]*[:=]\s*([\d.]+)',  # acc_rob: 0.xxx
            r'acc_rob\s+([\d.]+)',                  # acc_rob 0.xxx
            r'robust[_\s]+acc[_\s]*[:=]\s*([\d.]+)', # robust_acc: 0.xxx
            r'acc_rob=([\d.]+)',                    # acc_rob=0.xxx (from log output)
        ]
        for pattern in acc_patterns:
            acc_matches = re.findall(pattern, output, re.IGNORECASE)
            if acc_matches:
                # 取最后一个匹配的值
                metrics['robust_accuracy'] = float(acc_matches[-1])
                break
        
        # 解析损失
        loss_patterns = [
            r'loss[_\s]+rob[_\s]*[:=]\s*([\d.]+)',  # loss_rob: 0.xxx
            r'loss_rob\s+([\d.]+)',                  # loss_rob 0.xxx
            r'robust[_\s]+loss[_\s]*[:=]\s*([\d.]+)', # robust_loss: 0.xxx
            r'loss_rob=([\d.]+)',                    # loss_rob=0.xxx (from log output)
        ]
        for pattern in loss_patterns:
            loss_matches = re.findall(pattern, output, re.IGNORECASE)
            if loss_matches:
                metrics['robust_loss'] = float(loss_matches[-1])
                break
        
        # 解析列表格式的结果: [0.0060406370126304225]
        list_pattern = r'\[([\d.]+\d*)\]'
        list_matches = re.findall(list_pattern, output)
        if list_matches and 'robust_accuracy' not in metrics:
            # 如果找到了列表格式，使用最后一个值作为robust accuracy
            metrics['robust_accuracy'] = float(list_matches[-1])
        
        return metrics
    
    def parse_log_file(self, log_file):
        """从日志文件中解析结果"""
        if not os.path.exists(log_file):
            return {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            return self.parse_output(content)
        except Exception as e:
            print(f"Warning: Could not parse log file {log_file}: {e}")
            return {}
    
    def load_existing_results(self, results_dir):
        """从已有实验结果目录加载数据"""
        results_dir = Path(results_dir)
        if not results_dir.exists():
            print(f"Error: Results directory {results_dir} does not exist")
            return
        
        print(f"\nLoading existing results from {results_dir}")
        
        # 遍历所有实验目录
        for exp_dir in results_dir.iterdir():
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                log_file = exp_dir / 'experiment_log.txt'
                train_log = exp_dir / 'log' / 'train.log'
                
                # 尝试解析日志文件
                metrics = {}
                if log_file.exists():
                    metrics = self.parse_log_file(log_file)
                elif train_log.exists():
                    metrics = self.parse_log_file(train_log)
                
                # 检查实验是否成功（通过检查是否有模型文件或日志）
                success = (exp_dir / 'log').exists() or len(metrics) > 0
                
                self.results[exp_name] = {
                    'metrics': metrics,
                    'elapsed_time': 0,  # 无法从已有结果获取
                    'success': success,
                    'output': ''
                }
        
        print(f"Loaded {len(self.results)} experiments")
    
    def run_method_comparison(self, eps_values=[0.5, 1.0, 1.5], budget=6):
        """对比不同验证方法
        
        支持的方法：
        - IBP: Interval Bound Propagation
        - IBP+backward: IBP + 后向传播
        - IBP+backward_train: 训练时使用 IBP+backward
        - forward: 前向传播边界
        - forward+backward: 前向+后向边界
        """
        methods = ['IBP', 'IBP+backward', 'IBP+backward_train', 'forward', 'forward+backward']
        
        print("\n" + "="*60)
        print("Method Comparison Experiments")
        print("="*60)
        
        for method in methods:
            for eps in eps_values:
                name = f"method_{method}_eps_{eps}_budget_{budget}"
                self.run_experiment(
                    name,
                    robust=True,
                    method=method,
                    eps=eps,
                    budget=budget,
                    model='transformer'
                )
    
    def run_budget_sweep(self, method='IBP', eps=1.0, budgets=[1, 2, 3, 4, 5, 6]):
        """不同budget下的鲁棒性分析"""
        print("\n" + "="*60)
        print("Budget Sweep Experiments")
        print("="*60)
        
        for budget in budgets:
            name = f"budget_sweep_method_{method}_eps_{eps}_budget_{budget}"
            self.run_experiment(
                name,
                robust=True,
                method=method,
                eps=eps,
                budget=budget,
                model='transformer'
            )
    
    def run_eps_sweep(self, method='IBP', budget=6, eps_values=[0.5, 1.0, 1.5, 2.0]):
        """不同epsilon下的鲁棒性分析"""
        print("\n" + "="*60)
        print("Epsilon Sweep Experiments")
        print("="*60)
        
        for eps in eps_values:
            name = f"eps_sweep_method_{method}_eps_{eps}_budget_{budget}"
            self.run_experiment(
                name,
                robust=True,
                method=method,
                eps=eps,
                budget=budget,
                model='transformer'
            )
    
    def run_model_comparison(self, method='IBP', eps=1.0, budget=6):
        """对比不同模型架构"""
        models = ['transformer', 'lstm']
        
        print("\n" + "="*60)
        print("Model Architecture Comparison")
        print("="*60)
        
        for model in models:
            name = f"model_{model}_method_{method}_eps_{eps}_budget_{budget}"
            self.run_experiment(
                name,
                robust=True,
                method=method,
                eps=eps,
                budget=budget,
                model=model
            )
    
    def visualize_method_comparison(self, save_path='method_comparison.png'):
        """可视化方法对比结果"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 提取数据
        method_data = defaultdict(lambda: {'eps': [], 'acc': []})
        
        for exp_name, result in self.results.items():
            if 'method_' in exp_name and result['success']:
                # 解析实验名称，使用正则表达式处理包含加号的方法名
                # 格式: method_{method}_eps_{eps}_budget_{budget}
                # 方法名可能包含加号，需要匹配到 _eps_ 之前的所有内容
                match = re.match(r'method_(.+?)_eps_([\d.]+)_budget_(\d+)', exp_name)
                if match:
                    method = match.group(1)
                    eps = float(match.group(2))
                    
                    metrics = result['metrics']
                    if 'robust_accuracy' in metrics:
                        method_data[method]['eps'].append(eps)
                        method_data[method]['acc'].append(metrics['robust_accuracy'])
        
        # 绘制方法对比（按epsilon）
        ax1 = axes[0]
        colors = {
            'IBP': '#d62728',
            'IBP+backward': '#ff7f0e',
            'IBP+backward_train': '#9467bd',  # 紫色
            'forward': '#2ca02c',
            'forward+backward': '#1f77b4'
        }
        
        has_data = False
        for method, data in method_data.items():
            if data['eps']:
                has_data = True
                sorted_data = sorted(zip(data['eps'], data['acc']))
                eps_sorted, acc_sorted = zip(*sorted_data)
                ax1.plot(eps_sorted, acc_sorted, 
                        marker='o', label=method, 
                        color=colors.get(method, None),
                        linewidth=2.5, markersize=8)
        
        ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Robust Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Method Comparison: Robust Accuracy vs Epsilon', fontsize=14, fontweight='bold')
        if has_data:
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No data available', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=14, style='italic')
        ax1.grid(True, alpha=0.3)
        
        # 绘制budget对比
        ax2 = axes[1]
        budget_data = defaultdict(lambda: {'budget': [], 'acc': []})
        
        for exp_name, result in self.results.items():
            if 'budget_sweep' in exp_name and result['success']:
                # 解析budget值：budget_sweep_method_IBP_eps_1.0_budget_3
                # 找到最后一个budget后面的数字
                parts = exp_name.split('_')
                try:
                    # 找到所有'budget'的位置，取最后一个
                    budget_indices = [i for i, part in enumerate(parts) if part == 'budget']
                    if budget_indices:
                        budget_idx = budget_indices[-1] + 1
                        if budget_idx < len(parts):
                            budget = int(parts[budget_idx])
                            metrics = result['metrics']
                            if 'robust_accuracy' in metrics:
                                budget_data['all']['budget'].append(budget)
                                budget_data['all']['acc'].append(metrics['robust_accuracy'])
                except (ValueError, IndexError):
                    # 如果解析失败，尝试从metrics中获取budget_results
                    metrics = result['metrics']
                    if 'budget_results' in metrics:
                        for budget, acc in metrics['budget_results'].items():
                            budget_data['all']['budget'].append(budget)
                            budget_data['all']['acc'].append(acc)
        
        if budget_data['all']['budget']:
            sorted_data = sorted(zip(budget_data['all']['budget'], budget_data['all']['acc']))
            budget_sorted, acc_sorted = zip(*sorted_data)
            ax2.plot(budget_sorted, acc_sorted, 
                    marker='s', color='#9467bd',
                    linewidth=2.5, markersize=8)
            ax2.set_xlabel('Budget (Max Word Replacements)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Robust Accuracy', fontsize=12, fontweight='bold')
            ax2.set_title('Robust Accuracy vs Budget', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(budget_sorted)
        else:
            ax2.text(0.5, 0.5, 'No data available', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=14, style='italic')
            ax2.set_xlabel('Budget (Max Word Replacements)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Robust Accuracy', fontsize=12, fontweight='bold')
            ax2.set_title('Robust Accuracy vs Budget', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Method comparison plot saved to {save_path}")
        plt.close()
    
    def visualize_budget_results(self, save_path='budget_results.png'):
        """可视化budget结果（如果有详细budget数据）"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 收集所有budget结果
        all_budget_data = defaultdict(list)
        
        for exp_name, result in self.results.items():
            if result['success'] and 'budget_results' in result['metrics']:
                budget_results = result['metrics']['budget_results']
                for budget, acc in budget_results.items():
                    all_budget_data[budget].append(acc)
        
        if all_budget_data:
            budgets = sorted(all_budget_data.keys())
            means = [np.mean(all_budget_data[b]) for b in budgets]
            stds = [np.std(all_budget_data[b]) for b in budgets]
            
            ax.errorbar(budgets, means, yerr=stds, 
                       marker='o', capsize=5, capthick=2,
                       linewidth=2.5, markersize=10,
                       color='#2ca02c')
            ax.set_xlabel('Budget (Max Word Replacements)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Robust Accuracy', fontsize=12, fontweight='bold')
            ax.set_title('Robust Accuracy Across Different Budgets', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(budgets)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Budget results plot saved to {save_path}")
        else:
            print("No budget results found for visualization")
        
        plt.close()
    
    def visualize_model_comparison(self, save_path='model_comparison.png'):
        """可视化模型架构对比"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        model_data = defaultdict(list)
        
        for exp_name, result in self.results.items():
            if 'model_' in exp_name and result['success']:
                parts = exp_name.split('_')
                model_idx = parts.index('model') + 1
                if model_idx < len(parts):
                    model = parts[model_idx]
                    metrics = result['metrics']
                    if 'robust_accuracy' in metrics:
                        model_data[model].append(metrics['robust_accuracy'])
        
        if model_data:
            models = list(model_data.keys())
            means = [np.mean(model_data[m]) for m in models]
            stds = [np.std(model_data[m]) for m in models]
            
            x_pos = np.arange(len(models))
            bars = ax.bar(x_pos, means, yerr=stds, 
                         capsize=5, alpha=0.7,
                         color=['#1f77b4', '#ff7f0e'])
            
            ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
            ax.set_ylabel('Robust Accuracy', fontsize=12, fontweight='bold')
            ax.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.01, f'{mean:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        else:
            print("No model comparison data found")
        
        plt.close()
    
    def generate_summary_report(self, save_path='experiment_summary.json'):
        """生成实验总结报告"""
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results.values() if r['success']),
            'failed_experiments': sum(1 for r in self.results.values() if not r['success']),
            'total_time': sum(r['elapsed_time'] for r in self.results.values()),
            'experiments': {}
        }
        
        for name, result in self.results.items():
            summary['experiments'][name] = {
                'success': result['success'],
                'elapsed_time': result['elapsed_time'],
                'metrics': result['metrics']
            }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary report saved to {save_path}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        print(f"Total time: {summary['total_time']/60:.2f} minutes")
        
        return summary
    
    def create_comprehensive_visualization(self, save_dir='./visualizations'):
        """创建综合可视化"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        # 方法对比
        self.visualize_method_comparison(
            save_path=str(save_dir / 'method_comparison.png')
        )
        
        # Budget结果
        self.visualize_budget_results(
            save_path=str(save_dir / 'budget_results.png')
        )
        
        # 模型对比
        self.visualize_model_comparison(
            save_path=str(save_dir / 'model_comparison.png')
        )
        
        # 创建综合指标报告图
        self.visualize_metrics_summary(
            save_path=str(save_dir / 'metrics_summary.png')
        )
        
        print(f"\nAll visualizations saved to {save_dir}")
    
    def visualize_metrics_summary(self, save_path='metrics_summary.png'):
        """创建指标摘要可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 所有实验的鲁棒准确率对比
        ax1 = axes[0, 0]
        exp_names = []
        acc_values = []
        loss_values = []
        
        for exp_name, result in sorted(self.results.items()):
            if result['success'] and 'robust_accuracy' in result['metrics']:
                # 简化实验名称用于显示
                short_name = exp_name.replace('method_', '').replace('_eps_', ' ε=').replace('_budget_', ' B=')
                exp_names.append(short_name[:30])  # 限制长度
                acc_values.append(result['metrics']['robust_accuracy'])
                if 'robust_loss' in result['metrics']:
                    loss_values.append(result['metrics']['robust_loss'])
                else:
                    loss_values.append(0)
        
        if exp_names:
            x_pos = range(len(exp_names))
            bars = ax1.bar(x_pos, acc_values, alpha=0.7, color='#2ca02c')
            ax1.set_xlabel('Experiment', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Robust Accuracy', fontsize=10, fontweight='bold')
            ax1.set_title('Robust Accuracy Across All Experiments', fontsize=12, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, acc) in enumerate(zip(bars, acc_values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 鲁棒损失对比
        ax2 = axes[0, 1]
        if loss_values and any(l > 0 for l in loss_values):
            bars = ax2.bar(x_pos, loss_values, alpha=0.7, color='#d62728')
            ax2.set_xlabel('Experiment', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Robust Loss', fontsize=10, fontweight='bold')
            ax2.set_title('Robust Loss Across All Experiments', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, (bar, loss) in enumerate(zip(bars, loss_values)):
                if loss > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 成功/失败统计
        ax3 = axes[1, 0]
        success_count = sum(1 for r in self.results.values() if r['success'])
        fail_count = len(self.results) - success_count
        if success_count + fail_count > 0:
            ax3.pie([success_count, fail_count], 
                   labels=[f'Success ({success_count})', f'Failed ({fail_count})'],
                   autopct='%1.1f%%', startangle=90,
                   colors=['#2ca02c', '#d62728'])
            ax3.set_title('Experiment Success Rate', fontsize=12, fontweight='bold')
        
        # 4. 实验耗时分布
        ax4 = axes[1, 1]
        times = [r['elapsed_time'] for r in self.results.values() if r['success']]
        if times:
            ax4.hist(times, bins=10, alpha=0.7, color='#1f77b4', edgecolor='black')
            ax4.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Number of Experiments', fontsize=10, fontweight='bold')
            ax4.set_title('Experiment Duration Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axvline(sum(times)/len(times), color='r', linestyle='--', 
                       label=f'Mean: {sum(times)/len(times):.2f}s')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics summary plot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Language Model Robustness Experiments')
    parser.add_argument('--train_script', type=str, default='train.py',
                       help='Path to train.py script')
    parser.add_argument('--experiment_dir', type=str, default='./experiments',
                       help='Directory to save experiment results')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    
    # 实验类型选择
    parser.add_argument('--run_method_comparison', action='store_true',
                       help='Run method comparison experiments')
    parser.add_argument('--run_budget_sweep', action='store_true',
                       help='Run budget sweep experiments')
    parser.add_argument('--run_eps_sweep', action='store_true',
                       help='Run epsilon sweep experiments')
    parser.add_argument('--run_model_comparison', action='store_true',
                       help='Run model comparison experiments')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--load_existing', type=str, default=None,
                       help='Load existing experiment results from directory')
    parser.add_argument('--visualize_only', action='store_true',
                       help='Only generate visualizations from existing results')
    
    # 实验参数
    parser.add_argument('--eps_values', type=float, nargs='+', 
                       default=[0.5, 1.0, 1.5],
                       help='Epsilon values for experiments')
    parser.add_argument('--budget_values', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6],
                       help='Budget values for experiments')
    parser.add_argument('--method', type=str, default='IBP',
                       help='Default method for sweep experiments')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--train_model_first', action='store_true',
                       help='Train model before running experiments')
    parser.add_argument('--train_epochs', type=int, default=5,
                       help='Number of epochs to train (if train_model_first)')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = RobustnessExperimentRunner(
        base_dir=args.experiment_dir,
        train_script=args.train_script,
        pretrained_model=args.pretrained_model,
        train_model_first=args.train_model_first
    )
    
    # 如果需要，先训练模型
    if args.train_model_first or (not args.pretrained_model and args.run_all):
        model_type = 'transformer'  # 默认使用transformer
        runner.train_model_if_needed(model_type=model_type, num_epochs=args.train_epochs)
    
    # 如果指定了加载已有结果或仅可视化
    if args.load_existing:
        runner.load_existing_results(args.load_existing)
        runner.generate_summary_report(
            save_path=str(Path(args.experiment_dir) / 'experiment_summary.json')
        )
        runner.create_comprehensive_visualization(
            save_dir=args.visualization_dir
        )
        print("\n" + "="*60)
        print("Visualization from existing results completed!")
        print("="*60)
        return
    
    if args.visualize_only:
        # 尝试从默认实验目录加载
        runner.load_existing_results(args.experiment_dir)
        runner.create_comprehensive_visualization(
            save_dir=args.visualization_dir
        )
        print("\n" + "="*60)
        print("Visualization completed!")
        print("="*60)
        return
    
    # 检查是否指定了任何实验
    has_experiments = (args.run_all or args.run_method_comparison or 
                      args.run_budget_sweep or args.run_eps_sweep or 
                      args.run_model_comparison)
    
    if not has_experiments:
        print("\n" + "="*60)
        print("No experiments specified!")
        print("="*60)
        print("\nAvailable options:")
        print("  --run_all                    : Run all experiments")
        print("  --run_method_comparison      : Compare different verification methods")
        print("  --run_budget_sweep          : Sweep over different budget values")
        print("  --run_eps_sweep             : Sweep over different epsilon values")
        print("  --run_model_comparison       : Compare different model architectures")
        print("\nExample usage:")
        print("  python run_robustness_experiments.py --run_all")
        print("  python run_robustness_experiments.py --run_method_comparison --eps_values 0.5 1.0 1.5")
        print("\nRunning a quick demo with budget sweep (small scale)...")
        print("="*60 + "\n")
        # 运行一个小规模的演示实验
        args.run_budget_sweep = True
        args.budget_values = [1, 2, 3]  # 减少实验数量用于演示
    
    # 运行实验
    if args.run_all or args.run_method_comparison:
        print("\n>>> Starting Method Comparison Experiments")
        runner.run_method_comparison(
            eps_values=args.eps_values,
            budget=6
        )
    
    if args.run_all or args.run_budget_sweep:
        print("\n>>> Starting Budget Sweep Experiments")
        runner.run_budget_sweep(
            method=args.method,
            eps=1.0,
            budgets=args.budget_values
        )
    
    if args.run_all or args.run_eps_sweep:
        print("\n>>> Starting Epsilon Sweep Experiments")
        runner.run_eps_sweep(
            method=args.method,
            budget=6,
            eps_values=args.eps_values
        )
    
    if args.run_all or args.run_model_comparison:
        print("\n>>> Starting Model Comparison Experiments")
        runner.run_model_comparison(
            method=args.method,
            eps=1.0,
            budget=6
        )
    
    # 生成报告和可视化
    runner.generate_summary_report(
        save_path=str(Path(args.experiment_dir) / 'experiment_summary.json')
    )
    runner.create_comprehensive_visualization(
        save_dir=args.visualization_dir
    )
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()