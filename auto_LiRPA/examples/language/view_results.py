#!/usr/bin/env python3
"""
查看和展示实验结果指标的工具脚本
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_experiment_summary(summary_path):
    """加载实验摘要"""
    with open(summary_path, 'r') as f:
        return json.load(f)

def format_metrics_table(experiments):
    """格式化指标为表格"""
    rows = []
    for exp_name, exp_data in experiments.items():
        if exp_data.get('success', False):
            metrics = exp_data.get('metrics', {})
            row = {
                'Experiment': exp_name,
                'Success': '✓',
                'Time (s)': f"{exp_data.get('elapsed_time', 0):.2f}",
            }
            
            # 添加各种指标
            if 'robust_accuracy' in metrics:
                row['Robust Acc'] = f"{metrics['robust_accuracy']:.4f}"
            else:
                row['Robust Acc'] = 'N/A'
            
            if 'robust_loss' in metrics:
                row['Robust Loss'] = f"{metrics['robust_loss']:.4f}"
            else:
                row['Robust Loss'] = 'N/A'
            
            if 'budget_results' in metrics:
                budget_str = ', '.join([f"B{k}={v:.4f}" for k, v in sorted(metrics['budget_results'].items())])
                row['Budget Results'] = budget_str
            else:
                row['Budget Results'] = 'N/A'
            
            rows.append(row)
        else:
            rows.append({
                'Experiment': exp_name,
                'Success': '✗',
                'Time (s)': f"{exp_data.get('elapsed_time', 0):.2f}",
                'Robust Acc': 'N/A',
                'Robust Loss': 'N/A',
                'Budget Results': 'N/A',
                'Error': exp_data.get('error', 'Unknown error')
            })
    
    return pd.DataFrame(rows)

def print_detailed_results(summary):
    """打印详细结果"""
    print("\n" + "="*80)
    print("实验摘要")
    print("="*80)
    print(f"总实验数: {summary['total_experiments']}")
    print(f"成功实验: {summary['successful_experiments']}")
    print(f"失败实验: {summary['failed_experiments']}")
    print(f"总耗时: {summary['total_time']/60:.2f} 分钟")
    
    print("\n" + "="*80)
    print("详细实验结果")
    print("="*80)
    
    # 使用pandas DataFrame来格式化输出
    df = format_metrics_table(summary['experiments'])
    print(df.to_string(index=False))
    
    # 按实验类型分组统计
    print("\n" + "="*80)
    print("按实验类型统计")
    print("="*80)
    
    type_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'acc_values': []})
    
    for exp_name, exp_data in summary['experiments'].items():
        if 'method_' in exp_name:
            exp_type = 'Method Comparison'
        elif 'budget_sweep' in exp_name:
            exp_type = 'Budget Sweep'
        elif 'eps_sweep' in exp_name:
            exp_type = 'Epsilon Sweep'
        elif 'model_' in exp_name:
            exp_type = 'Model Comparison'
        else:
            exp_type = 'Other'
        
        type_stats[exp_type]['total'] += 1
        if exp_data.get('success', False):
            type_stats[exp_type]['success'] += 1
            metrics = exp_data.get('metrics', {})
            if 'robust_accuracy' in metrics:
                type_stats[exp_type]['acc_values'].append(metrics['robust_accuracy'])
    
    for exp_type, stats in type_stats.items():
        print(f"\n{exp_type}:")
        print(f"  总实验数: {stats['total']}")
        print(f"  成功数: {stats['success']}")
        if stats['acc_values']:
            avg_acc = sum(stats['acc_values']) / len(stats['acc_values'])
            print(f"  平均鲁棒准确率: {avg_acc:.4f}")
            print(f"  准确率范围: {min(stats['acc_values']):.4f} - {max(stats['acc_values']):.4f}")

def export_to_csv(summary, output_path):
    """导出结果到CSV"""
    df = format_metrics_table(summary['experiments'])
    df.to_csv(output_path, index=False)
    print(f"\n结果已导出到: {output_path}")

def print_budget_analysis(summary):
    """分析budget实验结果"""
    print("\n" + "="*80)
    print("Budget分析")
    print("="*80)
    
    budget_data = defaultdict(list)
    
    for exp_name, exp_data in summary['experiments'].items():
        if 'budget_sweep' in exp_name and exp_data.get('success', False):
            metrics = exp_data.get('metrics', {})
            if 'budget_results' in metrics:
                for budget, acc in metrics['budget_results'].items():
                    budget_data[int(budget)].append(acc)
    
    if budget_data:
        print("\nBudget -> 平均鲁棒准确率:")
        for budget in sorted(budget_data.keys()):
            avg_acc = sum(budget_data[budget]) / len(budget_data[budget])
            print(f"  Budget {budget}: {avg_acc:.4f} (基于 {len(budget_data[budget])} 个实验)")

def print_method_comparison(summary):
    """方法对比分析"""
    print("\n" + "="*80)
    print("方法对比分析")
    print("="*80)
    
    method_data = defaultdict(lambda: {'acc_values': [], 'loss_values': []})
    
    for exp_name, exp_data in summary['experiments'].items():
        if 'method_' in exp_name and exp_data.get('success', False):
            # 解析方法名
            parts = exp_name.split('_')
            if 'method' in parts:
                method_idx = parts.index('method') + 1
                if method_idx < len(parts):
                    method = parts[method_idx]
                    metrics = exp_data.get('metrics', {})
                    if 'robust_accuracy' in metrics:
                        method_data[method]['acc_values'].append(metrics['robust_accuracy'])
                    if 'robust_loss' in metrics:
                        method_data[method]['loss_values'].append(metrics['robust_loss'])
    
    if method_data:
        print("\n方法 -> 平均指标:")
        for method, data in method_data.items():
            if data['acc_values']:
                avg_acc = sum(data['acc_values']) / len(data['acc_values'])
                print(f"  {method}:")
                print(f"    平均鲁棒准确率: {avg_acc:.4f}")
            if data['loss_values']:
                avg_loss = sum(data['loss_values']) / len(data['loss_values'])
                print(f"    平均鲁棒损失: {avg_loss:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='查看实验结果')
    parser.add_argument('--summary', type=str, default='experiments/experiment_summary.json',
                       help='实验摘要JSON文件路径')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='导出结果到CSV文件')
    parser.add_argument('--budget-analysis', action='store_true',
                       help='显示budget分析')
    parser.add_argument('--method-comparison', action='store_true',
                       help='显示方法对比分析')
    
    args = parser.parse_args()
    
    # 加载摘要
    if not os.path.exists(args.summary):
        print(f"错误: 找不到摘要文件 {args.summary}")
        print("请先运行实验: python run_robustness_experiments.py")
        return
    
    summary = load_experiment_summary(args.summary)
    
    # 打印详细结果
    print_detailed_results(summary)
    
    # Budget分析
    if args.budget_analysis:
        print_budget_analysis(summary)
    
    # 方法对比
    if args.method_comparison:
        print_method_comparison(summary)
    
    # 导出CSV
    if args.export_csv:
        export_to_csv(summary, args.export_csv)
    
    print("\n" + "="*80)
    print("提示: 使用 --budget-analysis 查看budget分析")
    print("     使用 --method-comparison 查看方法对比")
    print("     使用 --export-csv <file> 导出到CSV")
    print("="*80)

if __name__ == '__main__':
    main()

