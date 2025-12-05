#!/usr/bin/env python3
"""
最小规模评估脚本 - 在训练目录中直接运行验证
"""
import subprocess
import sys
from pathlib import Path

def run_evaluation(train_dir, method='IBP', eps=1.0, budgets=[1, 2, 3]):
    """在训练目录中运行评估"""
    results = []
    
    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Running evaluation: method={method}, eps={eps}, budget={budget}")
        print(f"{'='*60}")
        
        cmd = [
            'python', 'train.py',
            '--device', 'cpu',
            '--robust',
            '--method', method,
            '--eps', str(eps),
            '--budget', str(budget),
            '--model', 'transformer',
            '--dir', train_dir,
            '--checkpoint', '2'  # 使用第2个epoch的模型
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # 解析结果
        output = result.stdout + result.stderr
        for line in output.split('\n'):
            if 'acc_rob' in line and 'budget' in line:
                print(f"  {line.strip()}")
                try:
                    parts = line.split()
                    budget_val = int(parts[parts.index('budget') + 1])
                    acc_val = float(parts[parts.index('acc_rob') + 1])
                    results.append((budget_val, acc_val))
                except:
                    pass
    
    return results

if __name__ == '__main__':
    train_dir = 'trained_model_minimal'
    
    print("\n" + "="*60)
    print("Running Minimal Evaluation")
    print("="*60)
    
    # 运行评估
    results = run_evaluation(train_dir, method='IBP', eps=1.0, budgets=[1, 2, 3])
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for budget, acc in results:
        print(f"Budget {budget}: Robust Accuracy = {acc:.4f}")
    
    print("\nNote: These results use the model from training directory.")
    print("The model checkpoint loading has compatibility issues,")
    print("but the model state is preserved in the training directory.")

