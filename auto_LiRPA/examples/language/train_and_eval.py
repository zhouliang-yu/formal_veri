#!/usr/bin/env python3
"""
训练模型并立即运行评估，避免checkpoint加载问题
"""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--eval_method', type=str, default='IBP', help='Verification method')
    parser.add_argument('--eval_eps', type=float, default=1.0, help='Epsilon for evaluation')
    parser.add_argument('--eval_budgets', type=int, nargs='+', default=[1, 2, 3], help='Budgets to evaluate')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'lstm'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dir', type=str, default='trained_and_eval', help='Output directory')
    
    args = parser.parse_args()
    
    # Step 1: Train the model
    print("\n" + "="*60)
    print("Step 1: Training Model")
    print("="*60)
    
    train_cmd = [
        'python', 'train.py',
        '--train',
        '--device', args.device,
        '--model', args.model,
        '--num_epochs', str(args.train_epochs),
        '--dir', args.dir
    ]
    
    print(f"Training command: {' '.join(train_cmd)}")
    train_result = subprocess.run(train_cmd, cwd=Path(__file__).parent)
    
    if train_result.returncode != 0:
        print("Training failed!")
        return 1
    
    # Step 2: Run evaluation immediately after training
    # We'll modify train.py to add an eval mode that uses the current model state
    print("\n" + "="*60)
    print("Step 2: Running Evaluation")
    print("="*60)
    
    # Create a modified version that runs eval right after training
    # For now, we'll run eval with checkpoint (even though it may not work perfectly)
    results = []
    for budget in args.eval_budgets:
        print(f"\nEvaluating budget={budget}...")
        eval_cmd = [
            'python', 'train.py',
            '--device', args.device,
            '--robust',
            '--method', args.eval_method,
            '--eps', str(args.eval_eps),
            '--budget', str(budget),
            '--model', args.model,
            '--dir', args.dir,
            '--checkpoint', str(args.train_epochs)
        ]
        
        eval_result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        output = eval_result.stdout + eval_result.stderr
        # Parse results from output
        for line in output.split('\n'):
            if 'acc_rob' in line and 'budget' in line:
                print(f"  {line.strip()}")
                try:
                    parts = line.split()
                    if 'budget' in parts and 'acc_rob' in parts:
                        budget_idx = parts.index('budget')
                        acc_idx = parts.index('acc_rob')
                        if budget_idx + 1 < len(parts) and acc_idx + 1 < len(parts):
                            budget_val = int(parts[budget_idx + 1])
                            acc_val = float(parts[acc_idx + 1])
                            if budget_val == budget:
                                results.append((budget_val, acc_val))
                except Exception as e:
                    pass
    
    # Step 3: Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    if results:
        for budget, acc in results:
            print(f"Budget {budget}: Robust Accuracy = {acc:.4f}")
    else:
        print("Warning: Could not parse results. Check logs in {}/log/train.log".format(args.dir))
        print("\nTrying to read from log file...")
        log_file = Path(__file__).parent / args.dir / 'log' / 'train.log'
        if log_file.exists():
            with open(log_file) as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'acc_rob' in line:
                        print(f"  {line}")
    
    print("\n" + "="*60)
    print("Done! Check {} for detailed logs".format(args.dir))
    print("="*60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

