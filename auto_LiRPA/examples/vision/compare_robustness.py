"""
Compare robustness of different training methods on MNIST using cnn_4layer model.
Trains 9 different models with different training configurations and evaluates their robustness.
Results are saved to a JSON file.
"""
import os
import json
import time
import random
import multiprocessing
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.eps_scheduler import LinearScheduler, FixedScheduler
from auto_LiRPA.utils import MultiAverageMeter
import sys
import os
# Add current directory to path for importing models
sys.path.insert(0, os.path.dirname(__file__))
import models

# Set random seeds for reproducibility
def set_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# Training configurations: 9 different experiments
EXPERIMENTS = [
    {"name": "natural", "bound_type": None, "eps": 0.0, "method": "natural"},
    {"name": "IBP_eps005", "bound_type": "IBP", "eps": 0.01, "method": "robust"},
    {"name": "IBP_eps01", "bound_type": "IBP", "eps": 0.02, "method": "robust"},
    {"name": "CROWN_IBP_eps01", "bound_type": "CROWN-IBP", "eps": 0.05, "method": "robust"},
    {"name": "CROWN_IBP_eps03", "bound_type": "CROWN-IBP", "eps": 0.1, "method": "robust"},
    {"name": "CROWN_eps01", "bound_type": "CROWN", "eps": 0.05, "method": "robust"},
    {"name": "CROWN_eps03", "bound_type": "CROWN", "eps": 0.1, "method": "robust"},
    {"name": "CROWN_IBP_eps02", "bound_type": "CROWN-IBP", "eps": 0.05, "method": "robust"},
]

def train_model(config, num_epochs=3, batch_size=256, lr=5e-4, device="cuda"):
    """Train a model with given configuration"""
    set_seed(100)
    
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print(f"Bound type: {config['bound_type']}, Eps: {config['eps']}")
    print(f"{'='*60}\n")
    
    # Create model
    model_ori = models.Models['cnn_4layer'](in_ch=1, in_dim=28)
    dummy_input = torch.randn(2, 1, 28, 28)
    
    # Prepare dataset
    train_data = torchvision.datasets.MNIST(
        "./data", train=True, download=True, 
        transform=transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(
        "./data", train=False, download=True, 
        transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        pin_memory=True, num_workers=min(multiprocessing.cpu_count(), 4))
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, pin_memory=True, 
        num_workers=min(multiprocessing.cpu_count(), 4))
    
    train_loader.mean = test_loader.mean = torch.tensor([0.0])
    train_loader.std = test_loader.std = torch.tensor([1.0])
    
    # Wrap model with auto_LiRPA
    model = BoundedModule(
        model_ori, dummy_input, 
        bound_opts={'conv_mode': 'patches'}, 
        device=device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Epsilon scheduler
    eps_scheduler = FixedScheduler(config['eps'])
    
    norm = float("inf")
    num_class = 10
    
    # Training curves storage
    training_curves = {
        'train_loss': [],
        'train_loss_batch': [],  # Per batch loss for more detailed tracking
        'test_accuracy': [],
        'epoch': [],
        'batch': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(len(train_loader))
        
        epoch_train_losses = []
        
        for i, (data, labels) in enumerate(train_loader):
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
            
            batch_method = config['method']
            if eps < 1e-20:
                batch_method = "natural"
            
            optimizer.zero_grad()
            
            # Generate specifications
            c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - \
                torch.eye(num_class).type_as(data).unsqueeze(0)
            I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
            c = (c[I].view(data.size(0), num_class - 1, num_class))
            
            # Bound input for Linf norm
            data_max = torch.reshape((1. - train_loader.mean) / train_loader.std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - train_loader.mean) / train_loader.std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / train_loader.std).view(1,-1,1,1), data_max)
            data_lb = torch.max(data - (eps / train_loader.std).view(1,-1,1,1), data_min)
            
            if device == "cuda" and torch.cuda.is_available():
                data, labels, c = data.cuda(), labels.cuda(), c.cuda()
                data_lb, data_ub = data_lb.cuda(), data_ub.cuda()
            
            # Create perturbation
            if eps > 0:
                ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
            else:
                # For natural training, still create a perturbation but with eps=0
                ptb = PerturbationLpNorm(norm=norm, eps=0.0)
            x = BoundedTensor(data, ptb)
            
            output = model(x)
            regular_ce = CrossEntropyLoss()(output, labels)
            if batch_method == "robust" and config['bound_type'] is not None:
                if config['bound_type'] == "IBP":
                    lb, ub = model.compute_bounds(IBP=True, C=c, method=None)
                elif config['bound_type'] == "CROWN":
                    lb, ub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                elif config['bound_type'] == "CROWN-IBP":
                    max_eps = eps_scheduler.get_max_eps() if hasattr(eps_scheduler, 'get_max_eps') else eps
                    factor = (max_eps - eps) / max_eps if max_eps > 0 else 0.0
                    ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
                    if factor < 1e-5:
                        lb = ilb
                    else:
                        clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                        lb = clb * factor + ilb * (1 - factor)
                
                lb_padded = torch.cat((
                    torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), 
                    lb), dim=1)
                fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
                robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
                loss = robust_ce
            else:
                loss = regular_ce
            
            loss.backward()
            optimizer.step()
            
            # Record batch loss
            batch_loss = loss.item()
            epoch_train_losses.append(batch_loss)
            training_curves['train_loss_batch'].append(batch_loss)
            training_curves['batch'].append((epoch - 1) * len(train_loader) + i)
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {batch_loss:.4f}")
        
        # Record epoch average loss
        avg_epoch_loss = np.mean(epoch_train_losses)
        training_curves['train_loss'].append(avg_epoch_loss)
        training_curves['epoch'].append(epoch)
        
        if eps_scheduler.reached_max_eps():
            lr_scheduler.step()
        
        # Evaluate on test set
        model.eval()
        eps_scheduler.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in test_loader:
                if device == "cuda" and torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            training_curves['test_accuracy'].append(accuracy)
            print(f"Epoch {epoch} Test Accuracy: {accuracy:.2f}%")
    
    # Save model
    model_path = f"models/{config['name']}.pth"
    os.makedirs("models", exist_ok=True)
    
    # Verify model weights are updated (check first layer weight sum)
    first_layer_key = list(model_ori.state_dict().keys())[0]
    weight_sum = model_ori.state_dict()[first_layer_key].sum().item()
    print(f"Model weight verification - First layer weight sum: {weight_sum:.6f}")
    
    torch.save({'state_dict': model_ori.state_dict()}, model_path)
    print(f"Model saved to {model_path}\n")
    
    return model_path, training_curves

def evaluate_robustness(model_path, config, num_samples=10, eps_test=0.2, device="cuda"):
    """Evaluate robustness of a trained model"""
    print(f"Evaluating robustness: {config['name']}")
    print(f"  Loading model from: {model_path}")
    
    # Load model
    model_ori = models.Models['cnn_4layer'](in_ch=1, in_dim=28)
    checkpoint = torch.load(model_path, map_location='cpu')
    model_ori.load_state_dict(checkpoint['state_dict'])
    
    # Verify model is loaded correctly by checking a weight sum (for debugging)
    first_layer_weight_sum = sum(model_ori.state_dict()[list(model_ori.state_dict().keys())[0]].flatten()).item()
    print(f"  Model first layer weight sum: {first_layer_weight_sum:.6f} (for verification)")
    
    # Prepare test data
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True,
        transform=transforms.ToTensor())
    
    # Select 10 samples
    N = min(num_samples, len(test_data))
    images = test_data.data[:N].view(N, 1, 28, 28)
    true_labels = test_data.targets[:N]
    images = images.to(torch.float32) / 255.0
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        images = images.cuda()
        model_ori = model_ori.cuda()
        device_tensor = torch.device('cuda')
    else:
        device_tensor = torch.device('cpu')
    
    # Wrap with auto_LiRPA
    dummy_input = torch.empty_like(images[:1])
    lirpa_model = BoundedModule(
        model_ori, dummy_input, 
        bound_opts={'conv_mode': 'patches'}, 
        device=device_tensor)
    
    # Get predictions on clean images (without perturbation)
    with torch.no_grad():
        pred_clean = lirpa_model(images)
        pred_labels = torch.argmax(pred_clean, dim=1).cpu().detach().numpy()
    
    # Print predictions for debugging
    print(f"  Predictions on clean images: {pred_labels.tolist()}")
    print(f"  True labels: {true_labels.tolist()}")
    clean_accuracy = (pred_labels == true_labels.numpy()).mean() * 100
    print(f"  Clean accuracy: {clean_accuracy:.2f}%")
    
    # Compute bounds using alpha-CROWN for tightest bounds
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps_test)
    bounded_images = BoundedTensor(images, ptb)
    
    # Set optimization options
    lirpa_model.set_bound_opts({
        'optimize_bound_args': {
            'iteration': 20, 
            'lr_alpha': 0.1
        }
    })
    
    # Compute bounds
    lb, ub = lirpa_model.compute_bounds(
        x=(bounded_images,), 
        method='CROWN-Optimized')
    
    # Compute margin bounds for each sample
    results = {
        'model_name': config['name'],
        'training_config': config,
        'test_eps': eps_test,
        'num_samples': N,
        'samples': []
    }
    
    robust_count = 0
    margin_lbs = []
    
    for i in range(N):
        true_label = true_labels[i].item()
        pred_label = pred_labels[i]
        
        # Compute margin bounds: compare true class with all other classes
        # We'll compute margin with the most dangerous class (highest upper bound)
        true_class_lb = lb[i][true_label].item()
        true_class_ub = ub[i][true_label].item()
        
        # Find the most dangerous class (highest upper bound among other classes)
        other_classes_ub = []
        for j in range(10):
            if j != true_label:
                other_classes_ub.append({
                    'class': j,
                    'ub': ub[i][j].item(),
                    'lb': lb[i][j].item()
                })
        
        max_other_ub = max([x['ub'] for x in other_classes_ub])
        most_dangerous_class = max(other_classes_ub, key=lambda x: x['ub'])['class']
        
        # Compute margin bound with most dangerous class
        C = torch.zeros(size=(1, 1, 10), device=device_tensor)
        C[0, 0, true_label] = 1.0
        C[0, 0, most_dangerous_class] = -1.0
        
        margin_lb, margin_ub = lirpa_model.compute_bounds(
            x=(bounded_images[i:i+1],), 
            method='CROWN-Optimized',
            C=C)
        
        margin_lb_val = margin_lb[0, 0].item()
        margin_ub_val = margin_ub[0, 0].item()
        
        # Check robustness: 
        # 1. Prediction must be correct on clean image
        # 2. Margin lower bound > 0 means robust (guaranteed correct prediction under perturbation)
        is_correct = (pred_label == true_label)
        is_robust = is_correct and (margin_lb_val > 0)
        
        if is_robust:
            robust_count += 1
        
        margin_lbs.append(margin_lb_val)
        
        sample_result = {
            'sample_id': i,
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'is_correct': bool(is_correct),
            'is_robust': bool(is_robust),
            'margin_lb': float(margin_lb_val),
            'margin_ub': float(margin_ub_val),
            'true_class_lb': float(true_class_lb),
            'true_class_ub': float(true_class_ub),
            'most_dangerous_class': int(most_dangerous_class),
            'max_other_class_ub': float(max_other_ub)
        }
        results['samples'].append(sample_result)
    
    # Overall statistics
    correct_count = sum(1 for s in results['samples'] if s['is_correct'])
    results['clean_accuracy'] = correct_count / N
    results['certified_robust_accuracy'] = robust_count / N
    results['avg_margin_lb'] = float(np.mean(margin_lbs))
    results['min_margin_lb'] = float(np.min(margin_lbs))
    results['max_margin_lb'] = float(np.max(margin_lbs))
    
    print(f"  Clean Accuracy: {results['clean_accuracy']:.2%}")
    print(f"  Certified Robust Accuracy: {results['certified_robust_accuracy']:.2%}")
    print(f"  Average Margin Lower Bound: {results['avg_margin_lb']:.4f}")
    print(f"  Min Margin Lower Bound: {results['min_margin_lb']:.4f}")
    print(f"  Robust samples: {robust_count}/{N}\n")
    
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    all_results = []
    all_training_curves = {}
    
    # Train and evaluate each model
    for config in EXPERIMENTS:
        try:
            # Train model
            model_path, training_curves = train_model(config, num_epochs=4, device=device)
            
            # Save training curves
            all_training_curves[config['name']] = training_curves
            
            # Evaluate robustness (using eps_test=0.2 for fair comparison)
            # Increase num_samples for more reliable statistics
            result = evaluate_robustness(
                model_path, config, 
                num_samples=100,  # Increased from 10 to 100 for better statistics
                eps_test=0.2, 
                device=device)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error in experiment {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to JSON
    output_file = "robustness_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save training curves to JSON
    curves_file = "training_curves.json"
    with open(curves_file, 'w') as f:
        json.dump(all_training_curves, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results Summary (evaluated at eps=0.2)")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Clean Acc':<12} {'Robust Acc':<12} {'Avg Margin LB':<15} {'Min Margin LB':<15}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model_name']:<25} "
              f"{result.get('clean_accuracy', 0):>6.2%}      "
              f"{result['certified_robust_accuracy']:>6.2%}      "
              f"{result['avg_margin_lb']:>10.4f}      "
              f"{result['min_margin_lb']:>10.4f}")
    
    print(f"\nDetailed results saved to {output_file}")
    print(f"Training curves saved to {curves_file}")

if __name__ == "__main__":
    main()

