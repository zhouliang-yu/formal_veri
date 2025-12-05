# Formal Verification for Deep Neural Network

## Installation

please refer to the installation instructions in the auto_LiRPA repository. You need to install the dependencies in the auto_LiRPA repository first.

https://auto-lirpa.readthedocs.io/en/latest/installation.html


## Depth / Dropout / Width vs. Certified Robustness

### Goal

Train MNIST multi-layer perceptrons with varying hidden-layer counts, dropout rates, and widths, then compare certified robustness under fixed $\ell_\infty$ perturbations using Auto-LiRPA. The script writes checkpoints and `bounds_depth*_drop*_width*.json` logs, which feed into the heatmap and curve visualizations.

### Training Command

```
python examples/vision/mnist_mlp_depth_verification.py \
  --depths 2 3 4 5 \
  --dropout-probs 0.0 0.1 0.2 0.3 \
  --hidden-dims 256 \
  --epochs 20 \
  --patience 3 \
  --min-delta 1e-3 \
  --train-samples 6000 \
  --eval-samples 64 \
  --eps-grid 0.04 0.02 0.01 0.005 \
  --cert-threshold 0.02 \
  --data-dir ./data \
  --output-dir ./results/mnist_mlp_dropout_sweep
```

### Plotting Command (heatmaps / curves)

```
python examples/vision/plot_mlp_depth_heatmap.py \
  --bounds-dir ./results/mnist_mlp_dropout_sweep \
  --y-axis dropout \
  --depth-filter 2 \
  --save-path ./results/mnist_mlp_dropout_sweep/dropout_heatmaps.pdf \
  --format pdf
```

results can be found in 
```
./results/mnist_mlp_depth
./results/mnist_mlp_dropout_sweep
./results/mnist_mlp_width_sweep


## Robust MNIST MLP Training (Pretrain vs. Direct)

### Goal

Compare post-training fine-tuning versus direct robust optimization on MNIST MLPs under multiple target radii. Outputs include checkpoints, `summary.json`, and visualizations (accuracy bars, certified curves).

### Training Command (with pretrain + direct comparison)

```
python examples/vision/mnist_mlp_robust_training.py \
  --epsilons 0.05 0.10 0.20 \
  --hidden-sizes 256 256 \
  --train-epochs 15 \
  --warmup-epochs 10 \
  --natural-weight 0.9 \
  --pretrain-epochs 5 \
  --pretrain-lr 2e-3 \
  --compare-direct \
  --match-direct-steps \
  --results-dir ./results/mnist_mlp_robust_mlp \
  --data-dir ./data
```

Use `--only-epsilon <value>` to retrain a single radius; use `--overwrite-pretrain` if the cached baseline should be regenerated.

results can be found in 
```
./results/mnist_mlp_robust_mlp
```

### Plotting Command

```
python examples/vision/plot_mnist_mlp_robust_summary.py \
  --summary ./results/mnist_mlp_robust_mlp/summary.json \
  --out-dir ./results/mnist_mlp_robust_mlp/viz \
  --stages finetuned direct \
  --format pdf
```

Generated files include:

- `test_acc_compare.pdf` (natural accuracy comparison)
- `curves_finetuned_vs_direct_eps*.pdf` (per-$\epsilon$ certified curves)
- `heatmap_finetuned.pdf`, `heatmap_direct.pdf` (optional heatmaps)


## The vision experiments (ResNet with MNIST) 
```
cd formal_veri/auto_LiRPA/auto_LiRPA/examples/vision
# pretrain models in /models dir
# run training
python simple_training.py
# run evaluation
python simple_verification.py
```



## The NLP experiments (transformer with sst) 
```
cd formal_veri/auto_LiRPA/auto_LiRPA/examples/language/
# download sst dataset via sst.py
# pretrain models in /models dir
# run training
python train.py --train --device cpu \
    --model transformer --num_epochs 10 \
    --dir trained_model
```
