"""
Train shallow MNIST MLPs with different depths and compare their Auto-LiRPA
certified robustness under a fixed input perturbation.

Usage (runs on CPU by default):
    python mnist_mlp_depth_verification.py --epochs 3 --train-samples 8000

Notes:
* Depth refers to the number of hidden layers (2~5 by default).
* We keep the script lightweight so it can be used as a tutorial-style reference
  for experimenting with simple fully connected networks.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


def build_mlp(
    hidden_layers: int,
    hidden_dim: int,
    num_classes: int = 10,
    dropout_prob: float = 0.0,
) -> nn.Sequential:
    """Construct an MLP with `hidden_layers` hidden layers of size `hidden_dim`."""
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1.")
    if not 0.0 <= dropout_prob < 1.0:
        raise ValueError("dropout_prob must be in [0, 1).")
    layers: List[nn.Module] = [Flatten()]
    in_dim = 28 * 28
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def get_loaders(
    data_dir: str,
    batch_size: int,
    train_samples: int,
    test_batch: int,
    val_split: float,
    split_seed: int,
) -> Dict[str, DataLoader]:
    """Return train/val/test DataLoaders (training subset for quicker experiments)."""
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    if train_samples and train_samples < len(train_set):
        train_set = Subset(train_set, list(range(train_samples)))
    if not 0 < val_split < 1:
        raise ValueError("val_split must be in (0, 1).")
    val_size = max(1, int(len(train_set) * val_split))
    train_size = len(train_set) - val_size
    if train_size <= 0:
        raise ValueError("val_split too large for the selected training subset.")
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )
    test_set = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False, num_workers=2)
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
) -> Dict[str, float]:
    """Train until validation accuracy plateaus; returns stats such as epochs used."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate_accuracy(model, val_loader, device)
        epochs_run = epoch + 1
        print(f"Epoch {epochs_run}/{max_epochs} - loss: {avg_loss:.4f} - val acc: {val_acc * 100:.2f}%")
        if val_acc - best_acc > min_delta:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Validation accuracy not improved for {patience} epoch(s); early stopping.")
                break
    model.load_state_dict(best_state)
    return {"epochs": epochs_run, "best_val_acc": best_acc}


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy on the provided loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def build_all_margin_spec(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Build a specification matrix that compares the ground-truth logit against every
    other logit. Positive lower bounds indicate the sample is certified robust.
    """
    batch = labels.shape[0]
    device = labels.device
    all_classes = torch.arange(num_classes, device=device).repeat(batch, 1)
    others = all_classes[all_classes != labels.unsqueeze(1)].view(batch, num_classes - 1)
    spec = torch.zeros(batch, num_classes - 1, num_classes, device=device)
    gt_idx = labels.view(batch, 1, 1).repeat(1, num_classes - 1, 1)
    spec.scatter_(dim=2, index=gt_idx, value=1.0)
    spec.scatter_(dim=2, index=others.unsqueeze(-1), value=-1.0)
    return spec


def certify_with_lirpa(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    eps: float,
    methods: List[str],
    train_mode: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute Auto-LiRPA bounds under L_inf epsilon-ball perturbations."""
    images = images.to(device)
    labels = labels.to(device)
    if train_mode:
        model.train()
        with torch.no_grad():
            _ = model(images)
    else:
        model.eval()
    lirpa_model = BoundedModule(model, torch.empty_like(images), device=device)
    if train_mode:
        with torch.no_grad():
            lirpa_model.train()
            _ = lirpa_model(images)
            lirpa_model.eval()
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_images = BoundedTensor(images, ptb)
    C = build_all_margin_spec(labels, num_classes=10)

    results = {}
    for method in methods:
        if "Optimized" in method:
            lirpa_model.set_bound_opts({"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(bounded_images,), method=method.split()[0], C=C)
        margin_lb = lb.view(lb.size(0), -1)
        certified = (margin_lb > 0).all(dim=1).float().mean().item()
        results[method] = {
            "min_margin_lb": margin_lb.min().item(),
            "mean_margin_lb": margin_lb.mean().item(),
            "certified_ratio": certified,
        }
    return results


def sweep_epsilons(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    eps_values: List[float],
    methods: List[str],
    target_ratio: float,
    train_mode: bool = False,
) -> List[Dict[str, object]]:
    """Evaluate robustness over all epsilons; note when the target ratio is first reached."""
    logs: List[Dict[str, object]] = []
    reached_eps = None
    for eps in eps_values:
        print(f"--- Evaluating robustness at eps = {eps:.4f} ---")
        verification = certify_with_lirpa(
            model,
            images.clone(),
            labels.clone(),
            device,
            eps=eps,
            methods=methods,
            train_mode=train_mode,
        )
        for method, stats in verification.items():
            print(
                f"[eps={eps:.3f}] {method} "
                f"| min lb: {stats['min_margin_lb']:.4f} "
                f"| mean lb: {stats['mean_margin_lb']:.4f} "
                f"| certified: {stats['certified_ratio'] * 100:.1f}%"
            )
            logs.append(
                {
                    "eps": eps,
                    "method": method,
                    "min_margin_lb": stats["min_margin_lb"],
                    "mean_margin_lb": stats["mean_margin_lb"],
                    "certified_ratio": stats["certified_ratio"],
                }
            )
        if reached_eps is None and any(
            stats["certified_ratio"] >= target_ratio for stats in verification.values()
        ):
            reached_eps = eps
            print(
                f"Reached target certified ratio ({target_ratio * 100:.1f}%) "
                f"at eps = {eps:.4f}; continuing sweep."
            )
    if reached_eps is None:
        print("Target certified ratio not achieved for any epsilon in the grid.")
    return logs


def select_device(force_cpu: bool) -> torch.device:
    """Prefer Apple MPS, then CUDA, otherwise CPU."""
    if not force_cpu:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def select_lirpa_device(train_device: torch.device, choice: str) -> torch.device:
    """Resolve the device used for Auto-LiRPA verification."""
    if choice == "auto":
        if train_device.type == "mps":
            print("Auto-LiRPA does not fully support MPS; falling back to CPU for bounds.")
            return torch.device("cpu")
        return train_device
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("LiRPA device 'cuda' requested but CUDA is unavailable.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("LiRPA device 'mps' requested but MPS is unavailable.")
        return torch.device("mps")
    return torch.device("cpu")


def run_experiment(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = select_device(args.cpu)
    print("Using device:", device)
    lirpa_device = select_lirpa_device(device, args.lirpa_device)
    print("LiRPA verification device:", lirpa_device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        test_batch=args.eval_samples,
        val_split=args.val_split,
        split_seed=args.seed,
    )
    test_iter = iter(loaders["test"])
    eval_images, eval_labels = next(test_iter)
    eval_images = eval_images[: args.eval_samples]
    eval_labels = eval_labels[: args.eval_samples]

    methods = args.methods
    hidden_dims = args.hidden_dims if args.hidden_dims else [args.hidden_dim]

    for depth in args.depths:
        for hidden_dim in hidden_dims:
            for dropout in args.dropout_probs:
                print("=" * 80)
                print(
                    f"Training MLP with {depth} hidden layers, hidden dim {hidden_dim}, "
                    f"dropout {dropout:.2f}."
                )
                model = build_mlp(
                    hidden_layers=depth,
                    hidden_dim=hidden_dim,
                    dropout_prob=dropout,
                ).to(device)
                drop_label = str(dropout).replace(".", "p")
                width_label = f"wd{hidden_dim}"
                model_save_path = output_dir / f"mlp_depth{depth}_{width_label}_drop{drop_label}.pt"

                if model_save_path.exists() and not args.overwrite:
                    print(f"Model checkpoint {model_save_path} exists; loading instead of retraining.")
                    checkpoint = torch.load(model_save_path, map_location=device)
                    model.load_state_dict(checkpoint["state_dict"])
                    train_stats = {
                        "epochs": checkpoint.get("epochs_trained", 0),
                        "best_val_acc": checkpoint.get("best_val_acc", 0.0),
                    }
                    test_acc = checkpoint.get("test_acc")
                    if test_acc is None:
                        test_acc = evaluate_accuracy(model, loaders["test"], device)
                else:
                    train_stats = train_model(
                        model,
                        loaders["train"],
                        loaders["val"],
                        device,
                        max_epochs=args.epochs,
                        lr=args.lr,
                        patience=args.patience,
                        min_delta=args.min_delta,
                    )
                    print(
                        f"Depth {depth} / width {hidden_dim} / dropout {dropout:.2f}: "
                        f"trained for {train_stats['epochs']} epoch(s)."
                    )
                    print(
                        f"Depth {depth} / width {hidden_dim} / dropout {dropout:.2f}: "
                        f"best val accuracy {train_stats['best_val_acc'] * 100:.2f}%"
                    )
                    test_acc = evaluate_accuracy(model, loaders["test"], device)
                    print(
                        f"Depth {depth} / width {hidden_dim} / dropout {dropout:.2f}: "
                        f"test accuracy {test_acc * 100:.2f}%"
                    )
                    torch.save(
                        {
                            "depth": depth,
                            "hidden_dim": hidden_dim,
                            "dropout": dropout,
                            "state_dict": model.state_dict(),
                            "best_val_acc": train_stats["best_val_acc"],
                            "epochs_trained": train_stats["epochs"],
                            "test_acc": test_acc,
                            "args": vars(args),
                        },
                        model_save_path,
                    )
                    print(f"Saved model to {model_save_path}")
                print(
                    f"Depth {depth} / width {hidden_dim} / dropout {dropout:.2f}: "
                    f"best val accuracy {train_stats['best_val_acc'] * 100:.2f}% "
                    f"| test acc {test_acc * 100:.2f}%"
                )

                model_for_lirpa = model.to(lirpa_device)
                requires_train_mode = dropout > 0
                bounds_path = output_dir / f"bounds_depth{depth}_{width_label}_drop{drop_label}.json"
                if bounds_path.exists() and not args.overwrite:
                    print(f"Bounds file {bounds_path} exists; skipping verification.")
                else:
                    bound_logs = sweep_epsilons(
                        model_for_lirpa,
                        eval_images,
                        eval_labels,
                        lirpa_device,
                        eps_values=args.eps_grid,
                        methods=methods,
                        target_ratio=args.cert_threshold,
                        train_mode=requires_train_mode,
                    )
                    for entry in bound_logs:
                        entry["depth"] = depth
                        entry["dropout"] = dropout
                        entry["hidden_dim"] = hidden_dim
                    with bounds_path.open("w") as f:
                        json.dump(bound_logs, f, indent=2)
                    print(f"Saved bound logs to {bounds_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare robustness of MNIST MLPs with Auto-LiRPA.")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5], help="Hidden layer counts.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Default width of each hidden layer.")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of widths to sweep; falls back to --hidden-dim if omitted.",
    )
    parser.add_argument(
        "--dropout-probs",
        type=float,
        nargs="+",
        default=[0.0],
        help="Dropout probabilities applied after each hidden ReLU.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop patience (epochs without loss improvement).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum loss decrease to reset patience.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train-samples", type=int, default=6000, help="Subset of MNIST train set.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training subset reserved for validation.")
    parser.add_argument("--eval-samples", type=int, default=32, help="Number of test samples for verification.")
    parser.add_argument(
        "--eps-grid",
        type=float,
        nargs="+",
        default=[0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02],
        help="List of epsilon values (provide from large to small) for robustness sweep.",
    )
    parser.add_argument(
        "--cert-threshold",
        type=float,
        default=0.01,
        help="Target certified ratio; sweep stops once any method reaches this level.",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Where MNIST is downloaded.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--lirpa-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for Auto-LiRPA bounds (auto uses training device unless it is MPS).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/mnist_mlp_depth",
        help="Directory to store trained models and bound summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain and recompute bounds even if checkpoints already exist.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "IBP",
            "IBP+backward (CROWN-IBP)",
            "backward (CROWN)",
            "CROWN-Optimized (alpha-CROWN)",
        ],
        help="Verification methods to run (names must match Auto-LiRPA options).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())

