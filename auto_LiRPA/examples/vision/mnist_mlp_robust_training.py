"""
Robust MNIST training with an MLP backbone and Auto-LiRPA-based certified loss.

This script is intentionally lightweight and focuses on:
* training identical MLPs under different L_inf epsilons;
* adding a robust loss term computed from LiRPA bounds (IBP by default);
* reporting both natural accuracy and certified ratios for multiple epsilons.

Example usage (train four models with eps in {0.05, 0.1, 0.2, 0.3}):

    python mnist_mlp_robust_training.py \
        --epsilons 0.05 0.1 0.2 0.3 \
        --hidden-sizes 256 256 \
        --train-epochs 15 \
        --bound-type IBP \
        --results-dir ./results/mnist_mlp_robust

To focus only on one (fast) bound method, keep --bound-type=IBP (default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


def build_mlp(hidden_sizes: List[int], num_classes: int = 10) -> nn.Sequential:
    layers: List[nn.Module] = [Flatten()]
    in_dim = 28 * 28
    for size in hidden_sizes:
        layers.append(nn.Linear(in_dim, size))
        layers.append(nn.ReLU())
        in_dim = size
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def get_loaders(batch_size: int, eval_samples: int | None, num_workers: int = 2) -> Dict[str, DataLoader]:
    transform = torchvision.transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    eval_subset = test_ds
    if eval_samples is not None and eval_samples < len(test_ds):
        eval_subset = Subset(test_ds, list(range(eval_samples)))
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "eval": DataLoader(eval_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders


def specification_matrix(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    device = labels.device
    eye = torch.eye(num_classes, device=device)
    c = eye[labels].unsqueeze(1) - eye.unsqueeze(0)
    mask = labels.unsqueeze(1) != torch.arange(num_classes, device=device).unsqueeze(0)
    return c[mask].view(labels.size(0), num_classes - 1, num_classes)


def compute_margin_lower_bounds(
    lirpa_model: BoundedModule,
    bounded_x: BoundedTensor,
    spec: torch.Tensor,
    bound_type: str,
    crown_ibp_mix: float,
) -> torch.Tensor:
    if bound_type == "IBP":
        lb, _ = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP", C=spec)
        return lb
    if bound_type == "CROWN":
        lb, _ = lirpa_model.compute_bounds(x=(bounded_x,), method="backward", C=spec, bound_upper=False)
        return lb
    if bound_type == "CROWN-IBP":
        ilb, _ = lirpa_model.compute_bounds(x=(bounded_x,), method="IBP", C=spec)
        clb, _ = lirpa_model.compute_bounds(x=(bounded_x,), method="backward", C=spec, bound_upper=False)
        alpha = float(torch.clamp(torch.tensor(crown_ibp_mix), 0.0, 1.0))
        return alpha * clb + (1.0 - alpha) * ilb
    raise ValueError(f"Unsupported bound type: {bound_type}")


@torch.no_grad()
def evaluate(
    lirpa_model: BoundedModule,
    loader: DataLoader,
    device: torch.device,
    eps_list: List[float],
    bound_type: str,
    crown_ibp_mix: float,
) -> Dict[str, float]:
    lirpa_model.eval()
    total = 0
    natural_correct = 0
    certified_counts = {eps: 0 for eps in eps_list}
    criterion = nn.CrossEntropyLoss(reduction="sum")
    natural_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        logits = lirpa_model(images)
        natural_loss += criterion(logits, labels).item()
        natural_correct += (logits.argmax(dim=1) == labels).sum().item()
        specs = specification_matrix(labels)
        for eps in eps_list:
            if eps <= 0:
                certified_counts[eps] += (logits.argmax(dim=1) == labels).sum().item()
                continue
            bounded = create_bounded_tensor(images, eps)
            lb = compute_margin_lower_bounds(lirpa_model, bounded, specs, bound_type, crown_ibp_mix)
            certified_counts[eps] += (lb > 0).all(dim=1).sum().item()
    metrics = {
        "natural_acc": natural_correct / max(1, total),
        "natural_loss": natural_loss / max(1, total),
    }
    for eps in eps_list:
        metrics[f"cert_acc_eps_{eps}"] = certified_counts[eps] / max(1, total)
    return metrics


def create_bounded_tensor(images: torch.Tensor, eps: float) -> BoundedTensor:
    if eps <= 0:
        return BoundedTensor(images, PerturbationLpNorm(norm=float("inf"), eps=0.0))
    x_l = torch.clamp(images - eps, 0.0, 1.0)
    x_u = torch.clamp(images + eps, 0.0, 1.0)
    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps, x_L=x_l, x_U=x_u)
    return BoundedTensor(images, ptb)


def train_epoch(
    lirpa_model: BoundedModule,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    eps: float,
    bound_type: str,
    crown_ibp_mix: float,
    natural_weight: float,
    enable_robust: bool = True,
) -> Dict[str, float]:
    lirpa_model.train()
    criterion = nn.CrossEntropyLoss()
    stats = {"loss": 0.0, "natural_loss": 0.0, "robust_loss": 0.0, "correct": 0, "total": 0}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        bounded = create_bounded_tensor(images, eps)
        logits = lirpa_model(bounded)
        ce_loss = criterion(logits, labels)
        use_robust = enable_robust and (1.0 - natural_weight) > 0 and eps > 0
        if use_robust:
            specs = specification_matrix(labels)
            lb = compute_margin_lower_bounds(lirpa_model, bounded, specs, bound_type, crown_ibp_mix)
            lb_padded = torch.cat([torch.zeros(lb.size(0), 1, device=lb.device), lb], dim=1)
            fake_labels = torch.zeros(lb.size(0), dtype=torch.long, device=lb.device)
            robust_loss = criterion(-lb_padded, fake_labels)
            loss = natural_weight * ce_loss + (1.0 - natural_weight) * robust_loss
        else:
            robust_loss = torch.zeros((), device=images.device)
            loss = ce_loss
        loss.backward()
        optimizer.step()
        stats["loss"] += loss.item() * labels.size(0)
        stats["natural_loss"] += ce_loss.item() * labels.size(0)
        stats["robust_loss"] += robust_loss.item() * labels.size(0)
        stats["correct"] += (logits.argmax(dim=1) == labels).sum().item()
        stats["total"] += labels.size(0)
    for key in ["loss", "natural_loss", "robust_loss"]:
        stats[key] /= max(1, stats["total"])
    stats["train_acc"] = stats["correct"] / max(1, stats["total"])
    return stats


def run_experiment(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    loaders = get_loaders(batch_size=args.batch_size, eval_samples=args.eval_samples)
    eval_eps = args.eval_epsilons or []
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, object]] = []
    dummy = torch.zeros(1, 1, 28, 28, device=device)

    def init_lirpa(state_dict: Dict[str, torch.Tensor] | None = None) -> BoundedModule:
        base_model = build_mlp(args.hidden_sizes).to(device)
        lirpa = BoundedModule(base_model, dummy, device=device)
        if state_dict is not None:
            lirpa.load_state_dict(state_dict)
        return lirpa

    pretrain_state: Dict[str, torch.Tensor] | None = None
    baseline_eval: Dict[str, float] | None = None
    baseline_test: Dict[str, float] | None = None
    baseline_path = output_dir / "mlp_pretrain.pt"
    if baseline_path.exists() and not args.overwrite_pretrain:
        print(f"Found existing pretrain checkpoint at {baseline_path}; loading.")
        checkpoint = torch.load(baseline_path, map_location=device)
        pretrain_state = checkpoint["state_dict"]
    elif args.pretrain_epochs > 0:
        print(f"No cached pretrain model found (or overwrite requested); training baseline for {args.pretrain_epochs} epochs.")
        lirpa_model = init_lirpa()
        pre_lr = args.pretrain_lr if args.pretrain_lr is not None else args.lr
        optimizer = optim.Adam(lirpa_model.parameters(), lr=pre_lr)
        for epoch in range(1, args.pretrain_epochs + 1):
            stats = train_epoch(
                lirpa_model,
                loaders["train"],
                optimizer,
                device,
                args.pretrain_eps,
                args.bound_type,
                args.crown_ibp_mix,
                natural_weight=1.0,
                enable_robust=False,
            )
            print(
                f"[baseline eps={args.pretrain_eps:.3f}] Epoch {epoch}/{args.pretrain_epochs} "
                f"| loss={stats['loss']:.4f} acc={stats['train_acc']*100:.2f}%"
            )
        pretrain_state = copy.deepcopy(lirpa_model.state_dict())
        torch.save({"state_dict": pretrain_state, "stage": "baseline", "args": vars(args)}, baseline_path)
        print(f"Saved baseline checkpoint to {baseline_path}")

    if pretrain_state is not None:
        baseline_model = init_lirpa(pretrain_state)
        scaled = [round(args.pretrain_eps * r, 6) for r in args.eval_ratio_eps]
        combined_eps = sorted(set((eval_eps or []) + scaled))
        baseline_eval = evaluate(
            baseline_model,
            loaders["eval"],
            device,
            combined_eps,
            args.bound_type,
            args.crown_ibp_mix,
        )
        baseline_test = evaluate(
            baseline_model,
            loaders["test"],
            device,
            [0.0],
            args.bound_type,
            args.crown_ibp_mix,
        )

    def build_eval_set(target_eps: float) -> List[float]:
        scaled = [round(target_eps * r, 6) for r in args.eval_ratio_eps]
        extra = eval_eps if eval_eps else []
        return sorted(set(scaled + extra))

    eps_schedule = args.epsilons
    if args.only_epsilon is not None:
        eps_schedule = [args.only_epsilon]

    existing_summary = []
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite_summary:
        with summary_path.open() as f:
            existing_summary = json.load(f)

    def update_summary(record: Dict) -> None:
        nonlocal existing_summary
        existing_summary = [rec for rec in existing_summary if abs(rec.get("epsilon", -1) - record["epsilon"]) > 1e-9]
        existing_summary.append(record)

    for eps in eps_schedule:
        print("=" * 80)
        print(f"Training robust MLP with epsilon={eps:.3f}")
        record: Dict[str, object] = {"epsilon": eps}
        if baseline_eval is not None and baseline_test is not None:
            record["baseline_eval"] = baseline_eval
            record["baseline_test"] = baseline_test

        def run_robust_stage(stage_name: str, init_state: Dict[str, torch.Tensor] | None, epochs: int) -> Tuple[Dict[str, float], BoundedModule]:
            lirpa = init_lirpa(init_state)
            optimizer = optim.Adam(lirpa.parameters(), lr=args.lr)
            warmup_epochs = max(0, min(args.warmup_epochs, epochs))

            def current_eps(epoch_idx: int) -> float:
                if warmup_epochs == 0:
                    return eps
                progress = min(1.0, epoch_idx / warmup_epochs)
                return args.warmup_eps_start + (eps - args.warmup_eps_start) * progress

            last_stats = None
            for epoch in range(1, epochs + 1):
                epoch_eps = current_eps(epoch)
                last_stats = train_epoch(
                    lirpa,
                    loaders["train"],
                    optimizer,
                    device,
                    epoch_eps,
                    args.bound_type,
                    args.crown_ibp_mix,
                    args.natural_weight,
                )
                print(
                    f"[{stage_name}] eps={eps:.3f} used={epoch_eps:.3f} "
                    f"Epoch {epoch}/{epochs} | loss={last_stats['loss']:.4f} "
                    f"natural={last_stats['natural_loss']:.4f} robust={last_stats['robust_loss']:.4f} "
                    f"acc={last_stats['train_acc']*100:.2f}%"
                )
            eval_set = build_eval_set(eps)
            eval_metrics = evaluate(
                lirpa,
                loaders["eval"],
                device,
                eval_set,
                args.bound_type,
                args.crown_ibp_mix,
            )
            test_metrics = evaluate(
                lirpa,
                loaders["test"],
                device,
                [0.0],
                args.bound_type,
                args.crown_ibp_mix,
            )
            metrics = {
                "train_loss": last_stats["loss"] if last_stats else 0.0,
                "train_acc": last_stats["train_acc"] if last_stats else 0.0,
                "eval": eval_metrics,
                "test_acc": test_metrics["natural_acc"],
            }
            return metrics, lirpa

        if pretrain_state is not None:
            print("Stage 2: robust fine-tuning from pretrained weights.")
            finetuned_metrics, ft_model = run_robust_stage("finetune", pretrain_state, args.train_epochs)
            record["finetuned"] = finetuned_metrics
            ckpt_path = output_dir / f"mlp_eps{eps:.3f}_finetuned.pt"
            torch.save(
                {"state_dict": ft_model.state_dict(), "stage": "finetuned", "epsilon": eps, "args": vars(args)},
                ckpt_path,
            )
            print(f"Saved fine-tuned checkpoint to {ckpt_path}")
        else:
            metrics, robust_model = run_robust_stage("robust", None, args.train_epochs)
            record["robust"] = metrics
            ckpt_path = output_dir / f"mlp_eps{eps:.3f}_robust.pt"
            torch.save(
                {"state_dict": robust_model.state_dict(), "stage": "robust", "epsilon": eps, "args": vars(args)},
                ckpt_path,
            )
            print(f"Saved robust checkpoint to {ckpt_path}")

        if args.compare_direct:
            print("Stage 3: direct robust training from scratch for comparison.")
            extra_epochs = args.pretrain_epochs if (args.match_direct_steps and args.pretrain_epochs > 0) else 0
            direct_epochs = max(1, args.train_epochs + extra_epochs)
            direct_metrics, direct_model = run_robust_stage("direct", None, direct_epochs)
            record["direct"] = direct_metrics
            direct_ckpt = output_dir / f"mlp_eps{eps:.3f}_direct.pt"
            torch.save(
                {"state_dict": direct_model.state_dict(), "stage": "direct", "epsilon": eps, "args": vars(args)},
                direct_ckpt,
            )
            print(f"Saved direct-training checkpoint to {direct_ckpt}")

        update_summary(record)

    with summary_path.open("w") as f:
        json.dump(sorted(existing_summary, key=lambda r: r["epsilon"]), f, indent=2)
    print(f"Wrote summary metrics to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust MNIST MLP training with Auto-LiRPA robust loss.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256], help="MLP hidden layer sizes.")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.2, 0.3], help="Epsilons to train separate models on.")
    parser.add_argument("--only-epsilon", type=float, default=None, help="Restrict training/evaluation to a single epsilon.")
    parser.add_argument("--eval-epsilons", type=float, nargs="+", default=None, help="Additional epsilons used for certification.")
    parser.add_argument("--eval-ratio-eps", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5], help="Ratios (times training epsilon) used for evaluation.")
    parser.add_argument("--train-epochs", type=int, default=10, help="Training epochs per epsilon.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--bound-type", type=str, choices=["IBP", "CROWN", "CROWN-IBP"], default="IBP", help="Bound method used for robust loss.")
    parser.add_argument("--crown-ibp-mix", type=float, default=0.5, help="Mixing factor (0..1) when using CROWN-IBP.")
    parser.add_argument("--natural-weight", type=float, default=0.2, help="Weight on natural CE (1-weight on robust CE).")
    parser.add_argument("--pretrain-epochs", type=int, default=0, help="Natural pretraining epochs before robust finetune.")
    parser.add_argument("--pretrain-eps", type=float, default=0.0, help="Epsilon used during pretraining (typically 0).")
    parser.add_argument("--pretrain-lr", type=float, default=2e-3, help="Learning rate for pretraining stage (defaults higher than robust lr).")
    parser.add_argument("--overwrite-pretrain", action="store_true", help="Force retraining of baseline even if checkpoint exists.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of epochs to linearly ramp epsilon.")
    parser.add_argument("--warmup-eps-start", type=float, default=0.0, help="Starting epsilon for warmup schedule.")
    parser.add_argument("--compare-direct", action="store_true", help="Also train a robust model from scratch for comparison.")
    parser.add_argument("--match-direct-steps", dest="match_direct_steps", action="store_true", help="Match total epochs of direct training with pretrain+finetune.")
    parser.add_argument("--no-match-direct-steps", dest="match_direct_steps", action="store_false", help="Do not extend direct training epochs.")
    parser.set_defaults(match_direct_steps=True)
    parser.add_argument("--overwrite-summary", action="store_true", help="Ignore existing summary records and rewrite (default merges).")
    parser.add_argument("--eval-samples", type=int, default=512, help="Number of eval samples for certification metrics.")
    parser.add_argument("--results-dir", type=str, default="./results/mnist_mlp_robust", help="Where to save checkpoints and metrics.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())

