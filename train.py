"""
Reproducible Training Pipeline for Semantic Segmentation
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 2 & 3:
Trains either the Baseline model or the Controlled Change variant (Dilation)
under strictly recorded, deterministic hyperparameters and outputs detailed training logs.
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.data.contract import (
    DATA_CONTRACT,
    SegmentationTransform,
    ReproducibleSegmentationDataset
)
from src.models.unet import get_model, count_parameters
from src.utils.metrics import SegmentationMetrics


class CombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Soft Dice Loss respecting ignored boundary labels.
    """
    def __init__(self, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # Soft Dice Loss on valid pixels only
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return ce_loss

        probs = F.softmax(logits, dim=1)
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0
        one_hot = F.one_hot(targets_clamped, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        valid_expanded = valid.unsqueeze(1).expand_as(probs)
        p = probs * valid_expanded
        g = one_hot * valid_expanded

        intersection = (p * g).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + g.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)

    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)

    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)

    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics


def main():
    parser = argparse.ArgumentParser(description="Reproducible Segmentation Experiment Trainer")
    parser.add_argument("--mechanism", type=str, choices=["baseline", "dilation"], default="baseline",
                        help="Mechanism: baseline (standard) or dilation (controlled atrous change)")
    parser.add_argument("--epochs", type=int, default=15, help="Fixed epoch budget")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save checkpoints & logs")
    args = parser.parse_args()

    # 1. Deterministic Seeding (Task 2 Requirement)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    # Output directory
    run_dir = os.path.join(args.output_dir, args.mechanism)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, f"train_{args.mechanism}.log")

    # 2. Data Contract & Loaders
    project_root = os.path.dirname(os.path.abspath(__file__))
    res = tuple(DATA_CONTRACT["resolution"])
    train_transform = SegmentationTransform(is_train=True, size=res)
    val_transform = SegmentationTransform(is_train=False, size=res)

    train_ds = ReproducibleSegmentationDataset(project_root, split="train", transform=train_transform)
    val_ds = ReproducibleSegmentationDataset(project_root, split="val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[Dataset] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. Model Architecture & Budget Accounting
    model = get_model(mechanism=args.mechanism, num_classes=2).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[Model: {args.mechanism.upper()}] Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 4. Optimization & Stopping Rule
    criterion = CombinedLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Record Metadata & Hyperparameters (Task 2 & 3 requirement)
    experiment_record = {
        "architecture": f"U-Net with {args.mechanism.upper()} bottleneck",
        "mechanism": args.mechanism,
        "seed": args.seed,
        "resolution": f"{res[0]}x{res[1]}",
        "augmentation": "RandomHorizontalFlip(p=0.5), ColorJitter(brightness=0.15, contrast=0.15)",
        "optimizer": {
            "type": "AdamW",
            "lr": args.lr,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR"
        },
        "stopping_rule": f"Fixed budget {args.epochs} epochs with checkpointing on best val mIoU",
        "parameter_budget": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        },
        "data_contract": DATA_CONTRACT
    }
    with open(os.path.join(run_dir, "experiment_record.json"), "w") as f:
        json.dump(experiment_record, f, indent=4)

    # 5. Training Loop with Epoch-by-Epoch Logging
    best_val_miou = 0.0
    history = []

    print("\n" + "=" * 80)
    print(f" STARTING REPRODUCIBLE EXPERIMENT: {args.mechanism.upper()}")
    print("=" * 80)
    header = f"{'Epoch':<7} | {'Train Loss':<12} | {'Train mIoU':<12} | {'Val Loss':<12} | {'Val mIoU':<12} | {'Pet IoU':<10} | {'Time':<8}"
    print(header)
    print("-" * 80)

    with open(log_file, "w") as log_f:
        log_f.write(f"Experiment: {args.mechanism.upper()}\n")
        log_f.write(f"Parameters: {total_params:,}\n")
        log_f.write(header + "\n")
        log_f.write("-" * 80 + "\n")

        for epoch in range(1, args.epochs + 1):
            t_start = time.time()
            train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t_start

            val_miou = val_metrics["mIoU"]
            pet_iou = val_metrics["per_class_iou"]["Pet"]

            line = f"{epoch:<7} | {train_loss:<12.4f} | {train_metrics['mIoU']:<12.4f} | {val_loss:<12.4f} | {val_miou:<12.4f} | {pet_iou:<10.4f} | {elapsed:<6.1f}s"
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mIoU": train_metrics["mIoU"],
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "val_pet_iou": pet_iou,
                "val_bg_iou": val_metrics["per_class_iou"]["Background"],
                "val_pixel_acc": val_metrics["pixel_accuracy"]
            })

            # Checkpoint best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)

    print("=" * 80)
    print(f"[*] Training finished! Best Validation mIoU: {best_val_miou:.4f}")
    print(f"[*] Checkpoints and logs saved to: {run_dir}")


if __name__ == "__main__":
    main()
