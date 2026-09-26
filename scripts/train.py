"""
Training script for Hybrid Fatigue Monitoring Model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging
from pathlib import Path

from models.hybrid_model import create_model, HybridFatigueModel
from data.dataset import FatigueVideoDataset, collate_fn
from data.preprocessing import get_train_transforms, get_val_transforms, PreprocessingConfig
from utils.config_loader import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class FatigueLoss(nn.Module):
    """Multi-task loss for fatigue monitoring."""

    def __init__(self, config: dict):
        super().__init__()
        self.fatigue_weight = config.get("fatigue_weight", 1.0)
        self.workload_weight = config.get("workload_weight", 0.5)
        self.blink_weight = config.get("blink_weight", 0.3)

        self.fatigue_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.workload_loss = nn.MSELoss()
        self.blink_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        fatigue_loss = self.fatigue_loss(outputs.fatigue_logits, targets["fatigue"])
        workload_loss = self.workload_loss(outputs.workload_score.squeeze(-1), targets["workload"])
        blink_loss = self.blink_loss(outputs.blink_logits, targets["blink"])

        total_loss = (
            self.fatigue_weight * fatigue_loss +
            self.workload_weight * workload_loss +
            self.blink_weight * blink_loss
        )

        return {
            "total": total_loss,
            "fatigue": fatigue_loss,
            "workload": workload_loss,
            "blink": blink_loss,
        }


def train_epoch(model, loader, optimizer, criterion, device, scaler, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {"fatigue": 0, "workload": 0, "blink": 0}

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)
        physio = batch["physio"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

        optimizer.zero_grad()

        with autocast(enabled=config.get("mixed_precision", True)):
            outputs = model(frames, physio)
            losses = criterion(outputs, targets)
            loss = losses["total"]

        scaler.scale(loss).backward()

        # Gradient clipping
        if config.get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k].item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "fatigue": f"{losses['fatigue'].item():.4f}",
            "workload": f"{losses['workload'].item():.4f}",
            "blink": f"{losses['blink'].item():.4f}",
        })

    n = len(loader)
    return {
        "total": total_loss / n,
        **{k: v / n for k, v in loss_components.items()},
    }


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    loss_components = {"fatigue": 0, "workload": 0, "blink": 0}

    correct_fatigue = 0
    total_fatigue = 0
    workload_mae = 0.0
    correct_blink = 0
    total_blink = 0

    for batch in tqdm(loader, desc="Validation"):
        frames = batch["frames"].to(device, non_blocking=True)
        physio = batch["physio"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

        with autocast(enabled=config.get("mixed_precision", True)):
            outputs = model(frames, physio)
            losses = criterion(outputs, targets)

        total_loss += losses["total"].item()
        for k in loss_components:
            loss_components[k] += losses[k].item()

        # Metrics
        fatigue_pred = outputs.fatigue_logits.argmax(dim=-1)
        correct_fatigue += (fatigue_pred == targets["fatigue"]).sum().item()
        total_fatigue += targets["fatigue"].size(0)

        workload_mae += torch.abs(outputs.workload_score.squeeze(-1) - targets["workload"]).sum().item()

        blink_pred = outputs.blink_logits.argmax(dim=-1)
        correct_blink += (blink_pred == targets["blink"]).sum().item()
        total_blink += targets["blink"].size(0)

    n = len(loader)
    return {
        "total": total_loss / n,
        **{k: v / n for k, v in loss_components.items()},
        "fatigue_acc": correct_fatigue / total_fatigue,
        "workload_mae": workload_mae / total_fatigue,
        "blink_acc": correct_blink / total_blink,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Fatigue Monitoring Model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--data-root", type=str, required=True, help="Data root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sequence-length", type=int, default=30, help="Sequence length")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_config = config.get("training", {})
    train_config.update(vars(args))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    preprocess_config = PreprocessingConfig()
    train_transform = get_train_transforms(preprocess_config)
    val_transform = get_val_transforms(preprocess_config)

    train_dataset = FatigueVideoDataset(
        args.data_root,
        split="train",
        sequence_length=args.sequence_length,
        transform=train_transform,
    )
    val_dataset = FatigueVideoDataset(
        args.data_root,
        split="val",
        sequence_length=args.sequence_length,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model_config = config["model"]
    model_config["physiological"] = model_config.get("physiological", {})
    model_config["physiological"]["feature_dim"] = 32

    model = create_model(model_config)
    model.to(device)

    # Loss and optimizer
    criterion = FatigueLoss(train_config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.mixed_precision)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch, train_config)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, train_config)

        # Step scheduler
        scheduler.step()

        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['total']:.4f} "
            f"(Fatigue: {train_metrics['fatigue']:.4f}, "
            f"Workload: {train_metrics['workload']:.4f}, "
            f"Blink: {train_metrics['blink']:.4f}) | "
            f"Val Loss: {val_metrics['total']:.4f} "
            f"(Fatigue Acc: {val_metrics['fatigue_acc']:.4f}, "
            f"Workload MAE: {val_metrics['workload_mae']:.4f}, "
            f"Blink Acc: {val_metrics['blink_acc']:.4f})"
        )

        # Save checkpoint
        is_best = val_metrics["total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total"]

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "config": config,
        }, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, best_path)
            logger.info(f"Saved best model to {best_path}")

        # Clean old checkpoints (keep last 5)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        for old_ckpt in checkpoints[:-5]:
            old_ckpt.unlink()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()