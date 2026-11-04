"""
FruitVision AI — Empirical EfficientNet-B0 Training Pipeline
Replaces synthetic image generation with a reproducible 2-stage transfer learning pipeline:
- Stage 1: Freeze EfficientNet-B0 feature extractor, train custom classification head.
- Stage 2: Unfreeze top convolutional blocks (stages 6-8) with differential learning rates.

Hardware:
- Automatic detection of CUDA (RTX 4050 Laptop GPU 6GB VRAM) and CPU fallback.
- Automatic Mixed Precision (torch.amp) on CUDA with GradScaler.
- Early stopping based on validation loss.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

from training.config import TrainingConfig
from training.dataset import create_dataloaders, CLASSES_4CLS
from training.preprocessing import IMAGENET_MEAN, IMAGENET_STD

def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Constructs EfficientNet-B0 with custom multi-layer classification head.
    Matches FruitVision's canonical architecture:
    Dropout(0.3) -> Linear(1280, 128) -> ReLU() -> Dropout(0.2) -> Linear(128, num_classes)
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )
    return model

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Any = None,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Execute single training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Execute validation epoch with torch.no_grad."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = correct / total if total > 0 else 0.0
    return val_loss, val_acc

def run_training(config: TrainingConfig = None, cli_args: argparse.Namespace = None) -> Dict[str, Any]:
    if config is None:
        config = TrainingConfig()

    # CLI overrides
    if cli_args:
        if cli_args.batch_size:
            config.batch_size = cli_args.batch_size
        if cli_args.epochs1 is not None:
            config.epochs_stage1 = cli_args.epochs1
        if cli_args.epochs2 is not None:
            config.epochs_stage2 = cli_args.epochs2

    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Runtime hardware detection
    if cli_args and cli_args.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*65)
    print("FRUITVISION AI — EFFICIENTNET-B0 EMPIRICAL TRAINING PIPELINE")
    print("="*65)
    print(f"Active Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        pin_memory = True
        use_amp = config.use_amp
    else:
        print("Hardware: CPU Mode (Multi-threaded fallback)")
        pin_memory = False
        use_amp = False

    print(f"Target Classes: {config.classes}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Stage 1 (Head Warmup): {config.epochs_stage1} epochs (LR={config.lr_stage1})")
    print(f"Stage 2 (Fine-Tuning): {config.epochs_stage2} epochs (Head LR={config.lr_stage2_head}, Backbone LR={config.lr_stage2_backbone})")
    print("="*65 + "\n")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        dataset_root=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        classes=config.classes
    )

    print(f"[Dataset] Train samples: {len(train_ds)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # Class weights for loss
    class_weights = train_ds.compute_class_weights().to(device)
    print(f"[Loss] Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)

    # Instantiate Model
    model = build_model(num_classes=config.num_classes, pretrained=True).to(device)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "stage": [],
        "stage1_end_epoch": config.epochs_stage1
    }

    start_time = time.time()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    # -------------------------------------------------------------
    # STAGE 1: Freeze Backbone, Train Classification Head
    # -------------------------------------------------------------
    print("[FruitVision AI] Starting Stage 1: Training Classification Head (Backbone Frozen)...")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=config.lr_stage1, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs_stage1 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_stage1, device, scaler, use_amp)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))
        history["stage"].append(1)

        print(f"Stage 1 | Epoch {epoch:2d}/{config.epochs_stage1} [{elapsed:4.1f}s] - Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:5.2f}% | Val Loss: {va_loss:.4f} Acc: {va_acc*100:5.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze Upper Convolutional Blocks (Stages 6 & 7 & 8)
    # -------------------------------------------------------------
    print(f"\n[FruitVision AI] Starting Stage 2: Fine-Tuning Top Blocks (Stages 6-8) with CosineAnnealingLR...")
    # Unfreeze stages 6 through 8 (the top blocks)
    for param in model.features[6:].parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW([
        {"params": model.features[6:].parameters(), "lr": config.lr_stage2_backbone},
        {"params": model.classifier.parameters(), "lr": config.lr_stage2_head}
    ], weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage2,
        T_max=config.epochs_stage2,
        eta_min=1e-6
    )

    patience_counter = 0

    for epoch_offset in range(1, config.epochs_stage2 + 1):
        epoch = config.epochs_stage1 + epoch_offset
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer_stage2, device, scaler, use_amp)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))
        history["stage"].append(2)

        print(f"Stage 2 | Epoch {epoch_offset:2d}/{config.epochs_stage2} (Total {epoch:2d}) [{elapsed:4.1f}s] - Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:5.2f}% | Val Loss: {va_loss:.4f} Acc: {va_acc*100:5.2f}%")

        if va_acc > best_val_acc or (abs(va_acc - best_val_acc) < 0.01 and va_loss < best_val_loss):
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  --> [NEW BEST] Val Acc: {va_acc*100:.2f}%, Val Loss: {va_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"  [Early Stopping] Validation loss failed to improve for {config.early_stopping_patience} consecutive epochs. Terminating Stage 2 early.")
                break

    total_training_duration = round(time.time() - start_time, 2)
    print(f"\n[FruitVision AI] Training Complete in {total_training_duration}s. Best Epoch: {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)")

    # Save Best Model Checkpoint
    os.makedirs(config.weights_dir, exist_ok=True)
    best_path = config.get_best_weights_path()
    backup_path = config.get_backup_weights_path()

    checkpoint_payload = {
        "model_state_dict": best_model_state,
        "classes": config.classes,
        "class_to_idx": {cls_name: i for i, cls_name in enumerate(config.classes)},
        "val_accuracy": float(best_val_acc),
        "val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "architecture": config.architecture,
        "history": history,
        "model_version": config.model_version,
        "input_resolution": [config.input_size, config.input_size, 3],
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "training_duration_seconds": total_training_duration,
        "device_used": str(device),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    torch.save(checkpoint_payload, best_path)
    torch.save(checkpoint_payload, backup_path)
    print(f"[FruitVision AI] Checkpoints successfully saved:")
    print(f"  Primary: {best_path}")
    print(f"  Backup:  {backup_path}")

    return checkpoint_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FruitVision AI Local Training")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs1", type=int, default=5, help="Stage 1 head warmup epochs")
    parser.add_argument("--epochs2", type=int, default=10, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training fallback")
    args = parser.parse_args()

    run_training(cli_args=args)
