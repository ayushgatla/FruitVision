"""
FruitVision AI — Empirical Model Evaluation, Verification & Benchmarking Script
Evaluates EfficientNet-B0 exclusively on held-out empirical test samples (138 real images).
Calculates authentic metrics (Accuracy, Macro/Weighted F1, Per-class Precision/Recall),
computes CUDA and CPU latency percentiles with GPU synchronization,
and plots genuine confusion matrix and training convergence curves.

Outputs:
- reports/classification_report.json
- reports/benchmark_report.json
- reports/confusion_matrix.png
- reports/training_curves.png
- training/confusion_matrix.png (legacy sync)
- training/loss_accuracy_curve.png (legacy sync)
"""

import os
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from training.config import TrainingConfig
from training.dataset import create_dataloaders, CLASSES_4CLS
from training.train_local import build_model
from training.preprocessing import inference_transform

def evaluate_model(
    weights_path: str = "weights/best.pt",
    dataset_root: str = "dataset/processed",
    reports_dir: str = "reports",
    benchmark_iterations: int = 100
) -> Dict[str, Any]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    legacy_training_dir = Path("training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*65)
    print("FRUITVISION AI — EMPIRICAL TEST EVALUATION & BENCHMARK")
    print("="*65)
    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Dataset: {dataset_root}")

    # Load Checkpoint
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    classes = ckpt.get("classes", CLASSES_4CLS)
    num_classes = len(classes)
    print(f"Active Checkpoint Classes ({num_classes}): {classes}")

    model = build_model(num_classes=num_classes, pretrained=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Load Real Held-Out Test Set
    _, _, test_loader, _ = create_dataloaders(
        dataset_root=dataset_root,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        classes=classes
    )

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    total_test_samples = len(all_targets)

    # Compute Core Metrics
    acc = float(accuracy_score(all_targets, all_preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, cls_name in enumerate(classes):
        per_class_metrics[cls_name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    report_text = classification_report(all_targets, all_preds, target_names=classes, digits=4, zero_division=0)

    # Safety Metric: Rotten/Defect Miss Rate (False Negative Rate for Bad fruit)
    bad_classes_idx = [i for i, c in enumerate(classes) if "bad" in c.lower() or "rotten" in c.lower()]
    bad_total = sum(per_class_supp[i] for i in bad_classes_idx)
    bad_correct = sum(cm[i, i] for i in bad_classes_idx)
    bad_miss_rate = round(float((bad_total - bad_correct) / bad_total), 4) if bad_total > 0 else 0.0

    print("\n" + "-"*65)
    print("HELD-OUT TEST RESULTS:")
    print(f"Total Test Samples: {total_test_samples}")
    print(f"Overall Accuracy:  {acc*100:6.2f}%")
    print(f"Macro Precision:   {macro_p*100:6.2f}%")
    print(f"Macro Recall:      {macro_r*100:6.2f}%")
    print(f"Macro F1-Score:    {macro_f1*100:6.2f}%")
    print(f"Weighted F1-Score: {weighted_f1*100:6.2f}%")
    print(f"Defect Miss Rate (FNR on Bad): {bad_miss_rate*100:5.2f}%")
    print("\nPer-Class Breakdown:")
    print(report_text)
    print("-"*65)

    # Plot Genuine Confusion Matrix
    plt.figure(figsize=(8, 6.5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annotations = np.empty_like(cm, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annotations[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 11}
    )
    plt.title(f"FruitVision AI — Held-Out Test Confusion Matrix\n(Accuracy: {acc*100:.2f}% | N={total_test_samples})", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Actual True Class", fontsize=11)
    plt.tight_layout()

    cm_path = reports_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    shutil.copy2(cm_path, legacy_training_dir / "confusion_matrix.png")
    print(f"[Artifact] Saved Confusion Matrix to: {cm_path}")

    # Plot Genuine Training & Validation Curves from Logged History
    history = ckpt.get("history", None)
    if history and "train_loss" in history and len(history["train_loss"]) > 0:
        epochs = history["epochs"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))

        # Accuracy Plot
        ax[0].plot(epochs, history["train_acc"], "o-", color="#1f77b4", label="Train Accuracy", linewidth=2)
        ax[0].plot(epochs, history["val_acc"], "s-", color="#2ca02c", label="Val Accuracy", linewidth=2)
        ax[0].axhline(y=0.90, color="r", linestyle="--", alpha=0.7, label="90% Target")
        if "stage1_end_epoch" in history:
            ax[0].axvline(x=history["stage1_end_epoch"] + 0.5, color="gray", linestyle=":", label="Stage 2 Unfreeze")
        ax[0].set_title("Empirical Training & Validation Accuracy", fontsize=12)
        ax[0].set_xlabel("Epoch", fontsize=11)
        ax[0].set_ylabel("Accuracy", fontsize=11)
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc="lower right")

        # Loss Plot
        ax[1].plot(epochs, history["train_loss"], "o-", color="#ff7f0e", label="Train Loss", linewidth=2)
        ax[1].plot(epochs, history["val_loss"], "s-", color="#d62728", label="Val Loss", linewidth=2)
        if "stage1_end_epoch" in history:
            ax[1].axvline(x=history["stage1_end_epoch"] + 0.5, color="gray", linestyle=":", label="Stage 2 Unfreeze")
        ax[1].set_title("Empirical Training & Validation Loss", fontsize=12)
        ax[1].set_xlabel("Epoch", fontsize=11)
        ax[1].set_ylabel("CrossEntropy Loss", fontsize=11)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc="upper right")

        plt.tight_layout()
        curves_path = reports_path / "training_curves.png"
        plt.savefig(curves_path, dpi=200)
        plt.close()
        shutil.copy2(curves_path, legacy_training_dir / "loss_accuracy_curve.png")
        print(f"[Artifact] Saved Training Curves to: {curves_path}")

    # Latency Benchmark
    print(f"\n[Benchmark] Running {benchmark_iterations} inference iterations on {device}...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 10 Warmup iterations
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed iterations
    latencies_ms = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    p99_lat = float(np.percentile(latencies_ms, 99))
    fps = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

    print(f"[Benchmark] Mean: {mean_lat:.2f}ms | Median (P50): {median_lat:.2f}ms | P95: {p95_lat:.2f}ms | Throughput: {fps} FPS")

    # Save Classification Report JSON
    eval_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights_path": weights_path,
        "model_architecture": ckpt.get("architecture", "EfficientNetB0"),
        "model_version": ckpt.get("model_version", "2.0.0-efficientnet-b0-4cls"),
        "total_test_samples": total_test_samples,
        "classes": classes,
        "accuracy": round(acc, 4),
        "macro_metrics": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1_score": round(float(macro_f1), 4)
        },
        "weighted_metrics": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1_score": round(float(weighted_f1), 4)
        },
        "defect_miss_rate_bad_fruit": bad_miss_rate,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": classes,
            "matrix": cm.tolist()
        }
    }

    eval_json_path = reports_path / "classification_report.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Save Benchmark Report JSON
    benchmark_results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "iterations": benchmark_iterations,
        "batch_size": 1,
        "input_resolution": [224, 224, 3],
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "median_p50": round(median_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2)
        },
        "throughput_fps": fps
    }

    bench_json_path = reports_path / "benchmark_report.json"
    with open(bench_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"[Artifact] Saved classification report to: {eval_json_path}")
    print(f"[Artifact] Saved latency benchmark to: {bench_json_path}")

    return {
        "classification_report": eval_results,
        "benchmark_report": benchmark_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FruitVision AI Empirical Evaluation")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to weights file")
    parser.add_argument("--dataset", type=str, default="dataset/processed", help="Path to processed dataset root")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        dataset_root=args.dataset,
        benchmark_iterations=args.iterations
    )
