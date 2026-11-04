"""
FruitVision AI — Independent Model Validation, Calibration & Robustness Suite
Phase 2 Execution
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    brier_score_loss
)

from training.train_local import build_model
from training.dataset import create_dataloaders
from training.preprocessing import inference_transform, IMAGENET_MEAN, IMAGENET_STD

def compute_ece(probs, targets, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    ece = 0.0
    bin_details = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            bin_details.append({
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": int(np.sum(in_bin)),
                "confidence": round(float(avg_confidence_in_bin), 4),
                "accuracy": round(float(accuracy_in_bin), 4),
                "diff": round(float(abs_diff), 4)
            })

    return float(ece), bin_details

def run_independent_validation():
    device = torch.device("cpu") # Validating on CPU to ensure 100% reproducibility
    weights_path = "weights/best.pt"
    dataset_root = "dataset/processed"

    print(f"Loading checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    classes = ckpt.get("classes")
    class_to_idx = ckpt.get("class_to_idx")
    print(f"Classes: {classes}")
    print(f"Class to index: {class_to_idx}")

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Independent evaluation on held-out test dataset
    _, _, test_loader, test_ds = create_dataloaders(
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
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    n_test = len(all_targets)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, c in enumerate(classes):
        per_class_metrics[c] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(per_class_supp[i])
        }

    # Calibration analysis
    ece, bin_details = compute_ece(all_probs, all_targets)
    one_hot_targets = np.zeros_like(all_probs)
    for i, t in enumerate(all_targets):
        one_hot_targets[i, t] = 1.0
    brier = float(brier_score_loss(one_hot_targets.ravel(), all_probs.ravel()))
    mean_confidence = float(np.mean(np.max(all_probs, axis=1)))

    print("\n--- HELD-OUT TEST RESULTS ---")
    print(f"Total samples: {n_test}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")
    print(f"Weighted F1: {weighted_f1*100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}%")
    print(f"Brier Score: {brier:.4f}")
    print(f"Mean Max Confidence: {mean_confidence*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    # 2. Perturbation & Robustness Analysis on Test Set
    test_files_by_class = defaultdict(list)
    test_dir = Path("dataset/processed/test")
    for c in classes:
        for f in (test_dir / c).glob("*.*"):
            test_files_by_class[c].append(f)

    def evaluate_transformed_images(transform_fn, desc):
        corr_correct = 0
        corr_total = 0
        confidences = []
        for c_idx, c in enumerate(classes):
            for f in test_files_by_class[c]:
                try:
                    with Image.open(f) as im:
                        im = im.convert("RGB")
                        im_mod = transform_fn(im)
                        t = inference_transform(im_mod).unsqueeze(0)
                        with torch.no_grad():
                            out = model(t)
                            pr = torch.softmax(out, dim=1).numpy()[0]
                        p_idx = np.argmax(pr)
                        confidences.append(float(pr[p_idx]))
                        if p_idx == c_idx:
                            corr_correct += 1
                        corr_total += 1
                except Exception as e:
                    pass
        acc_mod = float(corr_correct / corr_total) if corr_total > 0 else 0.0
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return {"accuracy": round(acc_mod, 4), "mean_confidence": round(mean_conf, 4), "total": corr_total, "correct": corr_correct}

    robustness_results = {}
    print("\n--- RUNNING ROBUSTNESS EVALUATION ---")

    # A. Brightness
    robustness_results["brightness_+30%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Brightness(im).enhance(1.3), "Brightness +30%"
    )
    robustness_results["brightness_-30%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Brightness(im).enhance(0.7), "Brightness -30%"
    )
    robustness_results["brightness_+60%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Brightness(im).enhance(1.6), "Brightness +60%"
    )
    robustness_results["brightness_-60%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Brightness(im).enhance(0.4), "Brightness -60%"
    )

    # B. Contrast
    robustness_results["contrast_+40%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Contrast(im).enhance(1.4), "Contrast +40%"
    )
    robustness_results["contrast_-40%"] = evaluate_transformed_images(
        lambda im: ImageEnhance.Contrast(im).enhance(0.6), "Contrast -40%"
    )

    # C. Blur
    robustness_results["gaussian_blur_radius_2"] = evaluate_transformed_images(
        lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)), "Gaussian Blur R=2"
    )
    robustness_results["gaussian_blur_radius_4"] = evaluate_transformed_images(
        lambda im: im.filter(ImageFilter.GaussianBlur(radius=4)), "Gaussian Blur R=4"
    )

    # D. Gaussian Noise
    def add_gaussian_noise(im, std=25):
        arr = np.array(im, dtype=np.float32)
        noise = np.random.normal(0, std, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    robustness_results["gaussian_noise_std_20"] = evaluate_transformed_images(
        lambda im: add_gaussian_noise(im, std=20), "Gaussian Noise std=20"
    )
    robustness_results["gaussian_noise_std_40"] = evaluate_transformed_images(
        lambda im: add_gaussian_noise(im, std=40), "Gaussian Noise std=40"
    )

    # E. Background shift simulation (replace background pixels with black or white)
    def mask_background(im, color=(0, 0, 0)):
        arr = np.array(im)
        gray = np.mean(arr, axis=2)
        bg = gray > 230
        arr[bg] = color
        return Image.fromarray(arr)

    robustness_results["dark_background_swap"] = evaluate_transformed_images(
        lambda im: mask_background(im, color=(20, 20, 20)), "Dark Background Swap"
    )

    for k, v in robustness_results.items():
        print(f"Robustness [{k:<24}]: Accuracy: {v['accuracy']*100:5.2f}% | Mean Conf: {v['mean_confidence']*100:5.2f}%")

    # 3. Completely New External Images Evaluation
    print("\n--- EXTERNAL UNSEEN IMAGES EVALUATION ---")
    external_images = [
        {"name": "images(6).jpg", "path": "sample_images/images(6).jpg", "true_class": "Apple_Bad", "description": "Web rotten apple with mold"},
        {"name": "images(7).jpg", "path": "sample_images/images(7).jpg", "true_class": "Apple_Bad", "description": "Web severely decayed apple"},
        {"name": "images(8).jpg", "path": "sample_images/images(8).jpg", "true_class": "Apple_Bad", "description": "Web brown rotting apple"},
        {"name": "images(10).jpg", "path": "sample_images/images(10).jpg", "true_class": "Apple_Bad", "description": "Web necrotic rot apple"},
        {"name": "sample_good_banana.jpg", "path": "sample_images/sample_good_banana.jpg", "true_class": "Banana_Good", "description": "External fresh banana bunch"},
        {"name": "sample_not_fruit.jpg", "path": "sample_images/sample_not_fruit.jpg", "true_class": "Non_Fruit", "description": "External non-fruit object"}
    ]

    external_results = []
    for item in external_images:
        if not os.path.exists(item["path"]):
            continue
        with Image.open(item["path"]) as im:
            im_rgb = im.convert("RGB")
            t = inference_transform(im_rgb).unsqueeze(0)
            with torch.no_grad():
                out = model(t)
                probs = torch.softmax(out, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            conf = float(probs[pred_idx])

            external_results.append({
                "file": item["name"],
                "description": item["description"],
                "true_class": item["true_class"],
                "predicted_class": pred_class,
                "confidence": round(conf, 4),
                "probabilities": {classes[j]: round(float(probs[j]), 4) for j in range(len(classes))},
                "is_correct": (pred_class == item["true_class"])
            })

    # Also test on 50 completely new unselected Lime images from raw dataset (Out-of-domain evaluation)
    lime_results = {"Lime_Bad": [], "Lime_Good": []}
    for l_cls in ["Lime_Bad", "Lime_Good"]:
        l_files = sorted(list(Path("dataset", l_cls).glob("*.*")))[:25]
        for lf in l_files:
            with Image.open(lf) as im:
                im_rgb = im.convert("RGB")
                t = inference_transform(im_rgb).unsqueeze(0)
                with torch.no_grad():
                    out = model(t)
                    probs = torch.softmax(out, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                lime_results[l_cls].append({
                    "file": lf.name,
                    "predicted_as": classes[pred_idx],
                    "confidence": round(float(probs[pred_idx]), 4)
                })

    print("\nExternal Real Photos Evaluation:")
    for er in external_results:
        print(f"  {er['file']:<25} True: {er['true_class']:<12} Pred: {er['predicted_class']:<12} Conf: {er['confidence']*100:5.2f}% ({er['is_correct']})")

    # 4. Assemble validation report payload
    validation_payload = {
        "checkpoint": weights_path,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "test_dataset_size": n_test,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_metrics
        },
        "calibration": {
            "expected_calibration_error": round(ece, 4),
            "brier_score": round(brier, 4),
            "mean_confidence": round(mean_confidence, 4),
            "bin_details": bin_details
        },
        "robustness_perturbations": robustness_results,
        "external_images_evaluation": external_results,
        "out_of_domain_lime_evaluation": {
            "lime_bad_predictions": lime_results["Lime_Bad"][:10],
            "lime_good_predictions": lime_results["Lime_Good"][:10],
            "summary": "Since the 4-class model has no Lime class, Lime images are forced into Apple/Banana categories."
        }
    }

    out_path = Path("reports/independent_model_validation_data.json")
    with open(out_path, "w") as f:
        json.dump(validation_payload, f, indent=2)
    print(f"\nSaved independent validation data to: {out_path}")

if __name__ == "__main__":
    run_independent_validation()
