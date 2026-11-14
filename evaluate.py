"""
Exact Evaluation Script for Semantic Segmentation Benchmark
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 4 Evidence Reporting:
- Evaluates models on held-out test split under identical Data Contract transforms.
- Reports per-class IoU, mIoU, Pixel Accuracy, Parameter Count, and Measured Latency.
- Strictly adheres to ground truth evaluation (no pseudo ground truth).
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.contract import (
    DATA_CONTRACT,
    SegmentationTransform,
    ReproducibleSegmentationDataset
)
from src.models.unet import get_model, count_parameters
from src.utils.metrics import SegmentationMetrics, measure_inference_latency


def evaluate_model_on_test(model_path, mechanism, test_loader, device):
    model = get_model(mechanism=mechanism, num_classes=2).to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Eval] Loaded weights from: {model_path}")
    else:
        print(f"[Warning] Weights not found at {model_path}. Running with initialized weights.")

    model.eval()
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

    test_results = metrics.compute()
    params_total, params_trainable = count_parameters(model)
    res = tuple(DATA_CONTRACT["resolution"])
    latency_stats = measure_inference_latency(model, input_size=(1, 3, res[0], res[1]), device=device)

    return {
        "mechanism": mechanism,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "mIoU": round(test_results["mIoU"] * 100.0, 2),
        "pixel_accuracy": round(test_results["pixel_accuracy"] * 100.0, 2),
        "per_class_iou": {k: round(v * 100.0, 2) for k, v in test_results["per_class_iou"].items()},
        "per_class_dice": {k: round(v * 100.0, 2) for k, v in test_results["per_class_dice"].items()},
        "latency": latency_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Exact Evaluation Script")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing run results")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    project_root = os.path.dirname(os.path.abspath(__file__))
    res = tuple(DATA_CONTRACT["resolution"])
    test_transform = SegmentationTransform(is_train=False, size=res)
    test_ds = ReproducibleSegmentationDataset(project_root, split="test", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    print("=================================================================")
    print(f"   EVALUATION SCRIPT: HELD-OUT TEST DATASET ({len(test_ds)} samples)   ")
    print("=================================================================\n")

    baseline_ckpt = os.path.join(args.results_dir, "baseline", "best_model.pth")
    dilated_ckpt = os.path.join(args.results_dir, "dilation", "best_model.pth")

    print("[1/2] Evaluating Baseline (Standard Bottleneck)...")
    base_res = evaluate_model_on_test(baseline_ckpt, "baseline", test_loader, device)

    print("\n[2/2] Evaluating Controlled Variant (Dilated Atrous Bottleneck)...")
    dil_res = evaluate_model_on_test(dilated_ckpt, "dilation", test_loader, device)

    # Comparison summary table
    summary = {
        "evaluation_dataset": DATA_CONTRACT["dataset_name"],
        "test_samples": len(test_ds),
        "resolution": f"{res[0]}x{res[1]}",
        "baseline": base_res,
        "controlled_variant_dilation": dil_res,
        "delta": {
            "mIoU_gain": round(dil_res["mIoU"] - base_res["mIoU"], 2),
            "pet_iou_gain": round(dil_res["per_class_iou"]["Pet"] - base_res["per_class_iou"]["Pet"], 2),
            "param_diff": dil_res["parameters_total"] - base_res["parameters_total"],
            "latency_diff_ms": round(dil_res["latency"]["mean_latency_ms"] - base_res["latency"]["mean_latency_ms"], 2)
        }
    }

    out_file = os.path.join(args.results_dir, "evaluation_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=4)

    # Print formatted comparison table
    print("\n" + "=" * 85)
    print("                      EVALUATION EVIDENCE TABLE                          ")
    print("=" * 85)
    print(f" {'Metric / Condition':<28} | {'Baseline (Standard)':<24} | {'Variant (Dilation)':<24}")
    print("-" * 85)
    print(f" {'Parameters (M)':<28} | {base_res['parameters_total']/1e6:<24.3f} | {dil_res['parameters_total']/1e6:<24.3f}")
    print(f" {'Background IoU (%)':<28} | {base_res['per_class_iou']['Background']:<24.2f} | {dil_res['per_class_iou']['Background']:<24.2f}")
    print(f" {'Pet / Foreground IoU (%)':<28} | {base_res['per_class_iou']['Pet']:<24.2f} | {dil_res['per_class_iou']['Pet']:<24.2f}")
    print(f" {'Mean IoU (mIoU %)':<28} | {base_res['mIoU']:<24.2f} | {dil_res['mIoU']:<24.2f}")
    print(f" {'Pixel Accuracy (%)':<28} | {base_res['pixel_accuracy']:<24.2f} | {dil_res['pixel_accuracy']:<24.2f}")
    print(f" {'Mean Latency (ms)':<28} | {base_res['latency']['mean_latency_ms']:<24.2f} | {dil_res['latency']['mean_latency_ms']:<24.2f}")
    print(f" {'Throughput (FPS)':<28} | {base_res['latency']['fps']:<24.1f} | {dil_res['latency']['fps']:<24.1f}")
    print(f" {'Latency Device Condition':<28} | {base_res['latency']['device']:<24} | {dil_res['latency']['device']:<24}")
    print("=" * 85)
    print(f"[*] Evaluation summary written to: {out_file}\n")


if __name__ == "__main__":
    main()
