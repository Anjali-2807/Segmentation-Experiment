"""
Evaluation Metrics and Latency Benchmark for Semantic Segmentation
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 4 Evidence Metrics:
- Per-class Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)
- Overall Pixel Accuracy & Class Dice Score
- Parameter counting
- Rigorous hardware-calibrated latency benchmark
"""

import time
import numpy as np
import torch
import torch.nn.functional as F

class SegmentationMetrics:
    """
    Computes per-class IoU, mIoU, and Pixel Accuracy while strictly respecting
    the ignored label contract (label 255 excluded from metric calculations).
    """
    def __init__(self, num_classes=2, ignore_index=255, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or {0: "Background", 1: "Pet"}
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        """
        Vectorized confusion matrix accumulation ignoring label 255.
        preds: Tensor of shape (B, H, W)
        targets: Tensor of shape (B, H, W)
        """
        p = preds.detach().cpu().numpy().flatten()
        t = targets.detach().cpu().numpy().flatten()

        valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        v_p = p[valid]
        v_t = t[valid]

        if len(v_t) > 0:
            indices = self.num_classes * v_t + v_p
            cm = np.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
            self.confusion_matrix += cm

    def compute(self):
        ious = {}
        dices = {}
        total_intersect = 0
        total_union = 0

        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 0.0
            dice = (2.0 * tp / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

            name = self.class_names.get(c, f"Class_{c}")
            ious[name] = float(iou)
            dices[name] = float(dice)
            total_intersect += tp
            total_union += denom

        miou = float(np.mean(list(ious.values())))
        total_correct = np.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        pixel_acc = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        return {
            "per_class_iou": ious,
            "mIoU": miou,
            "per_class_dice": dices,
            "mean_dice": float(np.mean(list(dices.values()))),
            "pixel_accuracy": pixel_acc,
            "confusion_matrix": self.confusion_matrix.tolist()
        }


def measure_inference_latency(model, input_size=(1, 3, 256, 256), device="cpu", num_warmup=25, num_runs=100):
    """
    Measures empirical inference latency under strictly recorded conditions.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size, device=device)

    # Warmup phase to stabilize CPU thread dispatch and caching
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = float(1000.0 / mean_lat)

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "fps": round(fps, 1),
        "batch_size": input_size[0],
        "resolution": f"{input_size[2]}x{input_size[3]}",
        "device": str(device),
        "num_runs": num_runs
    }
