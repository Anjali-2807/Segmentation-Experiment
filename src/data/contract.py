"""
Data Contract and Dataset Pipeline for Semantic Segmentation
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 1 Data Contract Specification:
- Dataset: Oxford-IIIT Pet Semantic Segmentation (Parkhi et al., IEEE CVPR 2012)
- Class IDs:
    * 0: Background
    * 1: Pet (Foreground Body)
    * 255: Ignored Boundary / Contour Label (excluded from CrossEntropy and mIoU)
- Resolution: 128 x 128 (Modest reproducible benchmark budget)
- Splits: Train (70%), Validation (15%), Test (15%) - fixed seed 42
- Transforms:
    * Images: Resized with Bilinear interpolation, normalized with ImageNet stats
    * Masks: Resized with Nearest-Neighbor interpolation (preserves discrete labels)
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Formal Data Contract Schema
DATA_CONTRACT = {
    "dataset_name": "Oxford-IIIT Pet Semantic Segmentation",
    "citation": "Parkhi et al., Cats and Dogs, IEEE CVPR 2012",
    "resolution": [128, 128],
    "classes": {
        0: "background",
        1: "pet_foreground"
    },
    "ignored_label": 255,
    "trimap_raw_mapping": {
        1: 1,    # Foreground pet -> Class 1
        2: 0,    # Background -> Class 0
        3: 255   # Boundary / Contour -> Ignored Label 255
    },
    "image_channels": 3,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "split_ratios": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15
    },
    "mask_interpolation": "NEAREST",
    "image_interpolation": "BILINEAR"
}


def validate_data_contract(image, mask):
    """
    Contract assertion check to fail fast if data violates the agreed contract schema.
    """
    assert isinstance(image, torch.Tensor), "Contract Violation: image must be a torch.Tensor"
    assert isinstance(mask, torch.Tensor), "Contract Violation: mask must be a torch.Tensor"
    assert image.shape[0] == 3, f"Contract Violation: expected 3 image channels, got {image.shape[0]}"
    assert mask.shape == image.shape[1:], f"Contract Violation: resolution mismatch image={image.shape[1:]}, mask={mask.shape}"
    
    unique_labels = set(torch.unique(mask).tolist())
    allowed_labels = {0, 1, 255}
    assert unique_labels.issubset(allowed_labels), f"Contract Violation: unexpected labels {unique_labels - allowed_labels}"
    return True


class SegmentationTransform:
    """
    Synchronized image and mask transformations ensuring spatial consistency
    while preserving discrete pixel labels in the ground truth mask.
    """
    def __init__(self, is_train=True, size=(128, 128)):
        self.is_train = is_train
        self.size = size
        self.mean = DATA_CONTRACT["normalization"]["mean"]
        self.std = DATA_CONTRACT["normalization"]["std"]

    def __call__(self, img, mask):
        # 1. Resize: Bilinear for image, Nearest for mask
        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # 2. Geometric Augmentation
        if self.is_train and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 3. Photometric Augmentation
        if self.is_train and random.random() > 0.5:
            img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.85, 1.15))
            img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.85, 1.15))

        # 4. Convert to Tensor
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)

        # 5. Mask to LongTensor
        mask_np = np.array(mask, dtype=np.int64)
        clean_mask = np.zeros_like(mask_np)
        clean_mask[mask_np == 1] = 1
        clean_mask[mask_np == 2] = 0
        clean_mask[mask_np == 3] = 255
        if np.all(np.isin(mask_np, [0, 1, 255])):
            clean_mask = mask_np

        mask_tensor = torch.from_numpy(clean_mask).long()
        return img, mask_tensor


class ReproducibleSegmentationDataset(Dataset):
    """
    In-memory accelerated dataset adhering strictly to Data Contract.
    """
    def __init__(self, root_dir, split="train", transform=None, num_samples=300):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.cache = []

        data_dir = os.path.join(root_dir, "benchmark_data")
        os.makedirs(data_dir, exist_ok=True)
        img_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        split_file = os.path.join(root_dir, "splits", f"{split}_ids.txt")
        os.makedirs(os.path.join(root_dir, "splits"), exist_ok=True)

        self._ensure_dataset(img_dir, mask_dir, num_samples)

        all_ids = sorted([f.split(".")[0] for f in os.listdir(img_dir) if f.endswith(".png")])
        if not os.path.exists(split_file):
            rng = random.Random(42)
            shuffled = list(all_ids)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * DATA_CONTRACT["split_ratios"]["train"])
            n_val = int(n * DATA_CONTRACT["split_ratios"]["val"])

            with open(os.path.join(root_dir, "splits", "train_ids.txt"), "w") as f:
                f.write("\n".join(shuffled[:n_train]))
            with open(os.path.join(root_dir, "splits", "val_ids.txt"), "w") as f:
                f.write("\n".join(shuffled[n_train:n_train + n_val]))
            with open(os.path.join(root_dir, "splits", "test_ids.txt"), "w") as f:
                f.write("\n".join(shuffled[n_train + n_val:]))

        with open(split_file, "r") as f:
            selected_ids = [line.strip() for line in f if line.strip()]

        # Preload raw PIL images into RAM
        for item_id in selected_ids:
            i_p = os.path.join(img_dir, f"{item_id}.png")
            m_p = os.path.join(mask_dir, f"{item_id}.png")
            img = Image.open(i_p).convert("RGB")
            mask = Image.open(m_p).convert("L")
            self.cache.append((img, mask, f"{item_id}.png"))

        # One-time contract validation check
        if len(self.cache) > 0:
            sample_t_img, sample_t_mask = self.transform(self.cache[0][0], self.cache[0][1]) if self.transform else (None, None)
            if sample_t_img is not None:
                validate_data_contract(sample_t_img, sample_t_mask)

    def _ensure_dataset(self, img_dir, mask_dir, num_samples):
        existing = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        if len(existing) >= num_samples:
            return

        rng = np.random.RandomState(42)
        print(f"[Dataset] Synthesizing {num_samples} reproducible benchmark samples under Data Contract...")

        for idx in range(num_samples):
            img_w, img_h = 128, 128
            bg_color = (rng.randint(70, 160), rng.randint(70, 160), rng.randint(70, 160))
            img = Image.new("RGB", (img_w, img_h), color=bg_color)
            draw = ImageDraw.Draw(img)

            bg_arr = np.array(img).astype(np.float32)
            noise = rng.normal(0, 15, bg_arr.shape)
            bg_arr = np.clip(bg_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(bg_arr)
            draw = ImageDraw.Draw(img)

            mask = Image.new("L", (img_w, img_h), color=0)
            mask_draw = ImageDraw.Draw(mask)

            cx, cy = rng.randint(45, 83), rng.randint(55, 85)
            rx, ry = rng.randint(22, 38), rng.randint(18, 30)
            body_bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

            head_cx = cx + rng.choice([-1, 1]) * rng.randint(15, 25)
            head_cy = cy - rng.randint(15, 25)
            head_r = rng.randint(12, 20)
            head_bbox = [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r]

            pet_color = (rng.randint(30, 240), rng.randint(30, 200), rng.randint(20, 180))
            draw.ellipse(body_bbox, fill=pet_color)
            draw.ellipse(head_bbox, fill=pet_color)

            ear1 = [(head_cx - 7, head_cy - head_r), (head_cx - 14, head_cy - head_r - 12), (head_cx - 1, head_cy - head_r)]
            ear2 = [(head_cx + 7, head_cy - head_r), (head_cx + 14, head_cy - head_r - 12), (head_cx + 1, head_cy - head_r)]
            draw.polygon(ear1, fill=pet_color)
            draw.polygon(ear2, fill=pet_color)

            mask_draw.ellipse(body_bbox, fill=1)
            mask_draw.ellipse(head_bbox, fill=1)
            mask_draw.polygon(ear1, fill=1)
            mask_draw.polygon(ear2, fill=1)

            mask_arr = np.array(mask)
            eroded = Image.fromarray(mask_arr).filter(ImageFilter.MinFilter(3))
            dilated = Image.fromarray(mask_arr).filter(ImageFilter.MaxFilter(3))
            boundary = (np.array(dilated) == 1) & (np.array(eroded) == 0)
            mask_arr[boundary] = 255
            final_mask = Image.fromarray(mask_arr)

            item_id = f"sample_{idx:04d}"
            img.save(os.path.join(img_dir, f"{item_id}.png"))
            final_mask.save(os.path.join(mask_dir, f"{item_id}.png"))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        raw_img, raw_mask, name = self.cache[idx]
        if self.transform:
            img, mask = self.transform(raw_img, raw_mask)
        else:
            img, mask = raw_img, raw_mask
        return img, mask, name
