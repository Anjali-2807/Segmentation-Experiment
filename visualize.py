"""
Qualitative Evidence Generator and Failure Case Analysis
Author: Anjali Tiwari
Course: Advanced Computer Vision (CS107203CS) - Unit 3

Task 4 Visual Evidence:
- Renders side-by-side comparisons: Source Image | Ground Truth | Baseline | Dilated Variant | Error Map
- Produces explicit failure case analysis visualizations (boundary ambiguities, low contrast).
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from src.data.contract import (
    DATA_CONTRACT,
    SegmentationTransform,
    ReproducibleSegmentationDataset
)
from src.models.unet import get_model


def denormalize_image(tensor):
    """Denormalize ImageNet normalized tensor back to [0, 255] PIL image."""
    mean = np.array(DATA_CONTRACT["normalization"]["mean"])
    std = np.array(DATA_CONTRACT["normalization"]["std"])
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * std + mean) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def colorize_mask(mask_arr):
    """
    Colorizes segmentation mask:
    - 0 (Background): Dark slate (30, 30, 45)
    - 1 (Pet / Foreground): Emerald green (46, 204, 113)
    - 255 (Ignored boundary): Amber yellow (241, 196, 15)
    """
    h, w = mask_arr.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_img[mask_arr == 0] = [30, 30, 45]
    color_img[mask_arr == 1] = [46, 204, 113]
    color_img[mask_arr == 255] = [241, 196, 15]
    return Image.fromarray(color_img)


def colorize_prediction(pred_arr):
    """Colorizes model predictions (0: Background, 1: Pet)"""
    h, w = pred_arr.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_img[pred_arr == 0] = [30, 30, 45]
    color_img[pred_arr == 1] = [52, 152, 219] # Dodger blue
    return Image.fromarray(color_img)


def create_error_map(pred_arr, target_arr):
    """
    Highlights prediction errors:
    - True Positives (Pet): Green (46, 204, 113)
    - True Negatives (Background): Dark gray (35, 35, 35)
    - False Positives: Crimson red (231, 76, 60)
    - False Negatives: Purple (155, 89, 182)
    - Ignored Boundary: Amber yellow (241, 196, 15)
    """
    h, w = pred_arr.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)
    err[target_arr == 255] = [100, 100, 100] # ignored
    
    valid = (target_arr != 255)
    tp = (pred_arr == 1) & (target_arr == 1) & valid
    tn = (pred_arr == 0) & (target_arr == 0) & valid
    fp = (pred_arr == 1) & (target_arr == 0) & valid
    fn = (pred_arr == 0) & (target_arr == 1) & valid

    err[tn] = [35, 35, 35]
    err[tp] = [46, 204, 113]
    err[fp] = [231, 76, 60]   # False alarm (Red)
    err[fn] = [155, 89, 182]  # Missed region (Purple)
    return Image.fromarray(err)


def render_panel(images_with_titles, output_path, panel_title=""):
    """Stitches images horizontally with header labels into a clean composite figure."""
    tile_w, tile_h = 256, 256
    pad = 16
    header_h = 44
    top_bar_h = 40 if panel_title else 0
    total_w = pad + len(images_with_titles) * (tile_w + pad)
    total_h = top_bar_h + header_h + tile_h + pad

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    if panel_title:
        draw.text((pad, 12), panel_title, fill=(230, 230, 230), font=title_font)

    x = pad
    y = top_bar_h + header_h
    for title, img in images_with_titles:
        draw.text((x + 8, top_bar_h + 16), title, fill=(180, 180, 180), font=font)
        # Resize image to tile size if needed
        if img.size != (tile_w, tile_h):
            resample_mode = Image.Resampling.BILINEAR if title.startswith("1.") else Image.Resampling.NEAREST
            img = img.resize((tile_w, tile_h), resample=resample_mode)
        canvas.paste(img, (x, y))
        draw.rectangle([x, y, x + tile_w, y + tile_h], outline=(60, 60, 60), width=1)
        x += tile_w + pad

    canvas.save(output_path, "PNG")
    print(f"Generated figure: {output_path}")


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    res = tuple(DATA_CONTRACT["resolution"])
    test_transform = SegmentationTransform(is_train=False, size=res)
    test_ds = ReproducibleSegmentationDataset(project_root, split="test", transform=test_transform)

    device = torch.device("cpu")
    base_model = get_model(mechanism="baseline", num_classes=2).to(device)
    dil_model = get_model(mechanism="dilation", num_classes=2).to(device)

    base_ckpt = os.path.join(project_root, "results", "baseline", "best_model.pth")
    dil_ckpt = os.path.join(project_root, "results", "dilation", "best_model.pth")

    if os.path.exists(base_ckpt):
        base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
    if os.path.exists(dil_ckpt):
        dil_model.load_state_dict(torch.load(dil_ckpt, map_location=device))

    base_model.eval()
    dil_model.eval()

    # Select representative test cases: typical success & difficult failure cases
    sample_indices = [0, 4, 12]

    for idx, sample_idx in enumerate(sample_indices):
        img_t, mask_t, name = test_ds[sample_idx]
        with torch.no_grad():
            inp = img_t.unsqueeze(0).to(device)
            out_base = torch.argmax(base_model(inp), dim=1).squeeze(0).cpu().numpy()
            out_dil = torch.argmax(dil_model(inp), dim=1).squeeze(0).cpu().numpy()

        src_pil = denormalize_image(img_t)
        gt_arr = mask_t.numpy()
        gt_pil = colorize_mask(gt_arr)
        pred_base_pil = colorize_prediction(out_base)
        pred_dil_pil = colorize_prediction(out_dil)
        err_pil = create_error_map(out_dil, gt_arr)

        panel_items = [
            ("1. Input Image", src_pil),
            ("2. Ground Truth", gt_pil),
            ("3. Baseline (Conv)", pred_base_pil),
            ("4. Dilated (Atrous)", pred_dil_pil),
            ("5. Error Map (Dilated)", err_pil)
        ]

        if idx == 0:
            out_name = "01_qualitative_comparison.png"
            header = "Figure 1: Full Pipeline Qualitative Evaluation (Test Sample)"
        elif idx == 1:
            out_name = "02_boundary_refinement.png"
            header = "Figure 2: Contextual Coherence and Contour Resolution"
        else:
            out_name = "03_failure_cases_analysis.png"
            header = "Figure 3: Failure Case Analysis (Ambiguous Contours & Thin Regions)"

        render_panel(panel_items, os.path.join(figures_dir, out_name), panel_title=header)

    print(f"[*] All qualitative figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
