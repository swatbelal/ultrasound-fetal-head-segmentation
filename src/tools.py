#!/usr/bin/env python3
import cv2
import json
import warnings
import numpy as np

import config as cfg

IMG_SIZE = tuple(cfg.IMG_SIZE)  # (H, W)


def read_image(path, img_size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0  # (H,W) float32


def read_mask_png(path, img_size=IMG_SIZE):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    mask = cv2.resize(mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def read_mask_json_if_exists(json_path, img_size=IMG_SIZE):
    # quick heuristic: if objects empty -> empty mask; if not empty -> warn and return empty
    with open(json_path, "r") as f:
        j = json.load(f)
    objects = j.get("objects", [])
    if not objects:
        return np.zeros(img_size, dtype=np.uint8), True
    warnings.warn(
        f"JSON {json_path} contains {len(objects)} objects but JSON->mask conversion is not implemented; "
        "treating as empty mask for now."
    )
    return np.zeros(img_size, dtype=np.uint8), True


def find_mask_for_image(img_path, mask_dir):
    """Look for PNG mask; falling back to .json or .png.json (treated as empty for now)."""
    stem = img_path.stem
    candidates = [
        mask_dir / (stem + ".png"),
        mask_dir / (stem + ".PNG"),
        mask_dir / (stem + ".jpg"),
        mask_dir / (stem + ".jpeg"),
    ]
    for c in candidates:
        if c.exists():
            return "png", c
    # check for json variants
    j1 = mask_dir / (stem + ".json")
    if j1.exists():
        return "json", j1
    j2 = mask_dir / (stem + ".png.json")
    if j2.exists():
        return "json", j2
    return None, None


def save_overlay(orig_gray, gt_mask, pred_mask, out_path, alpha=0.5):
    if orig_gray.dtype != np.uint8:
        base = (orig_gray * 255).astype(np.uint8)
    else:
        base = orig_gray.copy()
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    gt_col = np.zeros((h, w, 3), dtype=np.uint8)
    gt_col[:, :, 2] = (gt_mask * 255).astype(np.uint8)  # red channel
    pred_col = np.zeros((h, w, 3), dtype=np.uint8)
    pred_col[:, :, 1] = (pred_mask * 255).astype(np.uint8)  # green channel
    overlay = cv2.addWeighted(base_bgr, 1.0, gt_col, alpha, 0)
    overlay = cv2.addWeighted(overlay, 1.0, pred_col, alpha, 0)
    cv2.imwrite(str(out_path), overlay)


def to_img_name(stem, suffix):
    return f"{stem}_{suffix}.png"
