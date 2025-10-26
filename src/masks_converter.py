#!/usr/bin/env python3
import os
import cv2
import json
import zlib
import base64
import numpy as np
from pathlib import Path

import config as cfg


def decode_bitmap(bitmap_obj, img_size):
    """Decode base64 + zlib compressed bitmap mask."""
    data = bitmap_obj["data"]
    origin = bitmap_obj["origin"]

    # Decode from base64
    compressed_bytes = base64.b64decode(data)

    # Decompress (zlib)
    png_bytes = zlib.decompress(compressed_bytes)

    # Convert bytes → numpy array
    nparr = np.frombuffer(png_bytes, np.uint8)
    mask_patch = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Create blank mask
    mask = np.zeros((img_size["height"], img_size["width"]), dtype=np.uint8)

    # Place decoded patch at origin
    x, y = origin
    h, w = mask_patch.shape[:2]
    mask[y:y + h, x:x + w] = mask_patch[:, :, 3]  # take alpha channel

    return mask


def convert_json_to_mask(json_file, output_dir):
    stem = Path(json_file).stem  # e.g. "100_2HC.png" or "100_2HC"
    if not stem.endswith(".png"):
        stem = stem + ".png"  # only append if it's not already there

    out_path = Path(output_dir) / stem

    if out_path.exists():
        print(f"⏩ Skipping (already exists): {out_path}")
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    img_size = data["size"]
    objects = data["objects"]

    full_mask = np.zeros((img_size["height"], img_size["width"]), dtype=np.uint8)

    for obj in objects:
        if obj["geometryType"] == "bitmap":
            mask = decode_bitmap(obj["bitmap"], img_size)
            full_mask = np.maximum(full_mask, mask)

    cv2.imwrite(str(out_path), full_mask)
    print(f"✅ Saved mask: {out_path}")


def convert_all(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            convert_json_to_mask(os.path.join(json_dir, file), output_dir)


if __name__ == "__main__":
    convert_all(cfg.TRAIN_MASK_DIR, cfg.DATA_DIR / "train" / "masks_png")
    convert_all(cfg.VAL_MASK_DIR, cfg.DATA_DIR / "test" / "masks_png")
