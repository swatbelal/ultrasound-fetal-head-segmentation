#!/usr/bin/env python3
"""
inference.py

Usage examples:
  python inference.py --model /path/to/unet.h5 --input /path/to/image.png
  python inference.py --model /path/to/unet.h5 --input_dir /path/to/images --outdir ./results/infer --threshold 0.5
"""
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

import config as cfg
from tools import read_image, save_overlay

IMG_SIZE = tuple(cfg.IMG_SIZE)
DEFAULT_THRESHOLD = getattr(cfg, "THRESHOLD", 0.5)


def infer_image(model, img_path, out_dir: Path, threshold=DEFAULT_THRESHOLD, save_prob=False, gt_mask_path=None):
    img = read_image(img_path)
    inp = np.expand_dims(np.expand_dims(img, -1), 0)
    pred = model.predict(inp, verbose=0)[0]
    if pred.ndim == 3:
        pred = np.squeeze(pred, -1)
    pred = cv2.resize((pred * 255).astype(np.uint8), (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.astype(np.float32) / 255.0
    pred_mask = (pred >= threshold).astype(np.uint8)

    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # save original, prob map (optional), mask, overlay
    orig_uint8 = (img * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_image.png"), orig_uint8)
    cv2.imwrite(str(out_dir / f"{stem}_pred_mask.png"), (pred_mask * 255).astype(np.uint8))
    save_overlay(orig_uint8, np.zeros_like(pred_mask) if gt_mask_path is None else cv2.imread(str(gt_mask_path),
                                                                                              cv2.IMREAD_GRAYSCALE),
                 pred_mask, out_dir / f"{stem}_overlay.png")

    if save_prob:
        # save as .npy and a visually scaled grayscale heatmap
        np.save(str(out_dir / f"{stem}_prob.npy"), pred)
        prob_vis = (pred * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_prob.png"), prob_vis)

    return pred_mask


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved model (.h5 or TF SavedModel dir)")
    p.add_argument("--input", help="Single image path")
    p.add_argument("--input_dir", help="Directory of images")
    p.add_argument("--outdir", default=str(Path(cfg.RESULTS_DIR) / "inference"), help="Output directory")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--save_prob", action="store_true", help="Save probability maps (npy + png)")
    args = p.parse_args(argv)

    model = load_model(args.model, compile=False)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.input:
        paths = [Path(args.input)]
    elif args.input_dir:
        paths = sorted(
            [p for p in Path(args.input_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    else:
        raise SystemExit("Provide --input or --input_dir")

    for pth in paths:
        try:
            infer_image(model, pth, outdir, threshold=args.threshold, save_prob=args.save_prob)
            print(f"Saved outputs for {pth.name}")
        except Exception as e:
            print(f"Failed {pth}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
