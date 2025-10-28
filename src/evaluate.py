#!/usr/bin/env python3
import os
import json
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import config as cfg
from custom_logging import logger
from tools import (read_image, read_mask_png, read_mask_json_if_exists,
                   find_mask_for_image, save_overlay, to_img_name)

THRESHOLD = getattr(cfg, "THRESHOLD", 0.5)
IMG_SIZE = tuple(cfg.IMG_SIZE)  # (H, W)
RESULTS_DIR = Path(cfg.RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = RESULTS_DIR / "sample_predictions"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(cfg.MODEL_PATH)


# -----------------------
# Helpers
# -----------------------
def dice_np(y_true, y_pred, eps=1e-6):
    y_true_f = y_true.astype(np.float32).ravel()
    y_pred_f = y_pred.astype(np.float32).ravel()
    inter = np.sum(y_true_f * y_pred_f)
    denom = (np.sum(y_true_f) + np.sum(y_pred_f))
    if denom == 0:
        # both empty -> define dice as 1.0 (but we will report metrics both including and excluding empties)
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def iou_np(y_true, y_pred, eps=1e-6):
    y_true_f = y_true.astype(np.float32).ravel()
    y_pred_f = y_pred.astype(np.float32).ravel()
    inter = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - inter
    if union == 0:
        return 1.0
    return (inter + eps) / (union + eps)


# -----------------------
# Single-split evaluation
# -----------------------
def evaluate_split(model, img_dir: Path, mask_dir: Path, out_subdir: Path):
    out_subdir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")

    per_image = []
    dices = []
    ious = []
    dices_excl_empty = []
    ious_excl_empty = []

    stats = {
        "total_images": len(img_files),
        "evaluated": 0,
        "skipped_no_mask": 0,
        "empty_gt_count": 0,
        "json_flagged_empty": 0
    }

    for idx, img_path in enumerate(img_files, 1):
        mask_type, mask_path = find_mask_for_image(img_path, mask_dir)
        if mask_type is None:
            stats["skipped_no_mask"] += 1
            # still produce overlay with empty GT (optional) - here we skip inference if no mask
            logger.info(f"[{idx}/{len(img_files)}] ⚠️ No mask for {img_path.name}, skipping.")
            continue

        # load image
        img = read_image(img_path)

        # load mask
        if mask_type == "png":
            try:
                mask_true = read_mask_png(mask_path)
                json_flagged = False
            except FileNotFoundError:
                stats["skipped_no_mask"] += 1
                logger.info(f"[{idx}/{len(img_files)}] ⚠️ Couldn't open mask PNG for {img_path.name}, skipping.")
                continue
        else:
            mask_true, json_flagged = read_mask_json_if_exists(mask_path)
            if json_flagged:
                stats["json_flagged_empty"] += 1

        # ensure mask has IMG_SIZE
        if mask_true.shape != IMG_SIZE:
            mask_true = cv2.resize((mask_true * 255).astype(np.uint8),
                                   (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
            mask_true = (mask_true > 127).astype(np.uint8)

        # run model once
        inp = np.expand_dims(np.expand_dims(img, -1), 0)  # (1,H,W,1)
        try:
            pred = model.predict(inp, verbose=0)[0]
        except Exception as e:
            logger.warning(f"[{idx}/{len(img_files)}] model predict error for {img_path.name}: {e}")
            continue

        if pred.ndim == 3:
            pred = np.squeeze(pred, -1)
        # resize pred if needed
        if pred.shape != IMG_SIZE:
            pred = cv2.resize((pred * 255).astype(np.uint8), (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
            pred = pred.astype(np.float32) / 255.0
        pred = np.clip(pred, 0.0, 1.0)
        pred_mask = (pred >= THRESHOLD).astype(np.uint8)

        # compute metrics (including empties)
        d = float(dice_np(mask_true, pred_mask))
        j = float(iou_np(mask_true, pred_mask))
        dices.append(d)
        ious.append(j)

        # if GT non-empty, add to exclusive lists
        if mask_true.sum() == 0:
            stats["empty_gt_count"] += 1
        else:
            dices_excl_empty.append(d)
            ious_excl_empty.append(j)

        stats["evaluated"] += 1

        per_image.append({
            "image": img_path.name,
            "dice": d,
            "iou": j,
            "gt_sum": int(mask_true.sum()),
            "pred_sum": int(pred_mask.sum())
        })

        # save sample outputs
        stem = img_path.stem
        orig_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(str(out_subdir / to_img_name(stem, "image")), orig_uint8)
        cv2.imwrite(str(out_subdir / to_img_name(stem, "gt_mask")), (mask_true * 255).astype(np.uint8))
        cv2.imwrite(str(out_subdir / to_img_name(stem, "pred_mask")), (pred_mask * 255).astype(np.uint8))
        save_overlay(orig_uint8, mask_true, pred_mask, out_subdir / to_img_name(stem, "overlay"))

        if idx % 50 == 0 or idx == len(img_files):
            mean_inc = float(np.mean(dices)) if dices else 0.0
            mean_excl = float(np.mean(dices_excl_empty)) if dices_excl_empty else None
            logger.info(f"[{idx}/{len(img_files)}] Processed {idx}/{len(img_files)} - mean Dice (inc): {mean_inc:.4f}, "
                        f"mean Dice (excl-empty): {mean_excl if mean_excl is not None else 'N/A'}")

    # aggregate metrics
    mean_dice_incl = float(np.mean(dices)) if dices else 0.0
    mean_iou_incl = float(np.mean(ious)) if ious else 0.0
    mean_dice_excl = float(np.mean(dices_excl_empty)) if dices_excl_empty else None
    mean_iou_excl = float(np.mean(ious_excl_empty)) if ious_excl_empty else None

    return {
        "per_image": per_image,
        "metrics": {
            "mean_dice_including_empty": mean_dice_incl,
            "mean_iou_including_empty": mean_iou_incl,
            "mean_dice_excluding_empty_gt": mean_dice_excl,
            "mean_iou_excluding_empty_gt": mean_iou_excl
        },
        "counts": stats
    }


# -----------------------
# Main
# -----------------------
def main():
    try:
        # create lock
        pid = os.getpid()
        lock_data = {"pid": pid, "status": "running"}
        cfg.EVAL_LOCK_FILE.write_text(json.dumps(lock_data))
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            model = load_model(str(MODEL_PATH), compile=False)
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

        # evaluate two splits: train and test (user's cfg must point to correct folders)
        splits = {
            "train": (Path(cfg.TRAIN_IMG_DIR), Path(cfg.TRAIN_MASK_IMG_DIR)),
            "test": (Path(cfg.VAL_IMG_DIR), Path(cfg.VAL_MASK_IMG_DIR))  # adjust if your test has a separate cfg entry
        }

        # Define temp, backup directories
        temp_dir = str(SAMPLES_DIR) + "_temp"
        backup_dir = str(SAMPLES_DIR) + "_backup"

        # --- cleanup/recreate temp at the start ---
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        temp_dir = Path(temp_dir)
        logger.info(f"Working inside: {temp_dir}")

        overall_results = {}
        for name, (img_dir, mask_dir) in splits.items():
            logger.info(f"\n=== Evaluating split: {name} ===")
            if not img_dir.exists():
                logger.info(f"⚠️ Image dir for split '{name}' not found: {img_dir} (skipping split)")
                continue
            if not mask_dir.exists():
                logger.info(f"⚠️ Mask dir for split '{name}' not found: {mask_dir} (skipping split)")
                continue

            out_subdir = temp_dir / name
            out_subdir.mkdir(parents=True, exist_ok=True)
            split_res = evaluate_split(model, img_dir, mask_dir, out_subdir)
            overall_results[name] = split_res

            logger.info(f"✅ Split '{name}' done. Metrics (including empties): "
                        f"Dice={split_res['metrics']['mean_dice_including_empty']:.4f}, "
                        f"IoU={split_res['metrics']['mean_iou_including_empty']:.4f}")
            excl = split_res['metrics']['mean_dice_excluding_empty_gt']
            logger.info(f"   mean Dice (excluding empty GTs): {excl if excl is not None else 'N/A'}")
            logger.info(f"   Overlays saved to: {out_subdir}")

        # --- finalize: replace original with temp ---
        if os.path.exists(SAMPLES_DIR):
            logger.info(f"Backing up existing samples dir to {backup_dir}")
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)  # clear old backup if it exists
            os.rename(SAMPLES_DIR, backup_dir)  # safe rename instead of deleting

        # Move temp to final
        os.rename(temp_dir, SAMPLES_DIR)
        logger.info(f"Replaced {SAMPLES_DIR} with {temp_dir}")

        # write full metrics file
        metrics_path = RESULTS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(overall_results, f, indent=4)

        logger.info(f"\n📊 All done. Metrics saved to: {metrics_path}")
        logger.info(f"🖼️ Sample predictions saved under: {SAMPLES_DIR}")
    except Exception as e:
        with open(cfg.EVAL_LOCK_FILE, "a") as log:
            log.write(f"Error:\n{e}")
            logger.exception(traceback.format_exc())
        raise
    finally:
        # update lock when done
        done_data = {"pid": None, "status": "done"}
        cfg.EVAL_LOCK_FILE.write_text(json.dumps(done_data))


if __name__ == "__main__":
    main()
