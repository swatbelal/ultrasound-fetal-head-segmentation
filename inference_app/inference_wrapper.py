from pathlib import Path
import importlib, numpy as np, cv2
from tensorflow.keras.models import load_model

try:
    import config as cfg
except Exception as e:
    raise RuntimeError('Could not import src.config; ensure src is on PYTHONPATH and src/config.py exists') from e

MODEL_PATH = Path(cfg.MODEL_PATH)
IMG_SIZE = tuple(cfg.IMG_SIZE)
THRESHOLD = getattr(cfg, 'THRESHOLD', 0.5)
RESULTS_DIR = Path(cfg.RESULTS_DIR)
SAMPLES_DIR = RESULTS_DIR / 'sample_predictions'
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

inference_module = None
try:
    inference_module = importlib.import_module('inference')  # src/inference.py
except Exception:
    inference_module = None

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    _model = load_model(str(MODEL_PATH), compile=False)
    return _model


def read_image_fallback(path):
    if inference_module is not None and hasattr(inference_module, 'read_image'):
        return inference_module.read_image(path, img_size=IMG_SIZE)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
    return img.astype('float32') / 255.0


def predict_image_path(image_path, out_dir=None, stem_override=None):
    image_path = Path(image_path)
    out_dir = Path(out_dir) if out_dir else None
    if inference_module is not None and hasattr(inference_module, 'infer_image'):
        try:
            # src/inference.infer_image(model, img_path, out_dir, threshold=...)
            model = get_model()
            pred_mask = inference_module.infer_image(model, image_path, out_dir or SAMPLES_DIR, threshold=THRESHOLD)
            return {'image': image_path, 'pred_mask': pred_mask, 'saved_dir': str(out_dir) if out_dir else None,
                    'stem': stem_override or image_path.stem}
        except Exception:
            pass

    model = get_model()
    img = read_image_fallback(image_path)
    inp = np.expand_dims(np.expand_dims(img, -1), 0)
    pred = model.predict(inp, verbose=0)[0]
    if pred.ndim == 3:
        pred = np.squeeze(pred, -1)
    pred = np.clip(pred, 0.0, 1.0)
    pred_mask = (pred >= THRESHOLD).astype('uint8')

    stem = stem_override or image_path.stem
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        orig_uint8 = (img * 255).astype('uint8')
        cv2.imwrite(str(out_dir / f"{stem}_image.png"), orig_uint8)
        cv2.imwrite(str(out_dir / f"{stem}_pred_mask.png"), (pred_mask * 255).astype('uint8'))
        base_bgr = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2BGR)
        pred_col = base_bgr.copy() * 0
        pred_col[:, :, 1] = (pred_mask * 255).astype('uint8')
        overlay = cv2.addWeighted(base_bgr, 1.0, pred_col, 0.5, 0)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    return {'image': image_path, 'pred_prob': pred, 'pred_mask': pred_mask,
            'saved_dir': str(out_dir) if out_dir else None, 'stem': stem}


def run_on_dir(img_dir, mask_dir=None, results_out_dir=None, skip_empty_gt=True):
    img_dir = Path(img_dir)
    results_out_dir = Path(results_out_dir) if results_out_dir else (SAMPLES_DIR)
    results_out_dir.mkdir(parents=True, exist_ok=True)

    if inference_module is not None and hasattr(inference_module, 'main'):
        # prefer to delegate if a higher-level runner exists; but don't call CLI main directly.
        pass

    img_files = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
    per_image = []
    dices = []
    ious = []
    for p in img_files:
        out = predict_image_path(p, out_dir=results_out_dir, stem_override=p.stem)
        pred_mask = out.get('pred_mask')
        d = None;
        j = None;
        gt_sum = None
        if mask_dir:
            # try png mask
            mask_path = None
            for ext in ('.png', '.PNG'):
                c = Path(mask_dir) / (p.stem + ext)
                if c.exists():
                    mask_path = c;
                    break
            if mask_path:
                if inference_module is not None and hasattr(inference_module, 'read_mask_png'):
                    mask_true = inference_module.read_mask_png(mask_path, img_size=IMG_SIZE)
                else:
                    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask_img = cv2.resize(mask_img, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)
                    mask_true = (mask_img > 127).astype('uint8')
                gt_sum = int(mask_true.sum())
                if skip_empty_gt and gt_sum == 0:
                    per_image.append(
                        {'image': p.name, 'dice': None, 'iou': None, 'gt_sum': 0, 'pred_sum': int(pred_mask.sum())})
                    continue
                inter = float((mask_true.ravel() * pred_mask.ravel()).sum())
                denom = float(mask_true.sum() + pred_mask.sum())
                d = float((2.0 * inter + 1e-6) / (denom + 1e-6))
                union = float(mask_true.sum() + pred_mask.sum() - inter)
                j = float((inter + 1e-6) / (union + 1e-6)) if union != 0 else 1.0
                dices.append(d)
                ious.append(j)
        per_image.append({'image': p.name, 'dice': d, 'iou': j, 'gt_sum': gt_sum, 'pred_sum': int(pred_mask.sum())})
    import numpy as _np
    return {'per_image': per_image, 'mean_dice': float(_np.mean(dices)) if dices else None,
            'mean_iou': float(_np.mean(ious)) if ious else None}
