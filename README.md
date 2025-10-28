
# Fetal Head Ultrasound — Segmentation + Django Dashboard

A small U-Net-based segmentation project for fetal head ultrasound images, with:
- training scripts (`train.py`)
- evaluation / sample prediction generator (`evaluate.py`)
- ad-hoc inference (`inference.py`)
- a Django dashboard to run inference/evaluation, show logs, cancel runs and browse overlays.

This README covers installation, running, web UI, troubleshooting, and suggested commit messages.

## Quickstart — local

1. Create env and install:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Place dataset in `data/fetal-head-ultrasound/` with `train/img`, `train/masks_png`, `test/img`, `test/masks_png`.

3. Edit `src/config.py` if needed.

4. Train (optional):
   ```
   python src/train.py
   ```

5. Evaluate (generate overlays and metrics):
   ```
   python src/evaluate.py
   ```

6. Run ad-hoc inference:
   ```
   python src/inference.py --model .results/unet_fetal_head.keras --input data/.../img/1_HC.png --outdir .results/inference
   ```

7. Run Django server:
   ```
   python manage.py runserver 0.0.0.0:8585
   ```

## Important notes (troubleshooting)

- **Empty / black test masks**: many test annotations are JSON or intentionally empty. `evaluate.py` treats empty JSONs as empty masks. Use `mean_dice_excluding_empty_gt` to inspect performance on images with real GT masks.
- **Serving images in Django**: do not use absolute Windows paths in `<img src>`. Configure `MEDIA_URL`/`MEDIA_ROOT` and `RESULTS_URL`/`RESULTS_ROOT` and add `django.views.static.serve` url patterns for development.
- **Lock file & cancel**: `evaluate.py` writes `results/eval.lock` with `{"pid":..., "status":"running"}`. The dashboard reads it to show status and can `os.kill(pid, SIGTERM)` to cancel (platform caveats on Windows).
- **Log encoding errors**: read logs with `encoding="utf-8", errors="replace"` to avoid UnicodeDecodeError on Windows.
- **Docker (OpenCV deps)**: add `libgl1 libglib2.0-0` to container image.
