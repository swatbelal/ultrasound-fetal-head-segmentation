import json
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .forms import UploadImageForm
from . import inference_wrapper as wrapper

import config as cfg

RESULTS_DIR = Path(cfg.RESULTS_DIR)
SAMPLES_DIR = RESULTS_DIR / 'sample_predictions'
TRAIN_SAMPLES = SAMPLES_DIR / 'train'
TEST_SAMPLES = SAMPLES_DIR / 'test'

def load_metrics_json():
    m = RESULTS_DIR / 'metrics.json'
    if m.exists():
        try:
            return json.loads(m.read_text())
        except Exception:
            return None
    return None

def index(request):
    metrics = load_metrics_json()
    train_count = len(list((TRAIN_SAMPLES).glob('*_overlay.png'))) if TRAIN_SAMPLES.exists() else 0
    test_count = len(list((TEST_SAMPLES).glob('*_overlay.png'))) if TEST_SAMPLES.exists() else 0
    return render(request, 'inference_app/index.html', {'metrics': metrics, 'train_count': train_count, 'test_count': test_count})

def upload_view(request):
    form = UploadImageForm(request.POST or None, request.FILES or None)
    result = None
    if request.method == 'POST' and form.is_valid():
        f = form.cleaned_data['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        saved_name = fs.save(f.name, f)
        saved_path = Path(settings.MEDIA_ROOT) / saved_name
        out_dir = RESULTS_DIR / 'ad_hoc'
        out_dir.mkdir(parents=True, exist_ok=True)
        out = wrapper.predict_image_path(saved_path, out_dir=out_dir, stem_override=saved_path.stem)
        result = {
            'image': '/media/' + saved_name,
            'overlay': str(out_dir / f"{saved_path.stem}_overlay.png"),
            'pred_mask': str(out_dir / f"{saved_path.stem}_pred_mask.png")
        }
    return render(request, 'inference_app/upload.html', {'form': form, 'result': result})

def gallery(request):
    def list_overlays(folder):
        folder = Path(folder)
        if not folder.exists():
            return []
        imgs = sorted([p.name for p in folder.glob('*_overlay.png')])
        return imgs
    train_imgs = list_overlays(TRAIN_SAMPLES)
    test_imgs = list_overlays(TEST_SAMPLES)
    return render(request, 'inference_app/gallery.html', {'train_imgs': train_imgs, 'test_imgs': test_imgs,
                                                          'train_dir': str(TRAIN_SAMPLES), 'test_dir': str(TEST_SAMPLES)})

def run_eval(request):
    train_img_dir = Path(cfg.TRAIN_IMG_DIR)
    train_mask_dir = Path(cfg.TRAIN_MASK_IMG_DIR)
    test_img_dir = Path(cfg.VAL_IMG_DIR)
    test_mask_dir = Path(cfg.VAL_MASK_IMG_DIR)

    (SAMPLES_DIR / 'train').mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / 'test').mkdir(parents=True, exist_ok=True)

    res_train = {'error': 'train dirs not found'}
    res_test = {'error': 'test dirs not found'}

    if train_img_dir.exists() and train_mask_dir.exists():
        res_train = wrapper.run_on_dir(train_img_dir, mask_dir=train_mask_dir, results_out_dir=SAMPLES_DIR / 'train')
    if test_img_dir.exists():
        res_test = wrapper.run_on_dir(test_img_dir, mask_dir=test_mask_dir, results_out_dir=SAMPLES_DIR / 'test')

    combined = {'train': res_train, 'test': res_test}
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(combined, indent=4))
    return JsonResponse({'status': 'ok', 'train_mean_dice': combined['train'].get('mean_dice'), 'test_mean_dice': combined['test'].get('mean_dice')})
