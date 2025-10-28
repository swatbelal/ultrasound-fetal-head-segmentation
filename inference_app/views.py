# inference_app/views.py
import os
import sys
import json
import signal
import psutil
import subprocess
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils.encoding import force_str

from .forms import UploadImageForm
from . import inference_wrapper as wrapper

# import project config from src (you already used this style)
import src.config as cfg

# Paths (physical)
RESULTS_DIR = Path(cfg.RESULTS_DIR)
SAMPLES_DIR = RESULTS_DIR / "sample_predictions"
TRAIN_SAMPLES = SAMPLES_DIR / "train"
TEST_SAMPLES = SAMPLES_DIR / "test"
LOCK_FILE = Path(cfg.EVAL_LOCK_FILE)
LOG_FILE = Path(cfg.LOG_FILE_NAME)

# Public URL prefixes (how your nginx/django static serving exposes results)
# e.g. settings.RESULTS_URL should be "/results" and MEDIA_URL "/media"
RESULTS_URL = getattr(settings, "RESULTS_URL", "/results").rstrip("/")
TRAIN_URL = f"{RESULTS_URL}/sample_predictions/train"
TEST_URL = f"{RESULTS_URL}/sample_predictions/test"


def _read_metrics():
    """Read results/metrics.json if present and return pretty-printed JSON string or None."""
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text())
        # Return formatted JSON string
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return None


def index(request):
    """Dashboard home. Shows summary counts and last metrics."""
    metrics = _read_metrics()
    train_count = len(list(TRAIN_SAMPLES.glob("*_overlay.png"))) if TRAIN_SAMPLES.exists() else 0
    test_count = len(list(TEST_SAMPLES.glob("*_overlay.png"))) if TEST_SAMPLES.exists() else 0

    context = {
        "metrics": metrics,
        "train_count": train_count,
        "test_count": test_count,
        "results_url": RESULTS_URL,
    }
    return render(request, "inference_app/index.html", context)


def upload_view(request):
    """
    Upload a single image via form, run inference and show the overlay/pred.
    Uses wrapper.predict_image_path(...) so this remains decoupled from inference implementation.
    """
    form = UploadImageForm(request.POST or None, request.FILES or None)
    result = None

    if request.method == "POST" and form.is_valid():
        f = form.cleaned_data["image"]
        # save uploaded file to MEDIA_ROOT
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        saved_name = fs.save(f.name, f)
        saved_path = Path(settings.MEDIA_ROOT) / saved_name

        # ensure ad_hoc output dir exists inside results
        out_dir = RESULTS_DIR / "ad_hoc"
        out_dir.mkdir(parents=True, exist_ok=True)

        # call your wrapper to predict and save outputs
        wrapper.predict_image_path(saved_path, out_dir=out_dir, stem_override=saved_path.stem)

        # build public URLs (served under RESULTS_URL)
        stem = saved_path.stem
        result = {
            "image": f"{settings.MEDIA_URL.rstrip('/')}/{saved_name}",
            "overlay": f"{RESULTS_URL}/ad_hoc/{stem}_overlay.png",
            "pred_mask": f"{RESULTS_URL}/ad_hoc/{stem}_pred_mask.png",
        }

    return render(request, "inference_app/upload.html", {"form": form, "result": result})


def gallery(request):
    """
    Show gallery page listing overlay images in train/test sample folders.
    Templates should use `train_url`/`test_url` + filename to build <img> src.
    """

    def list_overlays(folder: Path):
        if not folder.exists():
            return []
        # return sorted basenames (e.g. 102_2HC_overlay.png)
        return sorted([p.name for p in folder.glob("*_overlay.png")])

    train_imgs = list_overlays(TRAIN_SAMPLES)
    test_imgs = list_overlays(TEST_SAMPLES)

    context = {
        "train_imgs": train_imgs,
        "test_imgs": test_imgs,
        # public URL roots for building <img src="...">
        "train_url": TRAIN_URL,
        "test_url": TEST_URL,
        "results_url": RESULTS_URL,
        # also include absolute FS paths if template wants to show debugging info
        "train_dir_fs": str(TRAIN_SAMPLES),
        "test_dir_fs": str(TEST_SAMPLES),
    }
    return render(request, "inference_app/gallery.html", context)


def upload_gallery(request):
    media_dir = Path(settings.MEDIA_ROOT)
    results_dir = Path(cfg.RESULTS_DIR) / "ad_hoc"

    groups = []

    if media_dir.exists():
        # sort uploads by newest first
        files = sorted(media_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)

        for f in files:
            stem = f.stem  # e.g. "90_HC"
            group = {
                "basename": f.name,
                "image": f"{settings.MEDIA_URL}{f.name}",
                "overlay": None,
                "pred_mask": None,
            }

            # check related results (ad_hoc outputs)
            overlay = results_dir / f"{stem}_overlay.png"
            pred_mask = results_dir / f"{stem}_pred_mask.png"

            if overlay.exists():
                group["overlay"] = f"{settings.RESULTS_URL}ad_hoc/{overlay.name}"
            if pred_mask.exists():
                group["pred_mask"] = f"{settings.RESULTS_URL}ad_hoc/{pred_mask.name}"

            groups.append(group)

    context = {
        "groups": groups,
    }
    return render(request, "inference_app/upload_gallery.html", context)


# --------------------
# SSE streaming runner
# --------------------
EVAL_SCRIPT = Path(settings.BASE_DIR) / "src" / "evaluate.py"


def _sse_from_process(proc):
    """
    Generator that yields SSE 'data:' events from process stdout lines.
    When finished, yields a final 'metrics' custom event with metrics.json content (if present).
    """
    try:
        # stream lines from process stdout
        for raw in proc.stdout:
            if raw is None:
                continue
            line = force_str(raw).rstrip("\n")
            # yield as SSE data event
            yield f"data: {line}\n\n"

        # process ended
        proc.wait()
        yield f"data: [process exited with code {proc.returncode}]\n\n"

        # append final metrics.json as an event if available
        metrics_path = RESULTS_DIR / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                # send as a named SSE event 'metrics' with JSON payload
                yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
            except Exception as e:
                yield f"data: [failed to read metrics.json: {e}]\n\n"
        else:
            yield f"data: [no metrics.json produced]\n\n"

    except GeneratorExit:
        # client disconnected; ensure child proc terminated
        try:
            proc.terminate()
        except Exception:
            pass
        raise
    except Exception as e:
        yield f"data: [streaming error: {e}]\n\n"


def is_process_alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def status_eval(request):
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            pid = data.get("pid")
            status = data.get("status", "unknown")

            # Optionally: verify if the process is still alive
            if pid:
                alive = is_process_alive(pid)
                if not alive:
                    status = "cancelled"
                    data["status"] = status
                    data["pid"] = None
                    LOCK_FILE.write_text(json.dumps(data))
            else:
                alive = False

            return JsonResponse({
                "running": alive,
                "status": status,
                "pid": pid
            })
        except Exception as e:
            return JsonResponse({"error": f"Failed to parse lock file: {str(e)}"})
    else:
        return JsonResponse({"running": False, "status": "none", "pid": None})


def view_log(request):
    if not LOG_FILE.exists():
        return HttpResponse("No log yet.", content_type="text/plain")
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        text = f"⚠️ Could not read log file: {e}"
    return HttpResponse(text, content_type="text/plain; charset=utf-8")


def cancel_eval(request):
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
        except BaseException as e:
            return JsonResponse({"error": f"Lock file unreadable: {str(e)}"})

        pid = data.get("pid")
        if not pid:
            # Lock exists but PID not yet set
            return JsonResponse({"error": "Evaluation is starting, PID not available yet"})

        try:
            os.kill(pid, signal.SIGTERM)  # graceful
            data["status"] = "cancelled"
            data["pid"] = None
            LOCK_FILE.write_text(json.dumps(data))
            return JsonResponse({"status": "cancelled"})
        except Exception as e:
            return JsonResponse({"error": str(e)})

    return JsonResponse({"error": "No running process"})


@csrf_exempt
@require_GET
def run_eval_stream(request):
    if LOCK_FILE.exists():
        data = json.loads(LOCK_FILE.read_text())
        if data.get("status") == "running":
            return JsonResponse({"error": "Evaluation already running"}, status=400)
        
    def event_stream():
        process = subprocess.Popen(
            [sys.executable, "src/evaluate.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"
        ret = process.wait()
        if ret == 0:
            yield "event: done\ndata: finished\n\n"
        else:
            yield f"event: error\ndata: process exited with code {ret}\n\n"

    return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
