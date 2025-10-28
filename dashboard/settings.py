import sys
from pathlib import Path
from src.config import RESULTS_DIR

BASE_DIR = Path(__file__).resolve().parent.parent

# allow importing src by adding it to sys.path
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

SECRET_KEY = "dev-dashboard-secret"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "inference_app",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]

ROOT_URLCONF = "dashboard.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "inference_app" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

WSGI_APPLICATION = "dashboard.wsgi.application"

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "inference_app" / "static"]
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
RESULTS_URL = "/results/"
RESULTS_ROOT = RESULTS_DIR

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
