#!/usr/bin/env python
import os
import sys
from pathlib import Path

# ensure repo root and src/ are importable
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
