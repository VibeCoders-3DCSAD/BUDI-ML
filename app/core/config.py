from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = APP_DIR
TRAINING_DIR = REPO_ROOT / "training"
TRAINING_SCRIPTS_DIR = TRAINING_DIR / "scripts"
MODELS_DIR = REPO_ROOT / "models"

SERVICE_NAME = "odin-ml"
SERVICE_VERSION = "0.4.0"
TRUSTED_HISTORY_SHARED_SECRET_ENV = "ODIN_TRUSTED_HISTORY_SHARED_SECRET"
MAX_FORECAST_MONTHS = 12


def ensure_training_scripts_on_path() -> None:
    """Make the training feature-engineering modules importable by the service.

    The service reuses the deterministic feature builders from the training
    pipeline (DRY) instead of re-implementing them. Adding the path is
    idempotent and only affects this process.
    """
    scripts = str(TRAINING_SCRIPTS_DIR)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
