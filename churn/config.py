"""Central configuration and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
SHAP_PLOTS_DIR = ARTIFACTS_DIR / "shap"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def setup_logging(level: int = logging.INFO) -> None:
    """Configure unified logging format for the whole pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
