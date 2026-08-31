from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
SPX_MINUTE_FILE = DATA_DIR / "^SP500.Last.txt"
TRAIN_FEATURES_FILE = DATA_DIR / "train_features_5m_clean.parquet"
PROCESSED_DIR = DATA_DIR / "processed_sets"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def processed_path(*parts: str) -> Path:
    return PROCESSED_DIR.joinpath(*parts)
