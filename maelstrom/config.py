"""Load configs/data.yaml with configs/local.yaml overrides."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    with open(CONFIG_DIR / "data.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    local = CONFIG_DIR / "local.yaml"
    if local.exists():
        with open(local) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if path is not None:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return cfg
