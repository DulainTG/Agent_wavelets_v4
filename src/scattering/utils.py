"""Utility helpers for saving complex data structures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def complex_dict_to_json(source: Dict, precision: int = 6) -> Dict:
    result = {}
    for key, value in source.items():
        if isinstance(value, dict):
            result[str(key)] = complex_dict_to_json(value, precision=precision)
        else:
            result[str(key)] = {
                "real": round(float(np.real(value)), precision),
                "imag": round(float(np.imag(value)), precision),
                "abs": round(float(np.abs(value)), precision),
                "angle": round(float(np.angle(value)), precision),
            }
    return result


def save_numpy_series(path: Path, scales: Iterable[int], values: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([np.asarray(list(scales)), values])
    np.savetxt(path, arr, header=header, fmt="%g")
