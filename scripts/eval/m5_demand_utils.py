"""Utilities for deriving benchmark demand traces from the real M5 data.

The benchmark environment is single-SKU/single-retailer at the demand edge,
whereas M5 contains 30,490 item-store daily sales series. We therefore map M5
into the environment by selecting one reproducible item-store window and scaling
its observed 30-day unit-sales path to the benchmark's demand level.

This module intentionally avoids a silent synthetic fallback. If the full M5
CSV is unavailable, it may use a previously generated metadata file containing
the derived 30-day real-data trace. A synthetic fallback is only available when
explicitly requested through ``allow_synthetic=True`` or
``GYM_INVMGMT_ALLOW_SYNTHETIC_M5=1``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gym_invmgmt.data_adapters import m5_wide_csv_to_spec

DEFAULT_SALES_PATH = PROJECT_ROOT / "data" / "external" / "m5" / "sales_train_evaluation.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "results" / "m5_window_metadata.json"
M5_METADATA_SCHEMA_VERSION = 1
M5_EXPECTED_ADAPTER = "m5_wide_csv"


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _synthetic_window(num_periods: int, base_mu: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(0)
    series = rng.poisson(base_mu, num_periods).astype(float)
    metadata = {
        "source": "synthetic_poisson_fallback",
        "warning": "Not derived from M5. Use only for local smoke tests.",
        "num_periods": int(num_periods),
        "base_mu": float(base_mu),
        "scaled_series": series.tolist(),
        "scaled_mean": float(np.mean(series)),
        "scaled_std": float(np.std(series)),
        "scaled_cv": float(np.std(series) / max(np.mean(series), 1e-9)),
    }
    return {"volatile": series}, metadata


def _load_cached_metadata(
    metadata_path: Path,
    num_periods: int,
    base_mu: float,
    search_days: int,
    min_window_mean: float,
    min_nonzero_frac: float,
    max_to_mean_cap: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("adapter") != M5_EXPECTED_ADAPTER:
        return None
    if metadata.get("source") == "synthetic_poisson_fallback":
        return None
    if int(metadata.get("num_periods", -1)) != int(num_periods):
        return None
    if abs(float(metadata.get("base_mu", np.nan)) - float(base_mu)) > 1e-9:
        return None
    expected_params = {
        "search_days": search_days,
        "min_window_mean": min_window_mean,
        "min_nonzero_frac": min_nonzero_frac,
        "max_to_mean_cap": max_to_mean_cap,
    }
    for key, expected in expected_params.items():
        if key not in metadata:
            return None
        if abs(float(metadata[key]) - float(expected)) > 1e-9:
            return None
    required_keys = {"series_id", "start_day", "end_day", "raw_series", "scaled_series"}
    if not required_keys.issubset(metadata):
        return None
    if int(metadata.get("metadata_schema_version", M5_METADATA_SCHEMA_VERSION)) != M5_METADATA_SCHEMA_VERSION:
        return None
    series = np.asarray(metadata["scaled_series"], dtype=float)
    if series.shape != (num_periods,):
        return None
    return {"volatile": series}, metadata


def _select_volatile_window(
    sales_path: Path,
    num_periods: int,
    base_mu: float,
    search_days: int = 365,
    min_window_mean: float = 8.0,
    min_nonzero_frac: float = 0.80,
    max_to_mean_cap: float = 4.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    spec = m5_wide_csv_to_spec(
        sales_path,
        num_periods=num_periods,
        base_mu=base_mu,
        search_days=search_days,
        min_window_mean=min_window_mean,
        min_nonzero_frac=min_nonzero_frac,
        max_to_mean_cap=max_to_mean_cap,
    )
    series = np.asarray(spec.demand_config["external_series"], dtype=float)
    return {"volatile": series}, spec.metadata


def get_m5_demand_windows(
    num_periods: int = 30,
    base_mu: float = 20.0,
    sales_path: str | os.PathLike[str] | None = None,
    metadata_path: str | os.PathLike[str] | None = None,
    search_days: int = 365,
    min_window_mean: float = 8.0,
    min_nonzero_frac: float = 0.80,
    max_to_mean_cap: float = 4.0,
    allow_synthetic: bool | None = None,
    write_metadata: bool = True,
) -> dict[str, np.ndarray]:
    """Return benchmark-ready M5 demand windows.

    Returns a dictionary with a ``volatile`` 30-period series scaled to mean
    ``base_mu``. The values are used by :class:`DemandEngine` as period-specific
    demand means for the benchmark's single retail-demand link.
    """
    sales = Path(os.environ.get("M5_SALES_PATH") or sales_path or DEFAULT_SALES_PATH)
    metadata_file = Path(metadata_path or DEFAULT_METADATA_PATH)

    if sales.exists():
        windows, metadata = _select_volatile_window(
            sales,
            num_periods,
            base_mu,
            search_days=search_days,
            min_window_mean=min_window_mean,
            min_nonzero_frac=min_nonzero_frac,
            max_to_mean_cap=max_to_mean_cap,
        )
        metadata = dict(metadata)
        metadata["metadata_schema_version"] = M5_METADATA_SCHEMA_VERSION
        if write_metadata:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump({k: _jsonify(v) for k, v in metadata.items()}, f, indent=2)
        return windows

    cached = _load_cached_metadata(
        metadata_file,
        num_periods,
        base_mu,
        search_days,
        min_window_mean,
        min_nonzero_frac,
        max_to_mean_cap,
    )
    if cached is not None:
        windows, _ = cached
        return windows

    if allow_synthetic is None:
        allow_synthetic = os.environ.get("GYM_INVMGMT_ALLOW_SYNTHETIC_M5") == "1"
    if allow_synthetic:
        windows, metadata = _synthetic_window(num_periods, base_mu)
        if write_metadata:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump({k: _jsonify(v) for k, v in metadata.items()}, f, indent=2)
        return windows

    raise FileNotFoundError(
        "Real M5 demand data is required for D_WalmartM5 scenarios. "
        f"Expected {sales}. Set M5_SALES_PATH to sales_train_evaluation.csv, "
        f"or provide a derived metadata file at {metadata_file}. "
        "For smoke tests only, set GYM_INVMGMT_ALLOW_SYNTHETIC_M5=1."
    )


if __name__ == "__main__":
    windows = get_m5_demand_windows()
    series = windows["volatile"]
    print("M5 volatile window:")
    print(np.array2string(series, precision=3, separator=", "))
    with DEFAULT_METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(json.dumps({k: metadata[k] for k in [
        "series_id", "start_day", "end_day", "raw_mean", "raw_cv",
        "scaled_mean", "scaled_cv"
    ] if k in metadata}, indent=2))
