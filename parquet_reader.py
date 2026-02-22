"""DIA Parquet reader — iterate spectra by isolation window.

Reads PSI-MS v2-int schema Parquet files. The key columns are:
  - scan_idx: UInt32 (frame/scan number)
  - scan_start_time: Float64 (RT in minutes)
  - ms_level: UInt8 (always 2 for DIA MS2)
  - precursor_mz: Float64 (isolation window center)
  - mz_array: List[Int64] (micro-Dalton encoded, x1e6)
  - intensity_array: List[UInt32]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl


def read_dia_parquet(path: str | Path) -> pl.DataFrame:
    """Read a DIA Parquet file into a polars DataFrame.

    Handles both micro-Dalton encoded (Int64) and float-encoded (Float64)
    m/z arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DIA Parquet not found: {path}")

    df = pl.read_parquet(path)

    # Ensure we have MS2 spectra
    if "ms_level" in df.columns:
        df = df.filter(pl.col("ms_level") == 2)

    required = {"scan_start_time", "precursor_mz", "mz_array", "intensity_array"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DIA Parquet missing columns: {missing}")

    return df


def get_isolation_windows(df: pl.DataFrame) -> list[tuple[float, float]]:
    """Extract unique isolation windows from DIA data.

    Returns sorted list of (center_mz, width_estimate) tuples.
    Width is estimated from spacing between consecutive windows.
    """
    centers = sorted(df["precursor_mz"].unique().to_list())
    if len(centers) < 2:
        return [(c, 25.0) for c in centers]  # Default 25 Da width

    # Estimate width from spacing
    spacings = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    median_spacing = sorted(spacings)[len(spacings) // 2]
    return [(c, median_spacing) for c in centers]


def iter_by_window(
    df: pl.DataFrame,
    rt_min: float | None = None,
    rt_max: float | None = None,
) -> Iterator[tuple[float, pl.DataFrame]]:
    """Iterate over DIA data grouped by isolation window.

    Yields (window_center_mz, spectra_df) pairs sorted by RT.
    Optionally filter by RT range.
    """
    if rt_min is not None:
        df = df.filter(pl.col("scan_start_time") >= rt_min)
    if rt_max is not None:
        df = df.filter(pl.col("scan_start_time") <= rt_max)

    windows = sorted(df["precursor_mz"].unique().to_list())
    for center_mz in windows:
        window_df = (
            df.filter(pl.col("precursor_mz") == center_mz)
            .sort("scan_start_time")
        )
        if len(window_df) > 0:
            yield center_mz, window_df


def decode_mz_array(mz_array: list[int] | list[float]) -> list[float]:
    """Decode m/z array from micro-Dalton Int64 to float Da.

    If already float, return as-is. If Int64 (micro-Dalton), divide by 1e6.
    """
    if not mz_array:
        return []
    if isinstance(mz_array[0], int):
        return [v / 1_000_000.0 for v in mz_array]
    return list(mz_array)


def extract_spectrum(
    row: dict,
) -> tuple[list[float], list[float], float]:
    """Extract (mz_values, intensities, rt) from a spectrum row.

    Returns decoded m/z values (in Da), intensity values, and RT.
    """
    mz_raw = row.get("mz_array", [])
    intensities = row.get("intensity_array", [])
    rt = row.get("scan_start_time", 0.0)

    mz_vals = decode_mz_array(mz_raw)
    int_vals = [float(v) for v in intensities] if intensities else []

    return mz_vals, int_vals, rt
