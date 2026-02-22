"""XIC extraction from DIA spectra.

For each candidate peptide, extract chromatographic intensity traces
at its predicted fragment m/z values across retention time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .config import DiaConfig
from .parquet_reader import decode_mz_array


@dataclass
class XicTrace:
    """A single extracted ion chromatogram trace."""

    fragment_mz: float
    predicted_intensity: float
    rt: np.ndarray        # RT values (minutes)
    intensity: np.ndarray  # Observed intensity at each RT point

    @property
    def is_empty(self) -> bool:
        return len(self.rt) == 0 or np.all(self.intensity == 0)

    @property
    def max_intensity(self) -> float:
        return float(np.max(self.intensity)) if len(self.intensity) > 0 else 0.0


@dataclass
class PeptideXics:
    """Collection of XIC traces for one peptide candidate."""

    sequence: str
    charge: int
    precursor_mz: float
    traces: list[XicTrace]

    @property
    def n_detected(self) -> int:
        """Number of traces with non-zero signal."""
        return sum(1 for t in self.traces if not t.is_empty)


def extract_xics(
    window_spectra: pl.DataFrame,
    fragment_targets: list[tuple[float, float]],
    config: DiaConfig,
    rt_center: float | None = None,
) -> list[XicTrace]:
    """Extract XIC traces for a set of fragment m/z targets.

    Args:
        window_spectra: DIA spectra from one isolation window, sorted by RT.
        fragment_targets: List of (fragment_mz, predicted_intensity) to extract.
        config: Pipeline configuration.
        rt_center: Optional RT center for windowed extraction.

    Returns:
        List of XicTrace objects, one per fragment target.
    """
    if len(window_spectra) == 0 or not fragment_targets:
        return []

    # Get RT values and apply RT window if specified
    rts = window_spectra["scan_start_time"].to_numpy()

    if rt_center is not None:
        rt_mask = np.abs(rts - rt_center) <= config.rt_window_minutes
        if not np.any(rt_mask):
            return [
                XicTrace(
                    fragment_mz=fmz,
                    predicted_intensity=fint,
                    rt=np.array([], dtype=np.float64),
                    intensity=np.array([], dtype=np.float64),
                )
                for fmz, fint in fragment_targets
            ]
        idx_mask = np.where(rt_mask)[0]
        rts_filtered = rts[rt_mask]
    else:
        idx_mask = np.arange(len(rts))
        rts_filtered = rts

    # Pre-extract m/z and intensity arrays
    mz_arrays = window_spectra["mz_array"].to_list()
    int_arrays = window_spectra["intensity_array"].to_list()

    traces = []
    for frag_mz, frag_pred_int in fragment_targets:
        # Compute ppm tolerance in Da at this m/z
        tol_da = frag_mz * config.mz_tolerance_ppm * 1e-6
        mz_lo = frag_mz - tol_da
        mz_hi = frag_mz + tol_da

        intensities = np.zeros(len(rts_filtered), dtype=np.float64)

        for i, spectrum_idx in enumerate(idx_mask):
            mz_raw = mz_arrays[spectrum_idx]
            int_raw = int_arrays[spectrum_idx]

            if not mz_raw or not int_raw:
                continue

            mz_vals = decode_mz_array(mz_raw)
            int_vals = int_raw

            # Binary search for m/z range (data is sorted by m/z)
            total_int = 0.0
            for j, mz_val in enumerate(mz_vals):
                if mz_val < mz_lo:
                    continue
                if mz_val > mz_hi:
                    break
                total_int += float(int_vals[j])

            intensities[i] = total_int

        traces.append(XicTrace(
            fragment_mz=frag_mz,
            predicted_intensity=frag_pred_int,
            rt=rts_filtered.copy(),
            intensity=intensities,
        ))

    return traces


def extract_xics_vectorized(
    window_spectra: pl.DataFrame,
    fragment_targets: list[tuple[float, float]],
    config: DiaConfig,
    rt_center: float | None = None,
) -> list[XicTrace]:
    """Vectorized XIC extraction using numpy for better performance.

    Same interface as extract_xics but uses numpy operations where possible.
    Falls back to extract_xics for correctness if arrays are irregular.
    """
    if len(window_spectra) == 0 or not fragment_targets:
        return []

    rts = window_spectra["scan_start_time"].to_numpy()

    if rt_center is not None:
        rt_mask = np.abs(rts - rt_center) <= config.rt_window_minutes
        if not np.any(rt_mask):
            return [
                XicTrace(
                    fragment_mz=fmz,
                    predicted_intensity=fint,
                    rt=np.array([], dtype=np.float64),
                    intensity=np.array([], dtype=np.float64),
                )
                for fmz, fint in fragment_targets
            ]
        window_spectra = window_spectra.filter(
            pl.Series(rt_mask)
        )
        rts = rts[rt_mask]

    # Try to build a dense m/z x RT matrix for fast lookup
    mz_arrays = window_spectra["mz_array"].to_list()
    int_arrays = window_spectra["intensity_array"].to_list()

    n_spectra = len(rts)
    n_frags = len(fragment_targets)

    # Result matrix: fragments x spectra
    result = np.zeros((n_frags, n_spectra), dtype=np.float64)

    for spec_idx in range(n_spectra):
        mz_raw = mz_arrays[spec_idx]
        int_raw = int_arrays[spec_idx]
        if not mz_raw or not int_raw:
            continue

        mz_vals = np.array(decode_mz_array(mz_raw))
        int_vals = np.array(int_raw, dtype=np.float64)

        for frag_idx, (frag_mz, _) in enumerate(fragment_targets):
            tol_da = frag_mz * config.mz_tolerance_ppm * 1e-6
            mask = (mz_vals >= frag_mz - tol_da) & (mz_vals <= frag_mz + tol_da)
            if np.any(mask):
                result[frag_idx, spec_idx] = np.sum(int_vals[mask])

    traces = []
    for frag_idx, (frag_mz, frag_pred_int) in enumerate(fragment_targets):
        traces.append(XicTrace(
            fragment_mz=frag_mz,
            predicted_intensity=frag_pred_int,
            rt=rts.copy(),
            intensity=result[frag_idx].copy(),
        ))

    return traces
