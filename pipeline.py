"""Pipeline orchestrator — runs the full DIA matching pipeline.

Optimized for real-scale data (232K spectra, 3M+ peptides).

Key optimizations over naive approach:
1. Pre-group spectra by isolation window (once, upfront)
2. Sort prior by precursor_mz for O(log N) candidate lookup
3. Cap candidates per window to avoid combinatorial explosion
4. Pre-decode m/z arrays to numpy for vectorized XIC extraction
"""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .candidates import get_fragment_targets
from .config import DiaConfig
from .decoys import add_decoy_column, generate_decoys
from .fdr import compute_peptide_level_fdr, compute_qvalues, fdr_summary, filter_fdr
from .features import compute_features
from .loss import LossMetrics, compute_loss
from .parquet_reader import decode_mz_array, read_dia_parquet
from .peak import detect_peaks_multi
from .prior import load_prior_parquet
from .scoring import compute_score
from .xic import XicTrace

logger = logging.getLogger(__name__)

# Maximum candidates per window to avoid combinatorial explosion
MAX_CANDIDATES_PER_WINDOW = 300


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""

    results_df: pl.DataFrame
    filtered_df: pl.DataFrame
    loss: LossMetrics
    fdr_stats: dict
    config: DiaConfig
    elapsed_seconds: float
    n_windows_processed: int
    n_candidates_scored: int


def run_pipeline(
    dia_path: str | Path,
    prior_path: str | Path,
    config: DiaConfig | None = None,
    species_ratios: dict | None = None,
    max_windows: int | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full DIA matching pipeline on real data."""
    if config is None:
        config = DiaConfig()
    if species_ratios is None:
        species_ratios = {0: {"human": 0.65, "yeast": 0.2, "ecoli": 0.15}}

    t0 = time.time()

    # --- Step 1: Load data ---
    if verbose:
        logger.info("Loading DIA data from %s", dia_path)
    dia_df = read_dia_parquet(dia_path)
    if verbose:
        logger.info("  %d spectra loaded", len(dia_df))

    if verbose:
        logger.info("Loading prior from %s", prior_path)
    prior_df = load_prior_parquet(prior_path)
    if verbose:
        logger.info("  %d peptides in prior", len(prior_df))

    # --- Step 2: Generate decoys ---
    if verbose:
        logger.info("Generating decoys (method=%s)", config.decoy_method)
    target_df = add_decoy_column(prior_df)
    decoy_df = generate_decoys(prior_df, config)
    combined_prior = pl.concat([target_df, decoy_df], how="diagonal_relaxed")
    if verbose:
        logger.info("  %d targets + %d decoys = %d total",
                     len(target_df), len(decoy_df), len(combined_prior))

    # --- Step 2.5: Pre-index for performance ---
    if verbose:
        logger.info("Pre-indexing prior by precursor_mz...")
    t_idx = time.time()

    # Sort prior by precursor_mz for binary search
    combined_prior = combined_prior.sort("precursor_mz")
    prior_mz_array = combined_prior["precursor_mz"].to_numpy()

    # Pre-group spectra by isolation window
    if verbose:
        logger.info("Pre-grouping spectra by isolation window...")
    window_groups = {}
    unique_windows = sorted(dia_df["precursor_mz"].unique().to_list())
    for w in unique_windows:
        wdf = dia_df.filter(pl.col("precursor_mz") == w).sort("scan_start_time")
        if len(wdf) > 0:
            window_groups[w] = wdf

    # Estimate window width from spacing
    if len(unique_windows) >= 2:
        spacings = [unique_windows[i + 1] - unique_windows[i]
                     for i in range(len(unique_windows) - 1)]
        window_width = float(sorted(spacings)[len(spacings) // 2])
    else:
        window_width = 25.0

    if verbose:
        logger.info("  %d windows, width=%.1f Da, indexed in %.1fs",
                     len(window_groups), window_width, time.time() - t_idx)

    # --- Step 3: Process each isolation window ---
    windows = list(window_groups.keys())
    if max_windows is not None:
        windows = windows[:max_windows]

    all_results = []
    n_windows = 0
    n_candidates_total = 0

    for window_center in windows:
        n_windows += 1
        if verbose and n_windows % 10 == 0:
            n_scored_so_far = sum(1 for _ in all_results)
            logger.info("  Window %d/%d (m/z=%.1f) | %d PSMs so far",
                         n_windows, len(windows), window_center, n_scored_so_far)

        window_spectra = window_groups[window_center]

        # Binary search for candidates in m/z range
        half_width = window_width / 2.0 + config.precursor_mz_tolerance_da
        mz_lo = window_center - half_width
        mz_hi = window_center + half_width
        idx_lo = bisect.bisect_left(prior_mz_array, mz_lo)
        idx_hi = bisect.bisect_right(prior_mz_array, mz_hi)

        if idx_lo >= idx_hi:
            continue

        candidates = combined_prior[idx_lo:idx_hi]

        # Filter to candidates with fragments
        if "frag_mz" in candidates.columns:
            candidates = candidates.filter(pl.col("frag_mz").list.len() > 0)

        if len(candidates) == 0:
            continue

        # Cap candidates (take highest precursor_mz priority = random subset)
        if len(candidates) > MAX_CANDIDATES_PER_WINDOW:
            candidates = candidates.sample(n=MAX_CANDIDATES_PER_WINDOW, seed=42)

        # Pre-decode spectra for this window (do once, reuse for all candidates)
        decoded_spectra = _decode_window_spectra(window_spectra)

        # Process candidates
        window_results = _process_candidates_fast(
            candidates, decoded_spectra, window_center, config
        )
        n_candidates_total += len(window_results)
        all_results.extend(window_results)

    # --- Step 4: Assemble results ---
    if not all_results:
        empty_df = pl.DataFrame({
            "sequence": [], "charge": [], "precursor_mz": [],
            "score": [], "is_decoy": [], "q_value": [],
            "window_mz": [], "apex_rt": [],
            "library_cosine": [], "co_elution": [],
            "delta_rt_sigma": [], "fragment_count": [],
            "log_total_intensity": [], "asymmetry": [], "jaggedness": [],
        })
        return PipelineResult(
            results_df=empty_df,
            filtered_df=empty_df,
            loss=LossMetrics(1.0, 10.0, 0, 0.0, 0, 0),
            fdr_stats={"total_targets": 0, "total_decoys": 0},
            config=config,
            elapsed_seconds=time.time() - t0,
            n_windows_processed=n_windows,
            n_candidates_scored=0,
        )

    results_df = pl.DataFrame(all_results)
    results_df = results_df.filter(pl.col("score") >= config.score_threshold)

    # --- Step 5: FDR control ---
    if verbose:
        logger.info("Computing q-values (%d PSMs)", len(results_df))
    results_df = compute_qvalues(results_df)
    peptide_results = compute_peptide_level_fdr(results_df)
    filtered_df = filter_fdr(peptide_results, config)

    # --- Step 6: Compute loss ---
    loss = compute_loss(
        filtered_df, prior_df, species_ratios,
        total_windows=len(windows),
    )

    fdr_stats = fdr_summary(peptide_results)
    elapsed = time.time() - t0

    if verbose:
        logger.info("Pipeline complete in %.1fs", elapsed)
        logger.info("  %s", loss.summary())
        logger.info("  FDR stats: %s", fdr_stats)

    return PipelineResult(
        results_df=peptide_results,
        filtered_df=filtered_df,
        loss=loss,
        fdr_stats=fdr_stats,
        config=config,
        elapsed_seconds=elapsed,
        n_windows_processed=n_windows,
        n_candidates_scored=n_candidates_total,
    )


@dataclass
class DecodedSpectra:
    """Pre-decoded spectra for fast XIC extraction."""
    rts: np.ndarray                  # (n_spectra,) RT values
    mz_arrays: list[np.ndarray]      # Per-spectrum decoded m/z (Da)
    int_arrays: list[np.ndarray]     # Per-spectrum intensity


def _decode_window_spectra(window_spectra: pl.DataFrame) -> DecodedSpectra:
    """Pre-decode all spectra in a window for fast XIC extraction."""
    rts = window_spectra["scan_start_time"].to_numpy()
    mz_raw_list = window_spectra["mz_array"].to_list()
    int_raw_list = window_spectra["intensity_array"].to_list()

    mz_arrays = []
    int_arrays = []
    for mz_raw, int_raw in zip(mz_raw_list, int_raw_list):
        if mz_raw and int_raw:
            mz_decoded = np.array(decode_mz_array(mz_raw))
            int_decoded = np.array(int_raw, dtype=np.float64)
            mz_arrays.append(mz_decoded)
            int_arrays.append(int_decoded)
        else:
            mz_arrays.append(np.array([], dtype=np.float64))
            int_arrays.append(np.array([], dtype=np.float64))

    return DecodedSpectra(rts=rts, mz_arrays=mz_arrays, int_arrays=int_arrays)


def _extract_xic_fast(
    decoded: DecodedSpectra,
    frag_mz: float,
    tol_da: float,
) -> np.ndarray:
    """Extract XIC for one fragment m/z from pre-decoded spectra.

    Uses numpy searchsorted for O(log N) lookup per spectrum.
    """
    mz_lo = frag_mz - tol_da
    mz_hi = frag_mz + tol_da
    n = len(decoded.rts)
    intensities = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mz_arr = decoded.mz_arrays[i]
        if len(mz_arr) == 0:
            continue
        lo = np.searchsorted(mz_arr, mz_lo, side="left")
        hi = np.searchsorted(mz_arr, mz_hi, side="right")
        if hi > lo:
            intensities[i] = np.sum(decoded.int_arrays[i][lo:hi])

    return intensities


def _process_candidates_fast(
    candidates: pl.DataFrame,
    decoded: DecodedSpectra,
    window_center: float,
    config: DiaConfig,
) -> list[dict]:
    """Process all candidates using pre-decoded spectra."""
    results = []

    for row in candidates.iter_rows(named=True):
        frag_targets = get_fragment_targets(row, config)
        if not frag_targets:
            continue

        predicted_rt = row.get("urt_prior", None)
        rt_sigma = row.get("urt_sigma", 5.0) or 5.0

        # Extract XICs using fast path
        traces = []
        for frag_mz, frag_pred_int in frag_targets:
            tol_da = frag_mz * config.mz_tolerance_ppm * 1e-6
            intensities = _extract_xic_fast(decoded, frag_mz, tol_da)
            traces.append(XicTrace(
                fragment_mz=frag_mz,
                predicted_intensity=frag_pred_int,
                rt=decoded.rts.copy(),
                intensity=intensities,
            ))

        if not traces:
            continue

        # Detect peaks
        peak_results = detect_peaks_multi(traces, config)

        # Compute features
        features = compute_features(
            peak_results,
            predicted_rt=predicted_rt,
            rt_sigma=rt_sigma,
            config=config,
        )

        if features.fragment_count == 0:
            continue

        score = compute_score(features, config)

        detected_peaks = [pr.peak for pr in peak_results if pr.peak is not None]
        apex_rt = float(np.median([p.apex_rt for p in detected_peaks]))

        results.append({
            "sequence": row["sequence"],
            "charge": row["charge"],
            "precursor_mz": row["precursor_mz"],
            "score": score,
            "is_decoy": row.get("is_decoy", False),
            "window_mz": window_center,
            "apex_rt": apex_rt,
            **features.to_dict(),
        })

    return results
