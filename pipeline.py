"""Pipeline orchestrator — runs the full DIA matching pipeline.

Calls modules in sequence:
1. Load prior + DIA data
2. Generate decoys
3. For each isolation window:
   a. Get candidates (predicate pushdown)
   b. Extract XICs for each candidate's fragments
   c. Detect peaks
   d. Compute RSM features
   e. Score
4. Compute q-values (FDR control)
5. Filter at FDR threshold
6. Compute loss metrics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .candidates import get_candidates_for_window, get_fragment_targets
from .config import DiaConfig
from .decoys import add_decoy_column, generate_decoys
from .fdr import compute_peptide_level_fdr, compute_qvalues, fdr_summary, filter_fdr
from .features import compute_features
from .loss import LossMetrics, compute_loss
from .parquet_reader import get_isolation_windows, iter_by_window, read_dia_parquet
from .peak import detect_peaks_multi
from .prior import load_prior_parquet
from .scoring import compute_score
from .xic import PeptideXics, extract_xics_vectorized

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from a pipeline run."""

    results_df: pl.DataFrame           # All scored PSMs (targets + decoys)
    filtered_df: pl.DataFrame          # PSMs passing FDR threshold
    loss: LossMetrics                  # 3-species evaluation metrics
    fdr_stats: dict                    # FDR summary statistics
    config: DiaConfig                  # Configuration used
    elapsed_seconds: float             # Wall-clock time
    n_windows_processed: int           # Number of isolation windows processed
    n_candidates_scored: int           # Total candidates scored


def run_pipeline(
    dia_path: str | Path,
    prior_path: str | Path,
    config: DiaConfig | None = None,
    species_ratios: dict | None = None,
    max_windows: int | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the full DIA matching pipeline.

    Args:
        dia_path: Path to DIA Parquet file.
        prior_path: Path to prior Parquet file.
        config: Pipeline configuration (uses defaults if None).
        species_ratios: Expected species mixing ratios for loss computation.
        max_windows: Limit number of windows to process (for testing).
        verbose: Print progress messages.

    Returns:
        PipelineResult with all outputs.
    """
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

    # Combine target + decoy libraries
    combined_prior = pl.concat([target_df, decoy_df], how="diagonal_relaxed")
    if verbose:
        logger.info("  %d targets + %d decoys = %d total",
                     len(target_df), len(decoy_df), len(combined_prior))

    # --- Step 3: Process each isolation window ---
    windows = get_isolation_windows(dia_df)
    if max_windows is not None:
        windows = windows[:max_windows]

    all_results = []
    n_windows = 0
    n_candidates_total = 0

    for window_center, window_width in windows:
        n_windows += 1
        if verbose and n_windows % 10 == 0:
            logger.info("  Processing window %d/%d (m/z=%.1f)",
                         n_windows, len(windows), window_center)

        # Get spectra for this window
        window_spectra = dia_df.filter(
            pl.col("precursor_mz") == window_center
        ).sort("scan_start_time")

        if len(window_spectra) == 0:
            continue

        # Get candidate peptides
        candidates = get_candidates_for_window(
            combined_prior, window_center, window_width, config
        )

        if len(candidates) == 0:
            continue

        # Process each candidate
        window_results = _process_candidates(
            candidates, window_spectra, window_center, config
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
            loss=LossMetrics(1.0, float("inf"), 0, 0.0, 0, 0),
            fdr_stats={"total_targets": 0, "total_decoys": 0},
            config=config,
            elapsed_seconds=time.time() - t0,
            n_windows_processed=n_windows,
            n_candidates_scored=0,
        )

    results_df = pl.DataFrame(all_results)

    # Pre-FDR score filter
    results_df = results_df.filter(pl.col("score") >= config.score_threshold)

    # --- Step 5: FDR control ---
    if verbose:
        logger.info("Computing q-values (%d PSMs)", len(results_df))
    results_df = compute_qvalues(results_df)

    # Peptide-level FDR
    peptide_results = compute_peptide_level_fdr(results_df)

    # Filter at FDR threshold
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


def _process_candidates(
    candidates: pl.DataFrame,
    window_spectra: pl.DataFrame,
    window_center: float,
    config: DiaConfig,
) -> list[dict]:
    """Process all candidates for one isolation window.

    Args:
        candidates: Candidate peptides for this window.
        window_spectra: DIA spectra for this window.
        window_center: Center m/z of the isolation window.
        config: Pipeline configuration.

    Returns:
        List of result dicts (one per candidate with valid peaks).
    """
    results = []

    for row in candidates.iter_rows(named=True):
        # Get fragment targets
        frag_targets = get_fragment_targets(row, config)
        if not frag_targets:
            continue

        # Get predicted RT if available
        predicted_rt = row.get("urt_prior", None)
        rt_sigma = row.get("urt_sigma", 5.0) or 5.0

        # Extract XICs
        traces = extract_xics_vectorized(
            window_spectra, frag_targets, config,
            rt_center=None,  # No lens yet — use full window
        )

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

        # Score
        score = compute_score(features, config)

        # Find consensus apex RT
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
