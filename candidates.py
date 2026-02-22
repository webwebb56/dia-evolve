"""Candidate filtering via predicate pushdown.

For each isolation window in the DIA run, filter the prior library
to find peptides whose precursor m/z falls within the window.
Optionally filter by predicted RT if a lens is available.
"""

from __future__ import annotations

import polars as pl

from .config import DiaConfig


def get_candidates_for_window(
    prior_df: pl.DataFrame,
    window_center_mz: float,
    window_width_da: float,
    config: DiaConfig,
    rt_center: float | None = None,
) -> pl.DataFrame:
    """Filter prior to candidates matching an isolation window.

    Args:
        prior_df: Full prior library.
        window_center_mz: Center m/z of the isolation window.
        window_width_da: Width of the isolation window in Da.
        config: Pipeline configuration.
        rt_center: Optional RT center for additional filtering.

    Returns:
        Filtered DataFrame of candidate peptides.
    """
    half_width = window_width_da / 2.0 + config.precursor_mz_tolerance_da
    mz_lo = window_center_mz - half_width
    mz_hi = window_center_mz + half_width

    candidates = prior_df.filter(
        (pl.col("precursor_mz") >= mz_lo) & (pl.col("precursor_mz") <= mz_hi)
    )

    # Filter by RT if available
    if rt_center is not None and "urt_prior" in candidates.columns:
        rt_half = config.rt_window_minutes
        # For now, no lens — use wide window on urt_prior
        # This is a placeholder; real lens would map urt_prior → observed RT
        candidates = candidates.filter(
            (pl.col("urt_prior") >= 0) & (pl.col("urt_prior") <= 100)
        )

    # Filter fragments: remove peptides with no fragments
    if "frag_mz" in candidates.columns:
        candidates = candidates.filter(
            pl.col("frag_mz").list.len() > 0
        )

    return candidates


def get_fragment_targets(
    candidate_row: dict,
    config: DiaConfig,
) -> list[tuple[float, float]]:
    """Get (mz, predicted_intensity) pairs for a candidate's fragments.

    Filters by minimum intensity and takes top-N.

    Args:
        candidate_row: A single row from the prior DataFrame.
        config: Pipeline configuration.

    Returns:
        List of (fragment_mz, predicted_relative_intensity) tuples,
        sorted by intensity descending.
    """
    frag_mz = candidate_row.get("frag_mz", [])
    frag_int = candidate_row.get("frag_int", [])

    if not frag_mz or not frag_int:
        return []

    # Pair and filter by minimum intensity
    pairs = [
        (mz, intensity)
        for mz, intensity in zip(frag_mz, frag_int)
        if intensity >= config.min_fragment_intensity
    ]

    # Sort by intensity descending, take top-N
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: config.max_fragments_per_peptide]
