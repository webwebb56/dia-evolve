"""FDR control — target-decoy analysis and q-value estimation.

Implements the standard target-decoy FDR procedure:
1. Score all targets and decoys
2. Sort by score descending
3. Compute q-values: q(i) = min_{j>=i} (2 * decoys_above / total_above)
4. Filter at configured FDR threshold

Per fdr-control.md specification.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import DiaConfig


def compute_qvalues(results_df: pl.DataFrame) -> pl.DataFrame:
    """Compute q-values from target-decoy scores.

    Args:
        results_df: DataFrame with columns:
            - score: Float64 (discriminant score)
            - is_decoy: Boolean

    Returns:
        Input DataFrame with added 'q_value' column.
    """
    if len(results_df) == 0:
        return results_df.with_columns(pl.lit(1.0).alias("q_value"))

    # Sort by score descending
    sorted_df = results_df.sort("score", descending=True)

    scores = sorted_df["score"].to_numpy()
    is_decoy = sorted_df["is_decoy"].to_numpy().astype(bool)

    n = len(scores)
    q_values = np.ones(n, dtype=np.float64)

    # Cumulative counts
    cum_decoys = np.cumsum(is_decoy).astype(np.float64)
    cum_total = np.arange(1, n + 1, dtype=np.float64)

    # FDR at each position: 2 * decoys / total
    # Factor of 2 because decoys estimate false targets
    fdr = 2.0 * cum_decoys / cum_total

    # q-value: monotonized FDR (minimum FDR at this score or lower)
    # Walk from bottom to top, keeping running minimum
    q_values[n - 1] = min(fdr[n - 1], 1.0)
    for i in range(n - 2, -1, -1):
        q_values[i] = min(fdr[i], q_values[i + 1])
        q_values[i] = min(q_values[i], 1.0)

    sorted_df = sorted_df.with_columns(
        pl.Series("q_value", q_values)
    )

    return sorted_df


def filter_fdr(results_df: pl.DataFrame, config: DiaConfig) -> pl.DataFrame:
    """Filter results at configured FDR threshold.

    Args:
        results_df: DataFrame with 'q_value' column.
        config: Pipeline configuration.

    Returns:
        Filtered DataFrame with only target PSMs below FDR threshold.
    """
    filtered = results_df.filter(
        (pl.col("q_value") <= config.fdr_threshold)
        & (~pl.col("is_decoy"))
    )
    return filtered


def compute_peptide_level_fdr(results_df: pl.DataFrame) -> pl.DataFrame:
    """Compute peptide-level FDR by keeping best score per peptide.

    Groups by (sequence, charge), keeps best score, recomputes q-values.

    Args:
        results_df: PSM-level results with scores.

    Returns:
        Peptide-level DataFrame with q-values.
    """
    # Keep best score per peptide
    peptide_df = (
        results_df
        .sort("score", descending=True)
        .unique(subset=["sequence", "charge"], keep="first")
    )

    # Recompute q-values at peptide level
    return compute_qvalues(peptide_df)


def fdr_summary(results_df: pl.DataFrame) -> dict:
    """Compute FDR summary statistics.

    Args:
        results_df: DataFrame with 'q_value' and 'is_decoy' columns.

    Returns:
        Dictionary with summary statistics.
    """
    targets = results_df.filter(~pl.col("is_decoy"))
    decoys = results_df.filter(pl.col("is_decoy"))

    thresholds = [0.01, 0.05, 0.10]
    counts = {}
    for t in thresholds:
        n_at_fdr = len(targets.filter(pl.col("q_value") <= t))
        counts[f"peptides_at_{int(t*100)}pct_fdr"] = n_at_fdr

    return {
        "total_targets": len(targets),
        "total_decoys": len(decoys),
        "mean_target_score": float(targets["score"].mean()) if len(targets) > 0 else 0.0,
        "mean_decoy_score": float(decoys["score"].mean()) if len(decoys) > 0 else 0.0,
        **counts,
    }
