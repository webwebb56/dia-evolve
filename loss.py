"""3-Species loss function — THE OBJECTIVE.

This file is NEVER modified by the AlphaEvolve agent.
It defines ground-truth evaluation against a 3-species benchmark.

Metrics:
1. species_fdr: fraction of IDs from unexpected species (lower = better)
2. quant_error: MSE of observed vs expected log2 ratios (lower = better)
3. depth: total peptides at 1% FDR (higher = better)
4. id_rate: fraction of windows with confident ID (higher = better)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class LossMetrics:
    """Multi-objective loss metrics from 3-species benchmark."""

    species_fdr: float     # Fraction of wrong-species IDs [0, 1] (lower = better)
    quant_error: float     # MSE of log2 ratio error (lower = better)
    depth: int             # Total peptides at 1% FDR (higher = better)
    id_rate: float         # Fraction of windows with confident ID [0, 1]
    total_targets: int     # Total target PSMs before FDR
    total_decoys: int      # Total decoy PSMs

    @property
    def composite_loss(self) -> float:
        """Compute weighted composite loss (lower = better).

        Weights chosen to balance:
        - species_fdr: Most important (wrong species = bad)
        - quant_error: Second (quantitative accuracy)
        - depth: Reward more IDs (negative contribution to loss)
        - id_rate: Mild reward for coverage

        The depth component is normalized to [0,1] range assuming
        a reasonable maximum of 20000 peptides.
        """
        depth_normalized = min(self.depth / 20000.0, 1.0)
        return (
            5.0 * self.species_fdr
            + 2.0 * self.quant_error
            - 3.0 * depth_normalized
            - 1.0 * self.id_rate
        )

    def to_dict(self) -> dict:
        return {
            "species_fdr": self.species_fdr,
            "quant_error": self.quant_error,
            "depth": self.depth,
            "id_rate": self.id_rate,
            "composite_loss": self.composite_loss,
            "total_targets": self.total_targets,
            "total_decoys": self.total_decoys,
        }

    def summary(self) -> str:
        return (
            f"Loss: {self.composite_loss:.4f} | "
            f"Depth: {self.depth} | "
            f"Species FDR: {self.species_fdr:.4f} | "
            f"Quant Error: {self.quant_error:.4f} | "
            f"ID Rate: {self.id_rate:.4f}"
        )


def compute_loss(
    results_df: pl.DataFrame,
    prior_df: pl.DataFrame,
    species_ratios: dict[int, dict[str, float]],
    total_windows: int,
) -> LossMetrics:
    """Compute multi-objective loss from 3-species benchmark results.

    Args:
        results_df: Identified peptides after FDR filtering.
            Required columns: sequence, charge, q_value, is_decoy, score
        prior_df: Prior library with species information.
            Required columns: sequence, species_id (or species_name)
        species_ratios: Expected mixing ratios per condition.
            Format: {condition_id: {"species_A": ratio_A, "species_B": ratio_B, ...}}
            For single-condition: {0: {"human": 0.65, "yeast": 0.2, "ecoli": 0.15}}
        total_windows: Total number of isolation windows in the DIA run
            (used for id_rate calculation).

    Returns:
        LossMetrics with all computed metrics.
    """
    if len(results_df) == 0:
        return LossMetrics(
            species_fdr=1.0,
            quant_error=10.0,  # Large but finite penalty
            depth=0,
            id_rate=0.0,
            total_targets=0,
            total_decoys=0,
        )

    # Separate targets and decoys
    targets = results_df.filter(~pl.col("is_decoy"))
    decoys = results_df.filter(pl.col("is_decoy"))

    if len(targets) == 0:
        return LossMetrics(
            species_fdr=1.0,
            quant_error=10.0,  # Large but finite penalty
            depth=0,
            id_rate=0.0,
            total_targets=0,
            total_decoys=len(decoys),
        )

    # --- Map peptides to species ---
    # Normalize species_ratios keys to match prior's species column type
    species_col = "species_id" if "species_id" in prior_df.columns else "species_name"

    # Auto-convert species_ratios keys to match prior column type
    normalized_ratios = {}
    for cond_key, ratios in species_ratios.items():
        normalized = {}
        for k, v in ratios.items():
            normalized[str(k)] = v
        normalized_ratios[cond_key] = normalized
    species_ratios = normalized_ratios

    if species_col not in prior_df.columns:
        # Can't compute species metrics without species info
        return LossMetrics(
            species_fdr=0.0,
            quant_error=0.0,
            depth=len(targets),
            id_rate=len(targets) / max(total_windows, 1),
            total_targets=len(targets),
            total_decoys=len(decoys),
        )

    # Join results with prior to get species
    species_map = prior_df.select(["sequence", species_col]).unique(subset=["sequence"])
    targets_with_species = targets.join(
        species_map, on="sequence", how="left"
    )

    # --- Metric 1: Species FDR ---
    # Identify expected species (those in the ratio dict)
    expected_species = set()
    for condition_ratios in species_ratios.values():
        expected_species.update(condition_ratios.keys())

    if species_col in targets_with_species.columns:
        species_values = targets_with_species[species_col].to_list()
        n_wrong = sum(
            1 for s in species_values
            if s is not None and str(s) not in expected_species
            and s not in expected_species
        )
        species_fdr = n_wrong / len(targets) if len(targets) > 0 else 0.0
    else:
        species_fdr = 0.0

    # --- Metric 2: Quant Error ---
    quant_error = _compute_quant_error(targets_with_species, species_ratios, species_col)

    # --- Metric 3: Depth ---
    # Unique peptides at 1% FDR
    depth = len(targets.unique(subset=["sequence", "charge"]))

    # --- Metric 4: ID Rate ---
    id_rate = min(len(targets) / max(total_windows, 1), 1.0)

    return LossMetrics(
        species_fdr=species_fdr,
        quant_error=quant_error,
        depth=depth,
        id_rate=id_rate,
        total_targets=len(targets),
        total_decoys=len(decoys),
    )


def _compute_quant_error(
    targets_df: pl.DataFrame,
    species_ratios: dict[int, dict[str, float]],
    species_col: str,
) -> float:
    """Compute quantitative error: MSE of observed vs expected log2 ratios.

    For a 3-species benchmark, the expected ratios are known (e.g., 65/20/15).
    We compute the observed fraction of IDs per species and compare.
    """
    if species_col not in targets_df.columns or len(targets_df) == 0:
        return 0.0

    # Get observed species counts
    species_counts = (
        targets_df
        .group_by(species_col)
        .agg(pl.len().alias("count"))
    )

    total = species_counts["count"].sum()
    if total == 0:
        return 10.0  # Large but finite penalty

    # Use first condition's ratios (single-condition benchmark)
    condition_key = list(species_ratios.keys())[0]
    expected_ratios = species_ratios[condition_key]

    # Compute observed ratios
    observed_ratios = {}
    for row in species_counts.iter_rows(named=True):
        species = str(row[species_col])
        observed_ratios[species] = row["count"] / total

    # MSE of log2(observed/expected) for species with expected ratios
    errors = []
    for species, expected in expected_ratios.items():
        observed = observed_ratios.get(species, 0.0)
        if observed > 0 and expected > 0:
            log2_ratio = np.log2(observed / expected)
            errors.append(log2_ratio ** 2)
        elif expected > 0 and observed == 0:
            # Missing species entirely: large penalty
            errors.append(10.0)

    return float(np.mean(errors)) if errors else 0.0
