"""Decoy library generation for target-decoy FDR estimation.

Implements pseudo-reverse strategy per fdr-control.md:
  Target:  PEPTIDER
  Reverse: REDITPE
  Keep C-terminal: REDITPER  (preserves tryptic constraint)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import DiaConfig


def generate_decoys(prior_df: pl.DataFrame, config: DiaConfig) -> pl.DataFrame:
    """Generate a decoy library from the target prior.

    Each target peptide gets one decoy with reversed sequence
    (keeping C-terminal residue fixed for tryptic constraint).

    The decoy retains the same precursor_mz, charge, and fragment properties
    but gets a modified sequence to mark it as a decoy.

    Args:
        prior_df: Target prior DataFrame.
        config: Pipeline configuration.

    Returns:
        Decoy DataFrame with same schema plus 'is_decoy' column.
    """
    if config.decoy_method == "pseudo_reverse":
        return _pseudo_reverse_decoys(prior_df)
    elif config.decoy_method == "shuffle":
        return _shuffle_decoys(prior_df)
    else:
        raise ValueError(f"Unknown decoy method: {config.decoy_method}")


def _pseudo_reverse_decoys(prior_df: pl.DataFrame) -> pl.DataFrame:
    """Generate pseudo-reverse decoy sequences.

    For each sequence:
    1. Keep C-terminal residue (tryptic constraint: K or R)
    2. Reverse the rest
    3. Concatenate
    """
    sequences = prior_df["sequence"].to_list()
    decoy_sequences = []
    for seq in sequences:
        if len(seq) <= 2:
            # Too short to reverse meaningfully
            decoy_sequences.append(seq[::-1])
        else:
            # Reverse all except C-terminal
            c_term = seq[-1]
            reversed_body = seq[:-1][::-1]
            decoy_seq = reversed_body + c_term
            # Check if decoy equals target (palindrome-like)
            if decoy_seq == seq:
                # Swap first two residues
                if len(decoy_seq) >= 2:
                    decoy_seq = decoy_seq[1] + decoy_seq[0] + decoy_seq[2:]
            decoy_sequences.append(decoy_seq)

    # Build decoy DataFrame
    decoy_df = prior_df.clone()
    decoy_df = decoy_df.with_columns([
        pl.Series("sequence", decoy_sequences),
        pl.lit(True).alias("is_decoy"),
    ])

    # Recompute fragment m/z for decoys using shifted fragments
    # For v1, we shift fragment m/z slightly to avoid exact overlap
    if "frag_mz" in decoy_df.columns:
        decoy_df = _shift_decoy_fragments(decoy_df)

    return decoy_df


def _shuffle_decoys(prior_df: pl.DataFrame) -> pl.DataFrame:
    """Generate shuffle-based decoy sequences.

    Randomly shuffle internal residues (keep N and C terminal).
    """
    rng = np.random.default_rng(seed=42)
    sequences = prior_df["sequence"].to_list()
    decoy_sequences = []

    for seq in sequences:
        if len(seq) <= 3:
            decoy_sequences.append(seq[::-1])
        else:
            internal = list(seq[1:-1])
            rng.shuffle(internal)
            decoy_seq = seq[0] + "".join(internal) + seq[-1]
            if decoy_seq == seq:
                # Swap two internal positions
                if len(internal) >= 2:
                    internal[0], internal[1] = internal[1], internal[0]
                    decoy_seq = seq[0] + "".join(internal) + seq[-1]
            decoy_sequences.append(decoy_seq)

    decoy_df = prior_df.clone()
    decoy_df = decoy_df.with_columns([
        pl.Series("sequence", decoy_sequences),
        pl.lit(True).alias("is_decoy"),
    ])

    if "frag_mz" in decoy_df.columns:
        decoy_df = _shift_decoy_fragments(decoy_df)

    return decoy_df


def _shift_decoy_fragments(decoy_df: pl.DataFrame) -> pl.DataFrame:
    """Shift decoy fragment m/z values to avoid exact target overlap.

    Applies small random shifts (within mass accuracy) to each fragment.
    This ensures decoys are scored independently from targets.
    """
    rng = np.random.default_rng(seed=123)
    frag_mz_list = decoy_df["frag_mz"].to_list()
    shifted = []
    for frags in frag_mz_list:
        if frags:
            # Shift each fragment by a small random amount (0.001-0.01 Da)
            shifts = rng.uniform(-0.005, 0.005, size=len(frags))
            shifted.append([float(mz + s) for mz, s in zip(frags, shifts)])
        else:
            shifted.append([])

    return decoy_df.with_columns(
        pl.Series("frag_mz", shifted, dtype=pl.List(pl.Float32))
    )


def add_decoy_column(target_df: pl.DataFrame) -> pl.DataFrame:
    """Add is_decoy=False column to target DataFrame."""
    if "is_decoy" not in target_df.columns:
        return target_df.with_columns(pl.lit(False).alias("is_decoy"))
    return target_df
