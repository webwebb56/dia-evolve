"""Prior library loading and in-memory representation.

Supports two modes:
1. Load from ADAPT prior Parquet (production path)
2. Build from FASTA + UniSpec predictions (bootstrap path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


def load_prior_parquet(path: str | Path) -> pl.DataFrame:
    """Load an ADAPT prior Parquet file.

    Expected columns (from adapt-prior schema.rs):
      - sequence: Utf8 (bare peptide sequence)
      - mods: Utf8 (ProForma modification string)
      - charge: Int8
      - precursor_mz: Float32
      - urt_prior: Float32 (universal RT, 0-100)
      - urt_sigma: Float32 (RT uncertainty)
      - frag_mz: List[Float32]
      - frag_int: List[Float32]
      - frag_meta: List[Int16] (optional)
      - species_ids: List[UInt16] (optional, for 3-species benchmark)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prior file not found: {path}")

    df = pl.read_parquet(path)
    required = {"sequence", "charge", "precursor_mz", "frag_mz", "frag_int"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prior missing required columns: {missing}")
    return df


def build_prior_from_fasta(
    fasta_paths: dict[str, str | Path],
    max_missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    charges: tuple[int, ...] = (2, 3),
    species_labels: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """Build a minimal prior from FASTA files using pyteomics.

    Args:
        fasta_paths: Mapping of species name to FASTA file path.
        max_missed_cleavages: Maximum missed cleavages for trypsin.
        min_length: Minimum peptide length.
        max_length: Maximum peptide length.
        charges: Charge states to generate.
        species_labels: Optional mapping of species name to numeric ID.

    Returns:
        polars DataFrame with prior columns.
    """
    from pyteomics import fasta, mass, parser

    rows = []
    if species_labels is None:
        species_labels = {name: i for i, name in enumerate(fasta_paths)}

    for species_name, fasta_path in fasta_paths.items():
        species_id = species_labels[species_name]
        for desc, seq in fasta.read(str(fasta_path)):
            peptides = parser.cleave(
                seq,
                rule="trypsin",
                missed_cleavages=max_missed_cleavages,
            )
            for pep in peptides:
                if not (min_length <= len(pep) <= max_length):
                    continue
                # Skip peptides with non-standard amino acids
                if any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in pep):
                    continue
                try:
                    mono_mass = mass.fast_mass(pep, ion_type="M", charge=0)
                except Exception:
                    continue

                for z in charges:
                    precursor_mz = (mono_mass + z * 1.00727646677) / z
                    rows.append({
                        "sequence": pep,
                        "mods": "",
                        "charge": z,
                        "precursor_mz": float(precursor_mz),
                        "mass": float(mono_mass),
                        "length": len(pep),
                        "missed_cleavages": parser.num_sites(pep, "trypsin"),
                        "species_id": species_id,
                        "species_name": species_name,
                    })

    df = pl.DataFrame(rows)

    # Deduplicate: same sequence can appear in multiple species
    # Keep species info as list for shared peptides
    df = df.unique(subset=["sequence", "charge"], keep="first")

    # Add placeholder fragment columns (to be filled by UniSpec)
    n = len(df)
    df = df.with_columns([
        pl.Series("frag_mz", [[] for _ in range(n)], dtype=pl.List(pl.Float32)),
        pl.Series("frag_int", [[] for _ in range(n)], dtype=pl.List(pl.Float32)),
        pl.lit(0.0).cast(pl.Float32).alias("urt_prior"),
        pl.lit(5.0).cast(pl.Float32).alias("urt_sigma"),
    ])

    return df


def add_theoretical_fragments(
    prior_df: pl.DataFrame,
    max_fragments: int = 15,
) -> pl.DataFrame:
    """Add theoretical b/y ion fragments to a prior DataFrame.

    This is a simple fallback when UniSpec predictions aren't available.
    Generates b and y ions for charge 1 only, uniform intensity.
    """
    from pyteomics import mass

    frag_mz_list = []
    frag_int_list = []

    for seq in prior_df["sequence"].to_list():
        mz_vals = []
        for i in range(1, len(seq)):
            try:
                # y ions
                y_seq = seq[i:]
                y_mz = mass.fast_mass(y_seq, ion_type="M", charge=0) + 1.00727646677
                mz_vals.append(float(y_mz))
                # b ions
                b_seq = seq[:i]
                b_mz = mass.fast_mass(b_seq, ion_type="M", charge=0) + 1.00727646677
                mz_vals.append(float(b_mz))
            except Exception:
                continue

        # Sort by m/z, take top-N
        mz_vals.sort()
        mz_vals = mz_vals[:max_fragments]
        # Uniform intensity for theoretical (no prediction model)
        int_vals = [1.0 / len(mz_vals)] * len(mz_vals) if mz_vals else []

        frag_mz_list.append(mz_vals)
        frag_int_list.append(int_vals)

    return prior_df.with_columns([
        pl.Series("frag_mz", frag_mz_list, dtype=pl.List(pl.Float32)),
        pl.Series("frag_int", frag_int_list, dtype=pl.List(pl.Float32)),
    ])
