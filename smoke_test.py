"""Smoke test — generate synthetic DIA data and run the pipeline end-to-end.

This creates a small synthetic dataset with known peptides and verifies
the pipeline produces results without crashing.

Usage:
    cd projects/adapt/prototypes
    python -m dia_evolve.smoke_test
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .config import DiaConfig
from .pipeline import run_pipeline


def generate_synthetic_prior(n_peptides: int = 50, seed: int = 42) -> pl.DataFrame:
    """Generate a synthetic prior library for testing.

    Creates peptides with realistic precursor m/z values and
    synthetic fragment m/z/intensity values.
    """
    rng = np.random.default_rng(seed)

    sequences = []
    charges = []
    precursor_mzs = []
    frag_mz_list = []
    frag_int_list = []
    species_ids = []

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    for i in range(n_peptides):
        # Generate random peptide (7-20 residues, ending in K or R)
        length = rng.integers(7, 21)
        seq = "".join(rng.choice(amino_acids, size=length - 1)) + rng.choice(["K", "R"])
        charge = int(rng.choice([2, 3]))

        # Approximate mass from sequence length (~111 Da per residue)
        approx_mass = length * 111.0 + rng.uniform(-10, 10)
        prec_mz = (approx_mass + charge * 1.00728) / charge

        # Generate 5-10 fragment m/z values (b/y ions)
        n_frags = rng.integers(5, 11)
        frag_mzs = sorted(rng.uniform(200, prec_mz * charge, size=n_frags).tolist())
        # Relative intensities (log-normal distribution)
        frag_ints = np.exp(rng.normal(0, 1, size=n_frags))
        frag_ints = (frag_ints / frag_ints.max()).tolist()

        # Assign species (human=0, yeast=1, ecoli=2)
        species = int(rng.choice([0, 0, 0, 1, 2]))  # 60/20/20 split

        sequences.append(seq)
        charges.append(charge)
        precursor_mzs.append(float(prec_mz))
        frag_mz_list.append([float(x) for x in frag_mzs])
        frag_int_list.append([float(x) for x in frag_ints])
        species_ids.append(species)

    return pl.DataFrame({
        "sequence": sequences,
        "mods": [""] * n_peptides,
        "charge": charges,
        "precursor_mz": precursor_mzs,
        "urt_prior": [float(i * 100 / n_peptides) for i in range(n_peptides)],
        "urt_sigma": [5.0] * n_peptides,
        "frag_mz": frag_mz_list,
        "frag_int": frag_int_list,
        "species_id": species_ids,
    }).cast({
        "charge": pl.Int8,
        "precursor_mz": pl.Float32,
        "urt_prior": pl.Float32,
        "urt_sigma": pl.Float32,
    })


def generate_synthetic_dia(
    prior_df: pl.DataFrame,
    n_scans_per_window: int = 30,
    n_inject: int = 10,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic DIA spectra containing some prior peptides.

    Injects fragment peaks from `n_inject` prior peptides into synthetic
    DIA spectra with noise, so the pipeline can detect them.
    """
    rng = np.random.default_rng(seed)

    # Get unique isolation windows (bin precursor m/z to ~25 Da windows)
    all_mz = prior_df["precursor_mz"].to_numpy()
    window_centers = np.arange(
        max(200, np.floor(all_mz.min() / 25) * 25),
        np.ceil(all_mz.max() / 25) * 25 + 25,
        25,
    )

    # Select peptides to inject
    inject_indices = rng.choice(len(prior_df), size=min(n_inject, len(prior_df)), replace=False)
    inject_peptides = prior_df[inject_indices.tolist()]

    rows = []
    scan_idx = 0

    for center_mz in window_centers:
        # Generate RT values
        rt_values = np.linspace(5.0, 60.0, n_scans_per_window)

        for rt in rt_values:
            scan_idx += 1

            # Base noise: random peaks
            n_noise = rng.integers(50, 200)
            noise_mz = sorted(rng.uniform(100, 2000, size=n_noise).tolist())
            noise_int = rng.exponential(50, size=n_noise).astype(int).tolist()

            mz_vals = list(noise_mz)
            int_vals = list(noise_int)

            # Inject fragment peaks from prior peptides
            for row in inject_peptides.iter_rows(named=True):
                prec_mz = row["precursor_mz"]
                # Check if this peptide's precursor falls in this window
                if abs(prec_mz - center_mz) > 12.5:
                    continue

                frag_mzs = row["frag_mz"]
                frag_ints = row["frag_int"]
                if not frag_mzs:
                    continue

                # Inject with Gaussian RT profile (wide peak for synthetic data)
                urt = row.get("urt_prior", 30.0) or 30.0
                peak_rt = 5.0 + urt * 55.0 / 100.0  # Map uRT to RT range
                sigma_rt = 3.0  # Wide peak (~6 min FWHM)
                rt_response = np.exp(-0.5 * ((rt - peak_rt) / sigma_rt) ** 2)

                if rt_response < 0.01:
                    continue

                for fmz, fint in zip(frag_mzs, frag_ints):
                    # Strong signal: 50K base intensity * relative * RT profile
                    intensity = int(fint * 50000 * rt_response * rng.uniform(0.8, 1.2))
                    if intensity > 10:
                        mz_vals.append(float(fmz))
                        int_vals.append(intensity)

            # Sort by m/z
            sorted_pairs = sorted(zip(mz_vals, int_vals))
            mz_sorted = [p[0] for p in sorted_pairs]
            int_sorted = [p[1] for p in sorted_pairs]

            # Encode as micro-Dalton (Int64)
            mz_uda = [int(mz * 1_000_000) for mz in mz_sorted]

            rows.append({
                "scan_idx": scan_idx,
                "scan_start_time": float(rt),
                "ms_level": 2,
                "precursor_mz": float(center_mz),
                "mz_array": mz_uda,
                "intensity_array": int_sorted,
            })

    return pl.DataFrame(rows).cast({
        "scan_idx": pl.UInt32,
        "scan_start_time": pl.Float64,
        "ms_level": pl.UInt8,
        "precursor_mz": pl.Float64,
    })


def run_smoke_test():
    """Run the full smoke test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("ADAPT-DIA Smoke Test")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic prior (50 peptides)...")
    prior_df = generate_synthetic_prior(n_peptides=50)
    print(f"   Prior shape: {prior_df.shape}")
    print(f"   Precursor m/z range: {prior_df['precursor_mz'].min():.1f} - {prior_df['precursor_mz'].max():.1f}")

    print("\n2. Generating synthetic DIA spectra...")
    dia_df = generate_synthetic_dia(prior_df, n_scans_per_window=60, n_inject=15)
    print(f"   DIA shape: {dia_df.shape}")
    print(f"   Unique windows: {dia_df['precursor_mz'].n_unique()}")

    # Write to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        prior_path = Path(tmpdir) / "prior.parquet"
        dia_path = Path(tmpdir) / "dia.parquet"

        prior_df.write_parquet(str(prior_path))
        dia_df.write_parquet(str(dia_path))

        print(f"\n3. Running pipeline...")

        config = DiaConfig(
            min_peak_intensity=10.0,    # Low threshold for synthetic data
            min_peak_snr=2.0,           # Low SNR for synthetic data
            mz_tolerance_ppm=30.0,      # Wider for synthetic (no exact calibration)
            rt_window_minutes=10.0,     # Wide RT window
        )

        species_ratios = {0: {"0": 0.6, "1": 0.2, "2": 0.2}}

        result = run_pipeline(
            dia_path=dia_path,
            prior_path=prior_path,
            config=config,
            species_ratios=species_ratios,
            verbose=True,
        )

        print(f"\n4. Results:")
        print(f"   Windows processed: {result.n_windows_processed}")
        print(f"   Candidates scored: {result.n_candidates_scored}")
        print(f"   Total PSMs: {len(result.results_df)}")
        print(f"   Filtered PSMs (1% FDR): {len(result.filtered_df)}")
        print(f"   Elapsed: {result.elapsed_seconds:.1f}s")
        print()
        print(f"   Loss metrics:")
        print(f"     {result.loss.summary()}")
        print()
        print(f"   FDR stats:")
        for k, v in result.fdr_stats.items():
            print(f"     {k}: {v}")

    # Verification checks
    print("\n5. Verification:")
    checks = {
        "Pipeline ran without errors": True,
        "Candidates were scored": result.n_candidates_scored > 0,
        "PSMs were produced": len(result.results_df) > 0,
        "Loss metrics are finite": (
            np.isfinite(result.loss.species_fdr) and
            np.isfinite(result.loss.composite_loss)
        ),
        "Elapsed time is reasonable": result.elapsed_seconds < 300,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {check}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks PASSED. Pipeline is functional.")
    else:
        print("Some checks FAILED. Review output above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_smoke_test())
