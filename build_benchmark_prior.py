"""Build a 3-species prior library from FASTA files for the LFQBench benchmark.

Usage:
    cd projects/adapt/prototypes
    python -m dia_evolve.build_benchmark_prior

Outputs:
    data/benchmark/lfqbench/prior_3species.parquet
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .prior import add_theoretical_fragments, build_prior_from_fasta

BASE = Path(__file__).resolve().parents[4]  # Mage-ADAPT root
FASTA_DIR = BASE / "data" / "benchmark" / "lfqbench" / "fasta"
OUTPUT = BASE / "data" / "benchmark" / "lfqbench" / "prior_3species.parquet"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    fasta_paths = {
        "human": str(FASTA_DIR / "human_swissprot.fasta"),
        "yeast": str(FASTA_DIR / "yeast_swissprot.fasta"),
        "ecoli": str(FASTA_DIR / "ecoli_swissprot.fasta"),
    }

    for name, path in fasta_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"FASTA not found: {path}")
        log.info("FASTA: %s → %s", name, path)

    species_labels = {"human": 0, "yeast": 1, "ecoli": 2}

    log.info("Building prior from FASTA (trypsin digest, charge 2-3)...")
    t0 = time.time()

    prior_df = build_prior_from_fasta(
        fasta_paths=fasta_paths,
        max_missed_cleavages=1,
        min_length=7,
        max_length=30,
        charges=(2, 3),
        species_labels=species_labels,
    )
    t1 = time.time()
    log.info("  Digest complete: %d peptidoforms in %.1fs", len(prior_df), t1 - t0)

    # Stats by species
    if "species_id" in prior_df.columns:
        for sid, name in [(0, "human"), (1, "yeast"), (2, "ecoli")]:
            n = len(prior_df.filter(prior_df["species_id"] == sid))
            log.info("  %s: %d peptidoforms", name, n)

    # Add theoretical b/y fragments
    log.info("Adding theoretical b/y ion fragments...")
    prior_df = add_theoretical_fragments(prior_df, max_fragments=15)
    t2 = time.time()
    log.info("  Fragments added in %.1fs", t2 - t1)

    # Stats on fragments
    frag_lens = prior_df["frag_mz"].list.len()
    log.info("  Fragment count: mean=%.1f, median=%.1f, max=%d",
             frag_lens.mean(), frag_lens.median(), frag_lens.max())

    # Precursor m/z range
    log.info("  Precursor m/z range: %.1f - %.1f",
             prior_df["precursor_mz"].min(), prior_df["precursor_mz"].max())

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prior_df.write_parquet(str(OUTPUT))
    size_mb = OUTPUT.stat().st_size / 1e6
    log.info("Prior written to %s (%.1f MB)", OUTPUT, size_mb)

    log.info("Done! Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
