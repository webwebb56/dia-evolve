"""CLI entry point for the DIA matching pipeline.

Usage:
    python -m dia_evolve.run --dia data/benchmark/run.parquet \\
                             --prior data/benchmark/prior.parquet \\
                             [--max-windows 50] [--verbose]

Or from the project root:
    python projects/adapt/prototypes/dia_evolve/run.py --dia ... --prior ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DiaConfig
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="ADAPT-DIA Python prototype — DIA matching pipeline",
    )
    parser.add_argument(
        "--dia", required=True, type=str,
        help="Path to DIA Parquet file",
    )
    parser.add_argument(
        "--prior", required=True, type=str,
        help="Path to prior Parquet file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write results Parquet (optional)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--max-windows", type=int, default=None,
        help="Limit number of isolation windows (for testing)",
    )
    parser.add_argument(
        "--species-ratios", type=str, default=None,
        help='Species ratios as JSON string, e.g. \'{"human": 0.65, "yeast": 0.2, "ecoli": 0.15}\'',
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress messages",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = DiaConfig()
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)

    # Parse species ratios
    species_ratios = {0: {"human": 0.65, "yeast": 0.2, "ecoli": 0.15}}
    if args.species_ratios:
        ratios = json.loads(args.species_ratios)
        species_ratios = {0: ratios}

    verbose = not args.quiet

    # Run pipeline
    result = run_pipeline(
        dia_path=args.dia,
        prior_path=args.prior,
        config=config,
        species_ratios=species_ratios,
        max_windows=args.max_windows,
        verbose=verbose,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ADAPT-DIA Pipeline Results")
    print("=" * 60)
    print(f"  Windows processed: {result.n_windows_processed}")
    print(f"  Candidates scored: {result.n_candidates_scored}")
    print(f"  Elapsed time: {result.elapsed_seconds:.1f}s")
    print()
    print(f"  {result.loss.summary()}")
    print()
    print("  FDR Statistics:")
    for k, v in result.fdr_stats.items():
        print(f"    {k}: {v}")
    print()

    # Write output
    if args.output:
        out_path = Path(args.output)
        result.filtered_df.write_parquet(str(out_path))
        print(f"  Results written to: {out_path}")

    # Also write metrics JSON alongside output
    metrics_path = Path(args.output or "pipeline_metrics.json").with_suffix(".json")
    metrics = {
        "config": config.to_dict(),
        "loss": result.loss.to_dict(),
        "fdr_stats": result.fdr_stats,
        "n_windows": result.n_windows_processed,
        "n_candidates": result.n_candidates_scored,
        "elapsed_seconds": result.elapsed_seconds,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics written to: {metrics_path}")

    print("=" * 60)

    # Return exit code based on whether we got any results
    return 0 if result.loss.depth > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
