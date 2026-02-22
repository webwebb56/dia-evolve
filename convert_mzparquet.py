"""Convert ThermoRawFileParser mzparquet (per-peak) to per-spectrum format.

ThermoRawFileParser outputs one row per (scan, m/z, intensity) peak.
Our pipeline expects one row per spectrum with array columns.

Usage:
    cd projects/adapt/prototypes
    python -m dia_evolve.convert_mzparquet \
        --input data/benchmark/lfqbench/parquet/file.mzparquet \
        --output data/benchmark/lfqbench/parquet/file.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl


def convert_mzparquet_to_spectrum(
    input_path: str | Path,
    output_path: str | Path,
    ms_level: int = 2,
) -> pl.DataFrame:
    """Convert per-peak mzparquet to per-spectrum parquet.

    Input schema (ThermoRawFileParser):
        scan: UInt32, level: UInt8, rt: Float32,
        mz: Float32, intensity: Float32,
        isolation_lower: Float32, isolation_upper: Float32

    Output schema (our pipeline):
        scan_idx: UInt32, scan_start_time: Float64,
        ms_level: UInt8, precursor_mz: Float64,
        mz_array: List[Int64] (µDa), intensity_array: List[UInt32]
    """
    log = logging.getLogger(__name__)

    input_path = Path(input_path)
    output_path = Path(output_path)

    log.info("Reading %s ...", input_path)
    t0 = time.time()

    # Read only MS2 data
    df = pl.read_parquet(str(input_path))
    log.info("  Total rows: %d", len(df))

    df = df.filter(pl.col("level") == ms_level)
    log.info("  MS%d rows: %d", ms_level, len(df))

    # Group by scan to build per-spectrum arrays
    log.info("Grouping by scan...")

    spectrum_df = (
        df.group_by("scan")
        .agg([
            pl.col("rt").first().alias("scan_start_time"),
            pl.col("level").first().alias("ms_level"),
            pl.col("isolation_lower").first().alias("iso_lower"),
            pl.col("isolation_upper").first().alias("iso_upper"),
            # Sort m/z within each scan and collect as arrays
            pl.col("mz").sort_by("mz").alias("mz_array_f32"),
            pl.col("intensity").sort_by("mz").alias("int_array_f32"),
        ])
        .sort("scan")
    )

    log.info("  Unique spectra: %d", len(spectrum_df))

    # Compute precursor_mz as center of isolation window
    spectrum_df = spectrum_df.with_columns([
        ((pl.col("iso_lower") + pl.col("iso_upper")) / 2.0)
        .cast(pl.Float64)
        .alias("precursor_mz"),
        pl.col("scan").alias("scan_idx"),
        pl.col("scan_start_time").cast(pl.Float64),
    ])

    # Convert m/z to µDa Int64 encoding and intensity to UInt32
    # Use polars list expressions for vectorized conversion
    log.info("Converting to µDa encoding...")

    spectrum_df = spectrum_df.with_columns([
        pl.col("mz_array_f32")
        .list.eval(pl.element().cast(pl.Float64) * 1_000_000)
        .list.eval(pl.element().cast(pl.Int64))
        .alias("mz_array"),
        pl.col("int_array_f32")
        .list.eval(pl.element().cast(pl.UInt32))
        .alias("intensity_array"),
    ])

    # Select final columns
    result = spectrum_df.select([
        "scan_idx",
        "scan_start_time",
        "ms_level",
        "precursor_mz",
        "mz_array",
        "intensity_array",
    ])

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(output_path))

    elapsed = time.time() - t0
    size_mb = output_path.stat().st_size / 1e6
    log.info("Written %s (%.1f MB) in %.1fs", output_path, size_mb, elapsed)
    log.info("  %d spectra, RT range: %.2f - %.2f min",
             len(result),
             result["scan_start_time"].min(),
             result["scan_start_time"].max())

    # Report isolation window info
    n_windows = result["precursor_mz"].n_unique()
    log.info("  %d unique isolation windows", n_windows)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert ThermoRawFileParser mzparquet to per-spectrum format",
    )
    parser.add_argument("--input", "-i", required=True, help="Input .mzparquet file")
    parser.add_argument("--output", "-o", required=True, help="Output .parquet file")
    parser.add_argument("--ms-level", type=int, default=2, help="MS level to extract (default: 2)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    convert_mzparquet_to_spectrum(args.input, args.output, args.ms_level)


if __name__ == "__main__":
    main()
