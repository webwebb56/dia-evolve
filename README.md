# dia-evolve

ADAPT-DIA Python prototype with AlphaEvolve-style optimization for DIA peptide identification.

## Overview

A modular DIA (Data-Independent Acquisition) matching pipeline designed for iterative optimization using Claude Code as an AlphaEvolve-style agent. The pipeline takes raw DIA mass spectrometry data and a prior peptide library, and produces peptide identifications with FDR control.

## Architecture

```
prior.py          → Load peptide library (Parquet or FASTA)
parquet_reader.py  → Read DIA Parquet files (PSI-MS v2-int schema)
candidates.py      → Predicate pushdown: filter prior per isolation window
xic.py             → XIC extraction from DIA spectra
peak.py            → Gaussian smoothing, apex finding, peak shape metrics
features.py        → RSM features: cosine, co-elution, delta-RT, peak shape
scoring.py         → Linear discriminant scoring (evolvable)
decoys.py          → Pseudo-reverse decoy generation
fdr.py             → Target-decoy q-value estimation
loss.py            → 3-species benchmark objective (NEVER modified by agent)
pipeline.py        → Full orchestrator
config.py          → All tunable parameters in one place
```

## Quick Start

```bash
# Install dependencies
pip install polars numpy scipy pyteomics pyarrow

# Run smoke test (synthetic data)
python -m smoke_test

# Run on real data
python -m run --dia data/run.parquet --prior data/prior.parquet
```

## AlphaEvolve Design

The agent optimizes the pipeline by modifying:
- **`config.py`** — parameter sweeps (fast, safe)
- **`features.py`** — feature engineering
- **`scoring.py`** — scoring function
- **`xic.py`** — extraction strategy
- **`peak.py`** — peak detection algorithm

The agent does NOT modify:
- **`loss.py`** — ground truth objective function

## Dependencies

- polars >= 1.0
- numpy
- scipy
- pyteomics
- pyarrow

## 3-Species Loss Function

The objective evaluates against a 3-species benchmark (Human/Yeast/Ecoli):
1. **Species FDR**: Wrong-species identifications (lower = better)
2. **Quant Error**: MSE of observed vs expected log2 ratios (lower = better)
3. **Depth**: Total peptides at 1% FDR (higher = better)
4. **ID Rate**: Fraction of windows with confident ID (higher = better)
