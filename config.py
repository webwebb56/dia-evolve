"""Evolvable configuration — all tunable parameters in one place.

The AlphaEvolve agent modifies THIS file for parameter sweeps.
Each parameter has a comment explaining its role and typical range.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiaConfig:
    """Complete configuration for the DIA matching pipeline."""

    # === XIC Extraction ===
    mz_tolerance_ppm: float = 20.0
    """Fragment m/z tolerance for XIC extraction (ppm). Range: 5-50."""

    rt_window_minutes: float = 5.0
    """RT window half-width around predicted apex (minutes). Range: 1-15."""

    min_fragment_intensity: float = 0.01
    """Minimum predicted relative intensity to use a fragment. Range: 0.001-0.1."""

    max_fragments_per_peptide: int = 15
    """Maximum number of fragments per peptide (top-N by predicted intensity)."""

    # === Peak Detection ===
    smooth_sigma_scans: float = 3.0
    """Gaussian smoothing sigma in scan units. Range: 1-10."""

    min_peak_snr: float = 3.0
    """Minimum signal-to-noise ratio for peak acceptance. Range: 1-10."""

    min_peak_width_scans: int = 3
    """Minimum peak width in scans. Range: 2-10."""

    min_peak_intensity: float = 100.0
    """Absolute minimum intensity for a peak. Range: 10-10000."""

    # === Scoring ===
    feature_weights: dict[str, float] = field(default_factory=lambda: {
        "library_cosine": 2.0,
        "co_elution": 1.5,
        "delta_rt_sigma": -0.5,
        "fragment_count": 0.3,
        "log_total_intensity": 0.2,
        "asymmetry": -0.3,
        "jaggedness": -0.4,
    })
    """Per-feature weights for linear discriminant scoring."""

    score_threshold: float = 0.0
    """Pre-FDR score threshold (discard very low scores early). Range: -5 to 5."""

    # === FDR Control ===
    fdr_threshold: float = 0.01
    """FDR threshold for final filtering. Standard: 0.01."""

    decoy_method: str = "pseudo_reverse"
    """Decoy generation method: 'pseudo_reverse' or 'shuffle'."""

    # === Candidate Filtering ===
    precursor_mz_tolerance_da: float = 0.5
    """How much wider than the isolation window to search for precursors (Da)."""

    # === Processing ===
    batch_size: int = 1000
    """Number of candidates to process at a time."""

    def to_dict(self) -> dict:
        """Serialize config to dict for logging."""
        return {
            k: v for k, v in self.__dict__.items()
        }
