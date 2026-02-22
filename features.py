"""RSM feature computation from XIC traces and peaks.

Computes the feature vector used for discriminant scoring.
Features follow the spec in docs/architecture/micro-xic-matching.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DiaConfig
from .peak import PeakResult
from .xic import XicTrace


@dataclass
class RsmFeatures:
    """RSM (Rescoring Model) feature vector for one peptide candidate."""

    library_cosine: float = 0.0
    co_elution: float = 0.0
    delta_rt_sigma: float = 0.0
    fragment_count: int = 0
    log_total_intensity: float = 0.0
    asymmetry: float = 0.0
    jaggedness: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "library_cosine": self.library_cosine,
            "co_elution": self.co_elution,
            "delta_rt_sigma": self.delta_rt_sigma,
            "fragment_count": float(self.fragment_count),
            "log_total_intensity": self.log_total_intensity,
            "asymmetry": self.asymmetry,
            "jaggedness": self.jaggedness,
        }


def compute_features(
    peak_results: list[PeakResult],
    predicted_rt: float | None = None,
    rt_sigma: float = 5.0,
    config: DiaConfig | None = None,
) -> RsmFeatures:
    """Compute RSM features from peak detection results.

    Args:
        peak_results: Peak detection results for each fragment XIC.
        predicted_rt: Predicted retention time (minutes), or None.
        rt_sigma: RT prediction uncertainty (minutes).
        config: Pipeline configuration (unused for now, for future expansion).

    Returns:
        RsmFeatures with computed values.
    """
    # Separate detected vs undetected peaks
    detected = [pr for pr in peak_results if pr.peak is not None]
    fragment_count = len(detected)

    if fragment_count == 0:
        return RsmFeatures(fragment_count=0)

    # --- Library Cosine Similarity ---
    library_cosine = _compute_library_cosine(detected)

    # --- Co-Elution Score ---
    co_elution = _compute_co_elution(detected)

    # --- Delta RT ---
    # Use median apex RT across detected fragments
    apex_rts = [pr.peak.apex_rt for pr in detected]
    median_apex_rt = float(np.median(apex_rts))

    delta_rt_sigma = 0.0
    if predicted_rt is not None and rt_sigma > 0:
        delta_rt_sigma = abs(median_apex_rt - predicted_rt) / rt_sigma

    # --- Total Intensity ---
    total_intensity = sum(pr.peak.apex_intensity for pr in detected)
    log_total_intensity = float(np.log1p(total_intensity))

    # --- Peak Shape (average across fragments) ---
    asymmetries = [pr.peak.asymmetry for pr in detected]
    jaggedness_vals = [pr.peak.jaggedness for pr in detected]
    avg_asymmetry = float(np.mean(np.abs(asymmetries)))
    avg_jaggedness = float(np.mean(jaggedness_vals))

    return RsmFeatures(
        library_cosine=library_cosine,
        co_elution=co_elution,
        delta_rt_sigma=delta_rt_sigma,
        fragment_count=fragment_count,
        log_total_intensity=log_total_intensity,
        asymmetry=avg_asymmetry,
        jaggedness=avg_jaggedness,
    )


def _compute_library_cosine(detected: list[PeakResult]) -> float:
    """Compute cosine similarity between predicted and observed fragment intensities.

    Uses log-transformed intensities as per micro-xic-matching.md:
      cosine = dot(pred, obs) / (||pred|| * ||obs||)
    where intensities are log(1 + x) transformed.
    """
    pred = np.array([pr.trace.predicted_intensity for pr in detected])
    obs = np.array([pr.peak.apex_intensity for pr in detected])

    # Log-transform
    pred_log = np.log1p(pred)
    obs_log = np.log1p(obs)

    # L2 norms
    pred_norm = np.linalg.norm(pred_log)
    obs_norm = np.linalg.norm(obs_log)

    if pred_norm == 0 or obs_norm == 0:
        return 0.0

    cosine = float(np.dot(pred_log, obs_log) / (pred_norm * obs_norm))
    return max(0.0, min(1.0, cosine))


def _compute_co_elution(detected: list[PeakResult]) -> float:
    """Compute co-elution score: mean pairwise Pearson correlation of fragment XICs.

    True peptide fragments elute together (high correlation).
    """
    if len(detected) < 2:
        return 1.0  # Single fragment: perfect co-elution by definition

    # Collect smoothed XICs aligned at the apex region
    xics = []
    for pr in detected:
        peak = pr.peak
        # Extract region around peak
        left = peak.left_idx
        right = peak.right_idx + 1
        segment = pr.smoothed[left:right]
        if len(segment) > 0:
            xics.append(segment)

    if len(xics) < 2:
        return 1.0

    # Align to same length (use shortest)
    min_len = min(len(x) for x in xics)
    if min_len < 3:
        return 0.5  # Too short to compute meaningful correlation

    # Trim all to same length (centered on apex)
    aligned = []
    for x in xics:
        start = (len(x) - min_len) // 2
        aligned.append(x[start:start + min_len])

    # Compute pairwise Pearson correlations
    correlations = []
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            a, b = aligned[i], aligned[j]
            std_a, std_b = np.std(a), np.std(b)
            if std_a > 0 and std_b > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)

    if not correlations:
        return 0.5

    return float(np.mean(correlations))
