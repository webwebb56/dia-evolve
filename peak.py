"""Peak detection on XIC traces.

Smoothing, apex finding, boundary detection, and peak shape metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import DiaConfig
from .xic import XicTrace


@dataclass
class Peak:
    """A detected chromatographic peak."""

    apex_idx: int           # Index of apex in the XIC trace
    apex_rt: float          # RT at apex (minutes)
    apex_intensity: float   # Intensity at apex
    left_idx: int           # Left boundary index
    right_idx: int          # Right boundary index
    area: float             # Integrated peak area
    width_scans: int        # Width in number of scans
    snr: float              # Signal-to-noise ratio
    asymmetry: float        # Peak asymmetry (-1 to 1)
    jaggedness: float       # Peak roughness metric


@dataclass
class PeakResult:
    """Peak detection result for one XIC trace."""

    trace: XicTrace
    peak: Peak | None       # Best peak, or None if no valid peak found
    smoothed: np.ndarray    # Smoothed intensity trace


def detect_peak(
    trace: XicTrace,
    config: DiaConfig,
) -> PeakResult:
    """Detect the best peak in an XIC trace.

    Steps:
    1. Gaussian smoothing
    2. Find local maxima
    3. Select best peak (highest intensity)
    4. Compute boundaries and shape metrics
    5. Apply quality filters

    Args:
        trace: XIC trace (rt, intensity arrays).
        config: Pipeline configuration.

    Returns:
        PeakResult with best peak (or None) and smoothed trace.
    """
    n = len(trace.intensity)
    if n < config.min_peak_width_scans:
        return PeakResult(trace=trace, peak=None, smoothed=trace.intensity.copy())

    # Step 1: Gaussian smoothing
    smoothed = gaussian_filter1d(
        trace.intensity.astype(np.float64),
        sigma=config.smooth_sigma_scans,
        mode="nearest",
    )

    # Step 2: Find local maxima via derivative sign change
    if n < 3:
        return PeakResult(trace=trace, peak=None, smoothed=smoothed)

    diff = np.diff(smoothed)
    maxima = []
    for i in range(len(diff) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            maxima.append(i + 1)

    # Also check endpoints if they're the global max
    if len(maxima) == 0:
        apex_idx = int(np.argmax(smoothed))
        if smoothed[apex_idx] > 0:
            maxima = [apex_idx]

    if not maxima:
        return PeakResult(trace=trace, peak=None, smoothed=smoothed)

    # Step 3: Select best peak (highest smoothed intensity)
    best_idx = max(maxima, key=lambda i: smoothed[i])
    apex_intensity = smoothed[best_idx]

    # Step 4: Estimate noise from baseline
    noise = _estimate_noise(smoothed, best_idx)
    snr = apex_intensity / noise if noise > 0 else apex_intensity

    # Step 5: Find boundaries (descend to 5% of apex or valley)
    threshold = apex_intensity * 0.05
    left_idx = best_idx
    while left_idx > 0 and smoothed[left_idx - 1] > threshold:
        if smoothed[left_idx - 1] > smoothed[left_idx]:
            break  # Hit another peak (valley)
        left_idx -= 1

    right_idx = best_idx
    while right_idx < n - 1 and smoothed[right_idx + 1] > threshold:
        if smoothed[right_idx + 1] > smoothed[right_idx]:
            break  # Hit another peak
        right_idx += 1

    width = right_idx - left_idx + 1

    # Step 6: Compute area (trapezoidal integration)
    if width > 1 and left_idx < len(trace.rt) and right_idx < len(trace.rt):
        area = float(np.trapz(
            smoothed[left_idx:right_idx + 1],
            trace.rt[left_idx:right_idx + 1],
        ))
    else:
        area = float(apex_intensity)

    # Step 7: Compute shape metrics
    asymmetry = _compute_asymmetry(smoothed, best_idx, left_idx, right_idx, trace.rt)
    jaggedness = _compute_jaggedness(trace.intensity, left_idx, right_idx, area)

    # Step 8: Quality filters
    if snr < config.min_peak_snr:
        return PeakResult(trace=trace, peak=None, smoothed=smoothed)
    if width < config.min_peak_width_scans:
        return PeakResult(trace=trace, peak=None, smoothed=smoothed)
    if apex_intensity < config.min_peak_intensity:
        return PeakResult(trace=trace, peak=None, smoothed=smoothed)

    peak = Peak(
        apex_idx=best_idx,
        apex_rt=float(trace.rt[best_idx]),
        apex_intensity=float(apex_intensity),
        left_idx=left_idx,
        right_idx=right_idx,
        area=area,
        width_scans=width,
        snr=snr,
        asymmetry=asymmetry,
        jaggedness=jaggedness,
    )

    return PeakResult(trace=trace, peak=peak, smoothed=smoothed)


def detect_peaks_multi(
    traces: list[XicTrace],
    config: DiaConfig,
) -> list[PeakResult]:
    """Detect peaks in multiple XIC traces."""
    return [detect_peak(trace, config) for trace in traces]


def _estimate_noise(smoothed: np.ndarray, apex_idx: int) -> float:
    """Estimate noise from baseline regions away from the apex.

    Uses the 10th percentile of the smoothed signal as noise estimate.
    """
    if len(smoothed) < 5:
        return max(float(np.min(smoothed)), 1.0)

    # Use percentile of full trace as noise floor
    noise = float(np.percentile(smoothed, 10))
    return max(noise, 1.0)  # Floor at 1.0 to avoid division by zero


def _compute_asymmetry(
    smoothed: np.ndarray,
    apex_idx: int,
    left_idx: int,
    right_idx: int,
    rt: np.ndarray,
) -> float:
    """Compute peak asymmetry.

    asymmetry = (right_width - left_width) / (right_width + left_width)
    Range: -1 (left-tailing) to +1 (right-tailing), 0 = symmetric.
    """
    if apex_idx <= left_idx or apex_idx >= right_idx:
        return 0.0

    left_width = rt[apex_idx] - rt[left_idx]
    right_width = rt[right_idx] - rt[apex_idx]
    total_width = left_width + right_width

    if total_width <= 0:
        return 0.0

    return float((right_width - left_width) / total_width)


def _compute_jaggedness(
    raw_intensity: np.ndarray,
    left_idx: int,
    right_idx: int,
    area: float,
) -> float:
    """Compute peak jaggedness (roughness).

    jaggedness = sum(|diff(intensity)|) / area
    Low values indicate smooth peaks; high values indicate noisy/jagged peaks.
    """
    if area <= 0 or right_idx <= left_idx:
        return 0.0

    segment = raw_intensity[left_idx:right_idx + 1]
    if len(segment) < 2:
        return 0.0

    total_variation = float(np.sum(np.abs(np.diff(segment))))
    return total_variation / area
