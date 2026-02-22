"""Discriminant scoring — combine RSM features into a single score.

v1: Linear combination of features with configurable weights.
The agent can evolve this to polynomial, learned, or other scoring functions.
"""

from __future__ import annotations

from .config import DiaConfig
from .features import RsmFeatures


def compute_score(features: RsmFeatures, config: DiaConfig) -> float:
    """Compute discriminant score from RSM features.

    v1: Linear combination using config.feature_weights.

    Args:
        features: RSM feature vector.
        config: Pipeline configuration with feature_weights.

    Returns:
        Float discriminant score (higher = more confident).
    """
    feat_dict = features.to_dict()
    weights = config.feature_weights

    score = 0.0
    for feature_name, weight in weights.items():
        if feature_name in feat_dict:
            score += weight * feat_dict[feature_name]

    return score
