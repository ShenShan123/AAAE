"""Formal structure-aware Transformer feature generation tools."""

from aaae.accuracy_predictor.features.transformer.swin_linear48_candidate_features import (
    FEATURE_COUNT as SWIN_LINEAR48_CAUSAL_PREFIX_FEATURE_COUNT,
)
from aaae.accuracy_predictor.features.transformer.swin_linear48_candidate_features import (
    SwinLinear48CandidateFeatureEvaluator,
    build_swin_linear48_candidate_features_from_signed_residuals,
)

__all__ = [
    "SWIN_LINEAR48_CAUSAL_PREFIX_FEATURE_COUNT",
    "SwinLinear48CandidateFeatureEvaluator",
    "build_swin_linear48_candidate_features_from_signed_residuals",
]
