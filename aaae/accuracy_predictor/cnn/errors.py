from __future__ import annotations


class CnnPredictorError(Exception):
    """Base class for stable CNN workflow failures."""


class ManifestError(CnnPredictorError, ValueError):
    """Raised when a workflow manifest is malformed or incomplete."""


class ExecutionUnavailableError(CnnPredictorError, RuntimeError):
    """Raised when an explicitly requested training device cannot be used."""


class TrainingError(CnnPredictorError, RuntimeError):
    """Raised when LightGBM training fails after inputs have been validated."""


class ArtifactError(CnnPredictorError, RuntimeError):
    """Raised when a bundle is incomplete, corrupt, or incompatible."""


class UnqualifiedModelError(ArtifactError):
    """Raised when a synthetic-only bundle is opened without explicit consent."""

