from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ArithmeticDomain = Literal["u8_numeric", "s8_numeric", "s8_twos_complement"]
EvidenceKind = Literal["measured", "synthetic"]


class DataContractError(ValueError):
    """Raised when CNN predictor input violates its data contract."""


class DomainMismatchError(DataContractError):
    """Raised when a profile and design batch use different domains."""


@dataclass(frozen=True)
class ApplicationProfile:
    network_id: str
    dataset_id: str
    domain: ArithmeticDomain
    layer_ids: tuple[str, ...]
    activation_histograms: NDArray[np.number]
    weight_histograms: NDArray[np.number]
    mac_counts: NDArray[np.number]
    propagation_gains: NDArray[np.number]
    profile_evidence: EvidenceKind


@dataclass(frozen=True)
class MultiplierDesignBatch:
    design_ids: tuple[str, ...]
    delta: NDArray[np.number]
    domain: ArithmeticDomain


@dataclass(frozen=True)
class FeatureTable:
    design_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: NDArray[np.float32]
    schema_name: str
    profile_digest: str

