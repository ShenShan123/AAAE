from __future__ import annotations


import json

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import numpy as np

METRIC_COLUMNS = [
    "muE",
    "sigmaE",
    "muED",
    "NMED",
    "sigmaED",
    "muRE",
    "sigmaRE",
    "muRED",
    "sigmaRED",
    "rmsE",
    "rmsRE",
    "ER",
    "changed_rows",
    "changed_frac",
]


DMULT_TARGET_COLUMNS = tuple(METRIC_COLUMNS[:12])


@dataclass(frozen=True)
class ArrayStandardScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        shape = (1,) * (arr.ndim - 1) + (int(self.mean.shape[0]),)
        return (arr - self.mean.reshape(shape)) / self.std.reshape(shape)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        shape = (1,) * (arr.ndim - 1) + (int(self.mean.shape[0]),)
        return (arr * self.std.reshape(shape)) + self.mean.reshape(shape)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(x) for x in self.mean.tolist()],
            "std": [float(x) for x in self.std.tolist()],
        }

    @staticmethod
    def from_json_dict(payload: dict[str, Any]) -> "ArrayStandardScaler":
        return ArrayStandardScaler(
            feature_names=tuple(str(item) for item in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def fit_standard_scaler(values: np.ndarray, train_indices: np.ndarray, feature_names: tuple[str, ...]) -> ArrayStandardScaler:
    arr = np.asarray(values, dtype=np.float32)
    train = arr[np.asarray(train_indices, dtype=np.int64)]
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return ArrayStandardScaler(feature_names=tuple(feature_names), mean=mean, std=std)


def fit_lastdim_scaler(values: np.ndarray, train_indices: np.ndarray, feature_names: tuple[str, ...]) -> ArrayStandardScaler:
    arr = np.asarray(values, dtype=np.float32)
    train = arr[np.asarray(train_indices, dtype=np.int64)]
    flat = train.reshape(-1, int(train.shape[-1]))
    mean = np.mean(flat, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(flat, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return ArrayStandardScaler(feature_names=tuple(feature_names), mean=mean, std=std)


def load_scaler(path: Path) -> ArrayStandardScaler:
    return ArrayStandardScaler.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


