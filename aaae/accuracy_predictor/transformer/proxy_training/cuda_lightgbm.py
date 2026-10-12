"""Serializable CUDA LightGBM estimator used by strict-50 packages."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor


class CudaNamedFeatureLGBMRegressor(LGBMRegressor):
    """Give dense CUDA matrices stable unique names before LightGBM sees them."""

    def fit(
        self,
        x: Any,
        y: Any,
        *args: Any,
        feature_name: Any = "auto",
        **kwargs: Any,
    ) -> "CudaNamedFeatureLGBMRegressor":
        if isinstance(feature_name, str) and feature_name == "auto":
            shape = getattr(x, "shape", None)
            if shape is not None and len(shape) == 2:
                feature_name = [
                    f"feature_{index:04d}" for index in range(int(shape[1]))
                ]
        return super().fit(
            x,
            y,
            *args,
            feature_name=feature_name,
            **kwargs,
        )
