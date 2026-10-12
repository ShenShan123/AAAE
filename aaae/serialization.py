"""Read historical model packages while keeping their Python code local."""

from __future__ import annotations

import importlib
import sys
import types


def load_joblib(path):
    """Resolve the old custom LightGBM pickle name to the relocated class."""
    import joblib

    old_name = "transformer.proxy_training.cuda_lightgbm"
    new_module = importlib.import_module("aaae.accuracy_predictor.transformer.proxy_training.cuda_lightgbm")
    for name in ("transformer", "transformer.proxy_training"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    sys.modules[old_name] = new_module
    sys.modules["transformer"].proxy_training = sys.modules["transformer.proxy_training"]
    sys.modules["transformer.proxy_training"].cuda_lightgbm = new_module
    return joblib.load(path)
