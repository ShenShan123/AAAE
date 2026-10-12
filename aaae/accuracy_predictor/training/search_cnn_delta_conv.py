#!/usr/bin/env python
"""Train, refit, and audit a profile-bound hybrid delta-CNN predictor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn.delta_conv import (
    DEFAULT_MODEL_ARCHITECTURE,
    INPUT_LOG_SCALE,
    MAGNITUDE_ORDER,
    SUPPORTED_MODEL_ARCHITECTURES,
    create_delta_conv_regressor,
)
from aaae.accuracy_predictor.cnn.features import SUPPORTED_FEATURE_SCHEMAS

LOSS_MODES = frozenset({"huber", "mse", "tail_mse"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeltaDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        delta_path: Path,
        row_indices: np.ndarray,
        engineered: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        correctness: np.ndarray | None = None,
    ) -> None:
        self.delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.engineered = np.asarray(engineered, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        self.sample_weight = np.asarray(sample_weight, dtype=np.float32)
        self.correctness = (
            None if correctness is None else np.asarray(correctness, dtype=np.bool_)
        )
        if self.correctness is not None and self.correctness.shape[0] != len(
            self.row_indices
        ):
            raise ValueError("correctness row count must match delta dataset rows")

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = np.asarray(self.delta[int(self.row_indices[index])])
        quadrant = np.asarray(
            raw[np.ix_(MAGNITUDE_ORDER, MAGNITUDE_ORDER)], dtype=np.float32
        )
        magnitude = np.log1p(np.abs(quadrant)) / INPUT_LOG_SCALE
        signed = np.sign(quadrant) * magnitude
        image = np.stack((signed, magnitude), axis=0).astype(np.float32, copy=False)
        correctness = (
            torch.empty(0, dtype=torch.float32)
            if self.correctness is None
            else torch.from_numpy(np.asarray(self.correctness[index], dtype=np.float32))
        )
        return (
            torch.from_numpy(image),
            torch.from_numpy(self.engineered[index]),
            torch.as_tensor(
                (self.target[index], self.sample_weight[index]),
                dtype=torch.float32,
            ),
            correctness,
        )


def _target_density_weights(target: np.ndarray) -> np.ndarray:
    edges = np.linspace(float(target.min()), float(target.max()) + 1e-12, 41)
    assignments = np.clip(np.searchsorted(edges, target, side="right") - 1, 0, 39)
    counts = np.bincount(assignments, minlength=40)
    populated = counts[counts > 0]
    raw = np.sqrt(float(np.median(populated)) / counts[assignments])
    clipped = np.clip(raw, 0.5, 6.0)
    return np.asarray(clipped / clipped.mean(), dtype=np.float32)


def _regression_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    mode: str,
    tail_fraction: float,
    tail_weight: float,
) -> torch.Tensor:
    error = predicted - target
    squared = error.square()
    if mode == "huber":
        huber = F.smooth_l1_loss(predicted, target, beta=0.01, reduction="none")
        return torch.mean(weight * (huber + 0.20 * squared))
    if mode == "mse":
        return torch.mean(weight * squared)
    if mode != "tail_mse":
        raise ValueError(f"unsupported regression loss mode: {mode!r}")
    tail_count = max(1, math.ceil(len(squared) * tail_fraction))
    tail = torch.topk(squared, k=tail_count, sorted=False).values.mean()
    return torch.mean(weight * squared) + tail_weight * tail


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(np.asarray(predicted, dtype=np.float64) - actual)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "over_5pp_count": int(np.sum(absolute > 0.05)),
    }


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    inference_dtype: str = "bfloat16",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for image, engineered, target_weight, _ in loader:
        image = image.to(device=device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        engineered = engineered.to(device=device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=inference_dtype == "bfloat16",
        ):
            predicted = model(image, engineered)
        predictions.append(predicted.float().cpu().numpy())
        actual.append(target_weight[:, 0].numpy())
    return np.concatenate(actual).astype(np.float64), np.concatenate(
        predictions
    ).astype(np.float64)


def _load_feature_cache(
    run: Path,
    split: str,
    *,
    features_dir: Path | None,
) -> np.ndarray:
    if features_dir is not None:
        path = features_dir / f"{split}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2:
            raise ValueError(f"feature cache must be two-dimensional: {path}")
        return np.asarray(values, dtype=np.float32)
    if split == "test":
        path = run / "_post_failure_test_stage_features.npy"
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2:
            raise ValueError(f"test feature cache must be two-dimensional: {path}")
        return np.asarray(values, dtype=np.float32)
    ame = np.load(run / f"_search_32_{split}.npy", mmap_mode="r")
    stage = np.load(run / f"_search_stage_role_{split}.npy", mmap_mode="r")
    return np.concatenate((ame, stage), axis=1).astype(np.float32)


def _load_aligned_labels(
    run: Path,
    labels_path: Path,
    *,
    split_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_split = run / "split.csv" if split_path is None else split_path
    split = pd.read_csv(resolved_split, dtype={"design_id": str})
    labels = pd.read_csv(labels_path, dtype={"design_id": str})
    labels = split[["design_id"]].merge(
        labels[["design_id", "top1_drop"]],
        on="design_id",
        validate="one_to_one",
    )
    return split, labels


def _normalise_features(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.clip(
        (np.asarray(values, dtype=np.float32) - mean) / std,
        -12.0,
        12.0,
    )


def _loader(
    *,
    delta_path: Path,
    rows: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    correctness: np.ndarray | None = None,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "drop_last": False,
    }
    return DataLoader(
        DeltaDataset(
            delta_path,
            rows,
            features,
            target,
            sample_weight,
            correctness,
        ),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        **options,
    )


def _load_auxiliary_correctness(
    path: Path | None,
    *,
    expected_rows: int,
) -> np.ndarray | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    values = np.load(resolved, mmap_mode="r", allow_pickle=False)
    if values.shape != (expected_rows, 1000) or values.dtype != np.bool_:
        raise ValueError(
            "auxiliary correctness must be a bool array with shape "
            f"({expected_rows}, 1000): {resolved}"
        )
    return values


def _auxiliary_head(encoded_features: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(encoded_features, 512),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(512, 1000),
    )


def _multitask_loss(
    logits: torch.Tensor,
    correctness: torch.Tensor,
    *,
    aggregate_weight: float,
) -> torch.Tensor:
    classification = F.binary_cross_entropy_with_logits(logits, correctness)
    predicted_accuracy = torch.sigmoid(logits.float()).mean(dim=1)
    actual_accuracy = correctness.float().mean(dim=1)
    aggregate = F.mse_loss(predicted_accuracy, actual_accuracy)
    return classification + aggregate_weight * aggregate


def train(args: argparse.Namespace) -> dict[str, object]:
    run = args.run_dir.expanduser().resolve()
    features_dir = (
        None if args.features_dir is None else args.features_dir.expanduser().resolve()
    )
    delta_path = args.delta.expanduser().resolve()
    split, labels = _load_aligned_labels(
        run,
        args.labels.expanduser().resolve(),
        split_path=None if args.split is None else args.split.expanduser().resolve(),
    )
    train_mask = split["split"].eq("train").to_numpy()
    validation_mask = split["split"].eq("validation").to_numpy()
    train_rows = np.flatnonzero(train_mask)
    validation_rows = np.flatnonzero(validation_mask)
    train_features = _load_feature_cache(
        run,
        "train",
        features_dir=features_dir,
    )
    validation_features = _load_feature_cache(
        run,
        "validation",
        features_dir=features_dir,
    )
    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    train_features = _normalise_features(train_features, feature_mean, feature_std)
    validation_features = _normalise_features(
        validation_features,
        feature_mean,
        feature_std,
    )
    train_target = labels.loc[train_mask, "top1_drop"].to_numpy(dtype=np.float32)
    validation_target = labels.loc[validation_mask, "top1_drop"].to_numpy(
        dtype=np.float32
    )
    train_weight = _target_density_weights(train_target)
    validation_weight = np.ones(len(validation_target), dtype=np.float32)
    auxiliary_correctness = _load_auxiliary_correctness(
        args.aux_correctness,
        expected_rows=len(split),
    )
    train_correctness = (
        None
        if auxiliary_correctness is None
        else np.asarray(auxiliary_correctness[train_rows], dtype=np.bool_)
    )

    train_loader = _loader(
        delta_path=delta_path,
        rows=train_rows,
        features=train_features,
        target=train_target,
        sample_weight=train_weight,
        correctness=train_correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(
        delta_path=delta_path,
        rows=validation_rows,
        features=validation_features,
        target=validation_target,
        sample_weight=validation_weight,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )

    device = torch.device("cuda")
    model_architecture = args.model_architecture or DEFAULT_MODEL_ARCHITECTURE
    model = create_delta_conv_regressor(
        train_features.shape[1],
        architecture=model_architecture,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)
    auxiliary_head = (
        None
        if train_correctness is None
        else _auxiliary_head(model.ENCODED_FEATURES).to(device)
    )
    parameters = list(model.parameters())
    if auxiliary_head is not None:
        parameters.extend(auxiliary_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.03,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_score = math.inf
    best_epoch = 0
    best_metrics: dict[str, float | int] = {}
    best_predictions = np.empty(0, dtype=np.float64)
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        loss_sum = 0.0
        row_count = 0
        for image, engineered, target_weight, correctness in train_loader:
            image = image.to(device=device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            engineered = engineered.to(device=device, non_blocking=True)
            target_weight = target_weight.to(device=device, non_blocking=True)
            correctness = correctness.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = model.encode(image, engineered)
                predicted = model.head(encoded).squeeze(1)
                target = target_weight[:, 0]
                weight = target_weight[:, 1]
                loss = _regression_loss(
                    predicted,
                    target,
                    weight,
                    mode=args.loss_mode,
                    tail_fraction=args.tail_fraction,
                    tail_weight=args.tail_weight,
                )
                if auxiliary_head is not None:
                    logits = auxiliary_head(encoded)
                    loss = loss + args.aux_weight * _multitask_loss(
                        logits,
                        correctness,
                        aggregate_weight=args.aux_aggregate_weight,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(image)
            row_count += len(image)
        scheduler.step()
        actual, predicted = _predict(
            model,
            validation_loader,
            device=device,
            inference_dtype=args.inference_dtype,
        )
        metrics = _metrics(actual, predicted)
        score = (
            float(metrics["max_abs_error"])
            + 0.20 * float(metrics["p99_abs_error"])
            + 0.10 * float(metrics["mae"])
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / row_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selection_score": score,
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_predictions = predicted.copy()
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch >= 12 and epoch - best_epoch >= args.patience:
            break

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "cnn_delta_conv_candidate_v1",
            "model_state": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_schema": args.feature_schema,
            "magnitude_order": MAGNITUDE_ORDER,
            "input_log_scale": INPUT_LOG_SCALE,
            "best_epoch": best_epoch,
            "seed": args.seed,
            "model_architecture": model_architecture,
        },
        output,
    )
    predictions_path = output.with_suffix(".validation_predictions.npy")
    np.save(predictions_path, best_predictions)
    result = {
        "schema_version": "cnn_delta_conv_validation_search_v1",
        "protocol": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows_used": 0,
            "test_labels_used": False,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "input": "lossless_sign_magnitude_129x129_signed_and_absolute_log1p",
            "engineered_feature_schema": args.feature_schema,
        },
        "configuration": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "workers": args.workers,
            "inference_dtype": args.inference_dtype,
            "model_architecture": model_architecture,
            "loss_mode": args.loss_mode,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "aux_correctness": (
                None
                if args.aux_correctness is None
                else str(args.aux_correctness.expanduser().resolve())
            ),
            "aux_weight": args.aux_weight,
            "aux_aggregate_weight": args.aux_aggregate_weight,
        },
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "validation_predictions": str(predictions_path),
        "validation_predictions_sha256": _sha256(predictions_path),
    }
    result_path = output.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def predict_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    run = args.run_dir.expanduser().resolve()
    features_dir = (
        None if args.features_dir is None else args.features_dir.expanduser().resolve()
    )
    delta_path = args.delta.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    split, labels = _load_aligned_labels(
        run,
        args.labels.expanduser().resolve(),
        split_path=None if args.split is None else args.split.expanduser().resolve(),
    )
    mask = split["split"].eq(args.predict_split).to_numpy()
    rows = np.flatnonzero(mask)
    if not len(rows):
        raise ValueError(f"split {args.predict_split!r} contains no rows")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_schema") != args.feature_schema:
        raise ValueError(
            "checkpoint feature schema differs from --feature-schema: "
            f"{checkpoint.get('feature_schema')!r} != {args.feature_schema!r}"
        )
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = _normalise_features(
        _load_feature_cache(
            run,
            args.predict_split,
            features_dir=features_dir,
        ),
        feature_mean,
        feature_std,
    )
    target = labels.loc[mask, "top1_drop"].to_numpy(dtype=np.float32)
    loader = _loader(
        delta_path=delta_path,
        rows=rows,
        features=features,
        target=target,
        sample_weight=np.ones(len(rows), dtype=np.float32),
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    checkpoint_architecture = checkpoint.get(
        "model_architecture", DEFAULT_MODEL_ARCHITECTURE
    )
    if checkpoint_architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        raise ValueError(
            "checkpoint contains an unsupported model architecture: "
            f"{checkpoint_architecture!r}"
        )
    if (
        args.model_architecture is not None
        and args.model_architecture != checkpoint_architecture
    ):
        raise ValueError(
            "checkpoint model architecture differs from --model-architecture: "
            f"{checkpoint_architecture!r} != {args.model_architecture!r}"
        )
    model = create_delta_conv_regressor(
        features.shape[1],
        architecture=str(checkpoint_architecture),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda")
    model = model.to(device=device, memory_format=torch.channels_last)
    actual, predicted = _predict(
        model,
        loader,
        device=device,
        inference_dtype=args.inference_dtype,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, predicted)
    result = {
        "schema_version": "cnn_delta_conv_checkpoint_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": args.predict_split,
        "rows": len(rows),
        "test_labels_used": args.predict_split == "test",
        "inference_dtype": args.inference_dtype,
        "model_architecture": str(checkpoint_architecture),
        "metrics": _metrics(actual, predicted),
        "predictions": str(output),
        "predictions_sha256": _sha256(output),
        "device": torch.cuda.get_device_name(0),
    }
    result_path = output.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def refit(args: argparse.Namespace) -> dict[str, object]:
    run = args.run_dir.expanduser().resolve()
    features_dir = (
        None if args.features_dir is None else args.features_dir.expanduser().resolve()
    )
    delta_path = args.delta.expanduser().resolve()
    split, labels = _load_aligned_labels(
        run,
        args.labels.expanduser().resolve(),
        split_path=None if args.split is None else args.split.expanduser().resolve(),
    )
    train_mask = split["split"].eq("train").to_numpy()
    validation_mask = split["split"].eq("validation").to_numpy()
    test_mask = split["split"].eq("test").to_numpy()
    development_rows = np.concatenate(
        (np.flatnonzero(train_mask), np.flatnonzero(validation_mask))
    )
    test_rows = np.flatnonzero(test_mask)

    development_features = np.concatenate(
        (
            _load_feature_cache(run, "train", features_dir=features_dir),
            _load_feature_cache(run, "validation", features_dir=features_dir),
        ),
        axis=0,
    )
    feature_mean = development_features.mean(axis=0, dtype=np.float64).astype(
        np.float32
    )
    feature_std = development_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    development_features = _normalise_features(
        development_features,
        feature_mean,
        feature_std,
    )
    test_features = _normalise_features(
        _load_feature_cache(run, "test", features_dir=features_dir),
        feature_mean,
        feature_std,
    )
    development_target = np.concatenate(
        (
            labels.loc[train_mask, "top1_drop"].to_numpy(dtype=np.float32),
            labels.loc[validation_mask, "top1_drop"].to_numpy(dtype=np.float32),
        )
    )
    test_target = labels.loc[test_mask, "top1_drop"].to_numpy(dtype=np.float32)
    auxiliary_correctness = _load_auxiliary_correctness(
        args.aux_correctness,
        expected_rows=len(split),
    )
    development_correctness = (
        None
        if auxiliary_correctness is None
        else np.asarray(auxiliary_correctness[development_rows], dtype=np.bool_)
    )
    train_loader = _loader(
        delta_path=delta_path,
        rows=development_rows,
        features=development_features,
        target=development_target,
        sample_weight=_target_density_weights(development_target),
        correctness=development_correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    test_loader = _loader(
        delta_path=delta_path,
        rows=test_rows,
        features=test_features,
        target=test_target,
        sample_weight=np.ones(len(test_rows), dtype=np.float32),
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )

    device = torch.device("cuda")
    model_architecture = args.model_architecture or DEFAULT_MODEL_ARCHITECTURE
    model = create_delta_conv_regressor(
        development_features.shape[1],
        architecture=model_architecture,
    ).to(
        device=device,
        memory_format=torch.channels_last,
    )
    auxiliary_head = (
        None
        if development_correctness is None
        else _auxiliary_head(model.ENCODED_FEATURES).to(device)
    )
    parameters = list(model.parameters())
    if auxiliary_head is not None:
        parameters.extend(auxiliary_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.scheduler_epochs, 1),
        eta_min=args.learning_rate * 0.03,
    )
    started = time.time()
    for epoch in range(1, args.refit_epochs + 1):
        model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        loss_sum = 0.0
        row_count = 0
        for image, engineered, target_weight, correctness in train_loader:
            image = image.to(device=device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            engineered = engineered.to(device=device, non_blocking=True)
            target_weight = target_weight.to(device=device, non_blocking=True)
            correctness = correctness.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = model.encode(image, engineered)
                predicted = model.head(encoded).squeeze(1)
                target = target_weight[:, 0]
                weight = target_weight[:, 1]
                loss = _regression_loss(
                    predicted,
                    target,
                    weight,
                    mode=args.loss_mode,
                    tail_fraction=args.tail_fraction,
                    tail_weight=args.tail_weight,
                )
                if auxiliary_head is not None:
                    logits = auxiliary_head(encoded)
                    loss = loss + args.aux_weight * _multitask_loss(
                        logits,
                        correctness,
                        aggregate_weight=args.aux_aggregate_weight,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
            row_count += len(image)
        scheduler.step()
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / row_count,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    actual, predicted = _predict(
        model,
        test_loader,
        device=device,
        inference_dtype=args.inference_dtype,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "cnn_delta_conv_refit_v1",
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_schema": args.feature_schema,
        "magnitude_order": MAGNITUDE_ORDER,
        "input_log_scale": INPUT_LOG_SCALE,
        "refit_epochs": args.refit_epochs,
        "scheduler_epochs": args.scheduler_epochs,
        "seed": args.seed,
        "model_architecture": model_architecture,
    }
    torch.save(checkpoint, output)
    predictions_path = output.with_suffix(".test_predictions.npy")
    np.save(predictions_path, predicted)
    result = {
        "schema_version": "cnn_delta_conv_post_failure_refit_v1",
        "protocol": {
            "development_rows": len(development_rows),
            "test_rows": len(test_rows),
            "test_labels_used_for_reporting": True,
            "test_labels_used_for_training": False,
            "selection_origin": args.selection_origin,
            "device": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "refit_epochs": args.refit_epochs,
            "scheduler_epochs": args.scheduler_epochs,
            "learning_rate": args.learning_rate,
            "workers": args.workers,
            "inference_dtype": args.inference_dtype,
            "model_architecture": model_architecture,
            "loss_mode": args.loss_mode,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "aux_correctness": (
                None
                if args.aux_correctness is None
                else str(args.aux_correctness.expanduser().resolve())
            ),
            "aux_weight": args.aux_weight,
            "aux_aggregate_weight": args.aux_aggregate_weight,
        },
        "test_metrics": _metrics(actual, predicted),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "test_predictions": str(predictions_path),
        "test_predictions_sha256": _sha256(predictions_path),
    }
    result_path = output.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("train", "predict", "refit"), default="train"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        help="Explicit split CSV; defaults to RUN_DIR/split.csv.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--features-dir",
        type=Path,
        help="Directory containing train.npy, validation.npy, and test.npy caches.",
    )
    parser.add_argument(
        "--feature-schema",
        choices=sorted(SUPPORTED_FEATURE_SCHEMAS),
        default="cnn_ame_basis32_stage_v3",
    )
    parser.add_argument(
        "--predict-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--refit-epochs", type=int, default=27)
    parser.add_argument("--scheduler-epochs", type=int, default=50)
    parser.add_argument(
        "--selection-origin",
        default="validation_selected_epoch",
        help="Audit description of the validation-only rule that fixed refit epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--model-architecture",
        choices=sorted(SUPPORTED_MODEL_ARCHITECTURES),
        help=(
            "Delta-CNN architecture. Training defaults to pooled_v1; prediction "
            "defaults to the architecture recorded in the checkpoint."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        choices=sorted(LOSS_MODES),
        default="huber",
        help="Regression objective used for candidate training and refit.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.10,
        help="Per-batch largest-error fraction used by tail_mse.",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=1.0,
        help="Additional top-k MSE weight used by tail_mse.",
    )
    parser.add_argument(
        "--aux-correctness",
        type=Path,
        help="Optional aligned [N,1000] bool top-1 correctness array for training only.",
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.0,
        help="Weight of per-image auxiliary supervision; requires --aux-correctness.",
    )
    parser.add_argument(
        "--aux-aggregate-weight",
        type=float,
        default=50.0,
        help="Aggregate-accuracy MSE weight inside the auxiliary objective.",
    )
    parser.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.batch_size < 1
        or args.epochs < 1
        or args.patience < 1
        or args.refit_epochs < 1
        or args.scheduler_epochs < 1
        or args.workers < 0
    ):
        raise SystemExit(
            "batch size, epochs, and patience must be positive; workers nonnegative"
        )
    if args.learning_rate <= 0.0:
        raise SystemExit("learning rate must be positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise SystemExit("tail fraction must be in (0, 1]")
    if args.tail_weight < 0.0:
        raise SystemExit("tail weight must be nonnegative")
    if args.aux_weight < 0.0 or args.aux_aggregate_weight < 0.0:
        raise SystemExit("auxiliary weights must be nonnegative")
    if args.aux_weight > 0.0 and args.aux_correctness is None:
        raise SystemExit("--aux-correctness is required when --aux-weight is positive")
    if not args.selection_origin.strip():
        raise SystemExit("selection origin must be nonempty")
    if args.mode == "predict" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required in predict mode")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.mode == "train":
        result = train(args)
    elif args.mode == "predict":
        result = predict_checkpoint(args)
    else:
        result = refit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
