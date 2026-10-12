#!/usr/bin/env python
"""Train, refit, and audit a delta-CNN based on per-image correctness."""

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
import torch
from torch.nn import functional as F

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn.correctness_conv import (
    DEFAULT_CORRECTNESS_HEAD,
    SUPPORTED_CORRECTNESS_HEADS,
    CorrectnessConvRegressor,
)
from aaae.accuracy_predictor.cnn.delta_conv import (
    DEFAULT_MODEL_ARCHITECTURE,
    SUPPORTED_MODEL_ARCHITECTURES,
)
from aaae.accuracy_predictor.cnn.features import SUPPORTED_FEATURE_SCHEMAS
from aaae.accuracy_predictor.training.search_cnn_delta_conv import (
    _load_aligned_labels,
    _load_auxiliary_correctness,
    _load_feature_cache,
    _loader,
    _metrics,
    _normalise_features,
    _target_density_weights,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction(
    logits: torch.Tensor,
    *,
    exact_accuracy: float,
    logit_temperature: float = 1.0,
) -> torch.Tensor:
    return exact_accuracy - torch.sigmoid(logits.float() / logit_temperature).mean(
        dim=1
    )


def _loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    correctness: torch.Tensor,
    weight: torch.Tensor,
    *,
    exact_accuracy: float,
    bce_weight: float,
    tail_fraction: float,
    tail_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        correctness,
        reduction="none",
    ).mean(dim=1)
    predicted = _prediction(logits, exact_accuracy=exact_accuracy)
    squared = (predicted - target).square()
    tail_count = max(1, math.ceil(len(squared) * tail_fraction))
    tail = torch.topk(squared, k=tail_count, sorted=False).values.mean()
    return (
        bce_weight * torch.mean(weight * bce)
        + torch.mean(weight * squared)
        + tail_weight * tail
    )


@torch.no_grad()
def _predict(
    model: CorrectnessConvRegressor,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    exact_accuracy: float,
    inference_dtype: str,
    logit_temperature: float = 1.0,
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
            predicted = _prediction(
                model.logits(image, engineered),
                exact_accuracy=exact_accuracy,
                logit_temperature=logit_temperature,
            )
        predictions.append(predicted.cpu().numpy())
        actual.append(target_weight[:, 0].numpy())
    return np.concatenate(actual).astype(np.float64), np.concatenate(
        predictions
    ).astype(np.float64)


def _prepared_data(
    args: argparse.Namespace,
    *,
    split_name: str,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run = args.run_dir.expanduser().resolve()
    split, labels = _load_aligned_labels(
        run,
        args.labels.expanduser().resolve(),
        split_path=args.split.expanduser().resolve(),
    )
    mask = split["split"].eq(split_name).to_numpy()
    rows = np.flatnonzero(mask)
    features = _load_feature_cache(
        run,
        split_name,
        features_dir=args.features_dir.expanduser().resolve(),
    )
    if feature_mean is None or feature_std is None:
        feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        feature_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
        feature_std[feature_std < 1e-6] = 1.0
    features = _normalise_features(features, feature_mean, feature_std)
    target = labels.loc[mask, "top1_drop"].to_numpy(dtype=np.float32)
    correctness_bank = _load_auxiliary_correctness(
        args.correctness,
        expected_rows=len(split),
    )
    if correctness_bank is None:
        raise ValueError("--correctness is required")
    correctness = np.asarray(correctness_bank[rows], dtype=np.bool_)
    return rows, features, target, correctness, feature_mean, feature_std


def train(args: argparse.Namespace) -> dict[str, object]:
    (
        train_rows,
        train_features,
        train_target,
        train_correctness,
        feature_mean,
        feature_std,
    ) = _prepared_data(args, split_name="train")
    (
        validation_rows,
        validation_features,
        validation_target,
        validation_correctness,
        _,
        _,
    ) = _prepared_data(
        args,
        split_name="validation",
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    exact = np.load(args.exact_correctness, allow_pickle=False)
    if exact.shape != (1000,) or exact.dtype != np.bool_:
        raise ValueError("--exact-correctness must be a bool array with shape (1000,)")
    exact_accuracy = float(exact.mean())
    train_loader = _loader(
        delta_path=args.delta.expanduser().resolve(),
        rows=train_rows,
        features=train_features,
        target=train_target,
        sample_weight=_target_density_weights(train_target),
        correctness=train_correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(
        delta_path=args.delta.expanduser().resolve(),
        rows=validation_rows,
        features=validation_features,
        target=validation_target,
        sample_weight=np.ones(len(validation_target), dtype=np.float32),
        correctness=validation_correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    architecture = args.model_architecture or DEFAULT_MODEL_ARCHITECTURE
    device = torch.device("cuda")
    model = CorrectnessConvRegressor(
        train_features.shape[1],
        architecture=architecture,
        correctness_head=args.correctness_head,
    ).to(device=device, memory_format=torch.channels_last)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.03,
    )
    best_score = math.inf
    best_epoch = 0
    best_metrics: dict[str, float | int] = {}
    best_predictions = np.empty(0, dtype=np.float64)
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
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
                loss = _loss(
                    model.logits(image, engineered),
                    target_weight[:, 0],
                    correctness,
                    target_weight[:, 1],
                    exact_accuracy=exact_accuracy,
                    bce_weight=args.bce_weight,
                    tail_fraction=args.tail_fraction,
                    tail_weight=args.tail_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
            row_count += len(image)
        scheduler.step()
        actual, predicted = _predict(
            model,
            validation_loader,
            device=device,
            exact_accuracy=exact_accuracy,
            inference_dtype=args.inference_dtype,
            logit_temperature=args.logit_temperature,
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
            "schema_version": "cnn_correctness_conv_candidate_v1",
            "model_state": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_schema": args.feature_schema,
            "model_architecture": architecture,
            "correctness_head": args.correctness_head,
            "exact_accuracy": exact_accuracy,
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        output,
    )
    predictions_path = output.with_suffix(".validation_predictions.npy")
    np.save(predictions_path, best_predictions)
    result = {
        "schema_version": "cnn_correctness_conv_validation_search_v1",
        "protocol": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows_used": 0,
            "test_labels_used": False,
            "device": torch.cuda.get_device_name(0),
        },
        "configuration": {
            "seed": args.seed,
            "model_architecture": architecture,
            "correctness_head": args.correctness_head,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "bce_weight": args.bce_weight,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "inference_dtype": args.inference_dtype,
            "logit_temperature": args.logit_temperature,
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
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def predict(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    rows, features, target, correctness, _, _ = _prepared_data(
        args,
        split_name=args.predict_split,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    loader = _loader(
        delta_path=args.delta.expanduser().resolve(),
        rows=rows,
        features=features,
        target=target,
        sample_weight=np.ones(len(target), dtype=np.float32),
        correctness=correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    architecture = str(checkpoint["model_architecture"])
    correctness_head = str(checkpoint.get("correctness_head", DEFAULT_CORRECTNESS_HEAD))
    model = CorrectnessConvRegressor(
        features.shape[1],
        architecture=architecture,
        correctness_head=correctness_head,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    model = model.to(device=device, memory_format=torch.channels_last)
    actual, predicted = _predict(
        model,
        loader,
        device=device,
        exact_accuracy=float(checkpoint["exact_accuracy"]),
        inference_dtype=args.inference_dtype,
        logit_temperature=args.logit_temperature,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, predicted)
    result = {
        "schema_version": "cnn_correctness_conv_checkpoint_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": args.predict_split,
        "rows": len(rows),
        "test_labels_used": args.predict_split == "test",
        "logit_temperature": args.logit_temperature,
        "correctness_head": correctness_head,
        "metrics": _metrics(actual, predicted),
        "predictions": str(output),
        "predictions_sha256": _sha256(output),
        "device": torch.cuda.get_device_name(0),
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def refit(args: argparse.Namespace) -> dict[str, object]:
    run = args.run_dir.expanduser().resolve()
    split, labels = _load_aligned_labels(
        run,
        args.labels.expanduser().resolve(),
        split_path=args.split.expanduser().resolve(),
    )
    train_mask = split["split"].eq("train").to_numpy()
    validation_mask = split["split"].eq("validation").to_numpy()
    test_mask = split["split"].eq("test").to_numpy()
    development_rows = np.concatenate(
        (np.flatnonzero(train_mask), np.flatnonzero(validation_mask))
    )
    test_rows = np.flatnonzero(test_mask)

    features_dir = args.features_dir.expanduser().resolve()
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
    correctness_bank = _load_auxiliary_correctness(
        args.correctness,
        expected_rows=len(split),
    )
    if correctness_bank is None:
        raise ValueError("--correctness is required")
    development_correctness = np.asarray(
        correctness_bank[development_rows], dtype=np.bool_
    )
    test_correctness = np.asarray(correctness_bank[test_rows], dtype=np.bool_)

    exact = np.load(args.exact_correctness, allow_pickle=False)
    if exact.shape != (1000,) or exact.dtype != np.bool_:
        raise ValueError("--exact-correctness must be a bool array with shape (1000,)")
    exact_accuracy = float(exact.mean())
    development_loader = _loader(
        delta_path=args.delta.expanduser().resolve(),
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
        delta_path=args.delta.expanduser().resolve(),
        rows=test_rows,
        features=test_features,
        target=test_target,
        sample_weight=np.ones(len(test_target), dtype=np.float32),
        correctness=test_correctness,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )

    architecture = args.model_architecture or DEFAULT_MODEL_ARCHITECTURE
    device = torch.device("cuda")
    model = CorrectnessConvRegressor(
        development_features.shape[1],
        architecture=architecture,
        correctness_head=args.correctness_head,
    ).to(device=device, memory_format=torch.channels_last)
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.scheduler_epochs, 1),
        eta_min=args.learning_rate * 0.03,
    )
    started = time.time()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.refit_epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for image, engineered, target_weight, correctness in development_loader:
            image = image.to(device=device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            engineered = engineered.to(device=device, non_blocking=True)
            target_weight = target_weight.to(device=device, non_blocking=True)
            correctness = correctness.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = _loss(
                    model.logits(image, engineered),
                    target_weight[:, 0],
                    correctness,
                    target_weight[:, 1],
                    exact_accuracy=exact_accuracy,
                    bce_weight=args.bce_weight,
                    tail_fraction=args.tail_fraction,
                    tail_weight=args.tail_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(image)
            row_count += len(image)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / row_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    actual, predicted = _predict(
        model,
        test_loader,
        device=device,
        exact_accuracy=exact_accuracy,
        inference_dtype=args.inference_dtype,
        logit_temperature=args.logit_temperature,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "cnn_correctness_conv_refit_v1",
            "model_state": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_schema": args.feature_schema,
            "model_architecture": architecture,
            "correctness_head": args.correctness_head,
            "exact_accuracy": exact_accuracy,
            "refit_epochs": args.refit_epochs,
            "scheduler_epochs": args.scheduler_epochs,
            "seed": args.seed,
        },
        output,
    )
    predictions_path = output.with_suffix(".test_predictions.npy")
    np.save(predictions_path, predicted)
    result = {
        "schema_version": "cnn_correctness_conv_post_failure_refit_v1",
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
            "model_architecture": architecture,
            "correctness_head": args.correctness_head,
            "batch_size": args.batch_size,
            "refit_epochs": args.refit_epochs,
            "scheduler_epochs": args.scheduler_epochs,
            "learning_rate": args.learning_rate,
            "bce_weight": args.bce_weight,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "workers": args.workers,
            "inference_dtype": args.inference_dtype,
            "logit_temperature": args.logit_temperature,
        },
        "test_metrics": _metrics(actual, predicted),
        "history": history,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256(output),
        "test_predictions": str(predictions_path),
        "test_predictions_sha256": _sha256(predictions_path),
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("train", "predict", "refit"), default="train"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--exact-correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--feature-schema",
        choices=sorted(SUPPORTED_FEATURE_SCHEMAS),
        required=True,
    )
    parser.add_argument(
        "--model-architecture",
        choices=sorted(SUPPORTED_MODEL_ARCHITECTURES),
    )
    parser.add_argument(
        "--correctness-head",
        choices=sorted(SUPPORTED_CORRECTNESS_HEADS),
        default=DEFAULT_CORRECTNESS_HEAD,
    )
    parser.add_argument(
        "--predict-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--refit-epochs", type=int, default=27)
    parser.add_argument("--scheduler-epochs", type=int, default=80)
    parser.add_argument(
        "--selection-origin",
        default="validation_selected_epoch",
        help="Audit description of the validation-only rule that fixed refit epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bce-weight", type=float, default=0.001)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "predict" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required in predict mode")
    if (
        args.batch_size < 1
        or args.epochs < 1
        or args.patience < 1
        or args.refit_epochs < 1
        or args.scheduler_epochs < 1
        or args.workers < 0
    ):
        raise SystemExit(
            "batch size, epochs, patience, refit epochs, and scheduler epochs must "
            "be positive; workers must be nonnegative"
        )
    if args.learning_rate <= 0.0 or args.bce_weight < 0.0 or args.tail_weight < 0.0:
        raise SystemExit("learning rate must be positive and loss weights nonnegative")
    if not math.isfinite(args.logit_temperature) or args.logit_temperature <= 0.0:
        raise SystemExit("logit temperature must be finite and positive")
    if not 0.0 < args.tail_fraction <= 1.0:
        raise SystemExit("tail fraction must be in (0, 1]")
    if not args.selection_origin.strip():
        raise SystemExit("selection origin must be nonempty")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.mode == "train":
        result = train(args)
    elif args.mode == "predict":
        result = predict(args)
    else:
        result = refit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
