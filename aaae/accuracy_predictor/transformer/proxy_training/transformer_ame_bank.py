from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from aaae.paths import LEGACY_WORKFLOW_ROOT

PROJECT_ROOT = LEGACY_WORKFLOW_ROOT
DEFAULT_D_NODE_META_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "archive"
    / "dataset_consolidation_20260508"
    / "D_node"
    / "meta"
)
DEFAULT_CALIB_SUBSET_PATH = DEFAULT_D_NODE_META_DIR / "calib_subset_500.json"
DEFAULT_EXACT_BASELINE_PATH = DEFAULT_D_NODE_META_DIR / "exact_baseline_500.npz"


@dataclass(frozen=True)
class TransformerAMEBankMeta:
    clean_margin_top1: np.ndarray
    clean_margin_top5: np.ndarray
    exact_top1_correct_bits: np.ndarray
    exact_top5_correct_bits: np.ndarray
    clean_entropy: np.ndarray
    exact_pred_class: np.ndarray
    true_label: np.ndarray
    bank_mean_top1: np.ndarray
    bank_mean_top5: np.ndarray
    bank_basis_top1: np.ndarray
    bank_basis_top5: np.ndarray
    coeff_mean_top1: np.ndarray
    coeff_std_top1: np.ndarray
    coeff_mean_top5: np.ndarray
    coeff_std_top5: np.ndarray
    basis_ready: bool

    @property
    def bank_dim(self) -> int:
        return int(self.bank_basis_top1.shape[0] + self.bank_basis_top5.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.clean_margin_top1.shape[0])


def _require_2d_logits(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected logits with shape [N, C], got {arr.shape}.")
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    arr = _require_2d_logits(logits).astype(np.float64, copy=False)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp, axis=1, keepdims=True), dtype=np.float32)


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    arr = _require_2d_logits(logits)
    if int(k) <= 0 or int(k) > int(arr.shape[1]):
        raise ValueError(f"k must satisfy 1 <= k <= class_count, got k={k}, class_count={arr.shape[1]}.")
    part = np.argpartition(arr, kth=int(arr.shape[1] - int(k)), axis=1)[:, -int(k) :]
    part_logits = np.take_along_axis(arr, part, axis=1)
    order = np.argsort(part_logits, axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1).astype(np.int64, copy=False)


def compute_entropy(logits: np.ndarray) -> np.ndarray:
    probs = softmax(logits).astype(np.float64, copy=False)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0)), axis=1)
    return np.asarray(entropy, dtype=np.float32)


def margin_top1_against_exact_class(logits: np.ndarray, exact_pred_class: np.ndarray) -> np.ndarray:
    arr = _require_2d_logits(logits).astype(np.float64, copy=False)
    exact_cls = np.asarray(exact_pred_class, dtype=np.int64).reshape(-1)
    if int(arr.shape[0]) != int(exact_cls.shape[0]):
        raise ValueError("logits row count must match exact_pred_class length.")
    row_idx = np.arange(arr.shape[0], dtype=np.int64)
    exact_logit = arr[row_idx, exact_cls]
    masked = arr.copy()
    masked[row_idx, exact_cls] = -np.inf
    runner_up = np.max(masked, axis=1)
    return np.asarray(exact_logit - runner_up, dtype=np.float32)


def margin_top5_for_true_label(logits: np.ndarray, true_label: np.ndarray) -> np.ndarray:
    arr = _require_2d_logits(logits).astype(np.float64, copy=False)
    labels = np.asarray(true_label, dtype=np.int64).reshape(-1)
    if int(arr.shape[0]) != int(labels.shape[0]):
        raise ValueError("logits row count must match true_label length.")
    row_idx = np.arange(arr.shape[0], dtype=np.int64)
    label_logit = arr[row_idx, labels]
    masked = arr.copy()
    masked[row_idx, labels] = -np.inf
    kth = int(arr.shape[1] - 5)
    fifth_nonlabel = np.partition(masked, kth=kth, axis=1)[:, kth]
    return np.asarray(label_logit - fifth_nonlabel, dtype=np.float32)


def compute_clean_bank(exact_logits: np.ndarray, true_label: np.ndarray) -> dict[str, np.ndarray]:
    logits = _require_2d_logits(exact_logits)
    labels = np.asarray(true_label, dtype=np.int64).reshape(-1)
    if int(logits.shape[0]) != int(labels.shape[0]):
        raise ValueError("exact_logits row count must match true_label length.")
    exact_pred_class = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    exact_top5 = topk_indices(logits, 5)
    exact_top1_correct_bits = np.asarray(exact_pred_class == labels, dtype=np.uint8)
    exact_top5_correct_bits = np.asarray(np.any(exact_top5 == labels[:, None], axis=1), dtype=np.uint8)
    return {
        "clean_margin_top1": margin_top1_against_exact_class(logits, exact_pred_class),
        "clean_margin_top5": margin_top5_for_true_label(logits, labels),
        "exact_top1_correct_bits": exact_top1_correct_bits,
        "exact_top5_correct_bits": exact_top5_correct_bits,
        "clean_entropy": compute_entropy(logits),
        "exact_pred_class": exact_pred_class,
        "true_label": labels.astype(np.int64, copy=False),
    }


def compute_runtime_bank_from_logits(
    approx_logits: np.ndarray,
    *,
    clean_margin_top1: np.ndarray,
    clean_margin_top5: np.ndarray,
    exact_pred_class: np.ndarray,
    true_label: np.ndarray,
    exact_top1_correct_bits: np.ndarray,
    exact_top5_correct_bits: np.ndarray,
) -> dict[str, np.ndarray | float]:
    logits = _require_2d_logits(approx_logits)
    labels = np.asarray(true_label, dtype=np.int64).reshape(-1)
    exact_cls = np.asarray(exact_pred_class, dtype=np.int64).reshape(-1)
    approx_top1 = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    approx_top5 = topk_indices(logits, 5)
    approx_top1_correct_bits = np.asarray(approx_top1 == labels, dtype=np.uint8)
    approx_top5_correct_bits = np.asarray(np.any(approx_top5 == labels[:, None], axis=1), dtype=np.uint8)

    approx_margin_top1 = margin_top1_against_exact_class(logits, exact_cls)
    approx_margin_top5 = margin_top5_for_true_label(logits, labels)
    delta_margin_top1 = np.asarray(approx_margin_top1 - np.asarray(clean_margin_top1, dtype=np.float32), dtype=np.float32)
    delta_margin_top5 = np.asarray(approx_margin_top5 - np.asarray(clean_margin_top5, dtype=np.float32), dtype=np.float32)

    exact_top1 = np.asarray(exact_top1_correct_bits, dtype=np.uint8)
    exact_top5 = np.asarray(exact_top5_correct_bits, dtype=np.uint8)
    top1_drop = float(np.mean(exact_top1.astype(np.float32) - approx_top1_correct_bits.astype(np.float32), dtype=np.float64))
    top5_drop = float(np.mean(exact_top5.astype(np.float32) - approx_top5_correct_bits.astype(np.float32), dtype=np.float64))
    return {
        "delta_margin_top1": delta_margin_top1,
        "delta_margin_top5": delta_margin_top5,
        "approx_top1_correct_bits": approx_top1_correct_bits,
        "approx_top5_correct_bits": approx_top5_correct_bits,
        "top1_drop": top1_drop,
        "top5_drop": top5_drop,
    }


def _fit_single_basis(values: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected bank matrix with shape [N, S], got {arr.shape}.")
    rank = int(rank)
    feature_dim = int(arr.shape[1])
    mean = np.mean(arr, axis=0, dtype=np.float64).astype(np.float32)
    centered = arr.astype(np.float64, copy=False) - mean.astype(np.float64, copy=False)[None, :]
    if int(arr.shape[0]) <= 1:
        basis = np.zeros((rank, feature_dim), dtype=np.float32)
        coeff = np.zeros((int(arr.shape[0]), rank), dtype=np.float32)
        coeff_mean = np.zeros((rank,), dtype=np.float32)
        coeff_std = np.ones((rank,), dtype=np.float32)
        return mean, basis, coeff_mean, coeff_std, coeff

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    take = min(rank, int(vt.shape[0]))
    basis = np.zeros((rank, feature_dim), dtype=np.float32)
    if take > 0:
        basis[:take] = np.asarray(vt[:take], dtype=np.float32)
    coeff = project_onto_basis(arr, mean=mean, basis=basis)
    coeff_mean = np.mean(coeff, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.maximum(np.std(coeff, axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    return mean, basis, coeff_mean, coeff_std, coeff


def project_onto_basis(values: np.ndarray, *, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    basis_arr = np.asarray(basis, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected values with shape [N, S], got {arr.shape}.")
    centered = arr.astype(np.float64, copy=False) - mean_arr.astype(np.float64, copy=False)[None, :]
    coeff = centered @ basis_arr.astype(np.float64, copy=False).T
    return np.asarray(coeff, dtype=np.float32)


def reconstruct_from_basis(coeff: np.ndarray, *, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coeff_arr = np.asarray(coeff, dtype=np.float32)
    basis_arr = np.asarray(basis, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    if coeff_arr.ndim != 2:
        raise ValueError(f"Expected coeff with shape [N, R], got {coeff_arr.shape}.")
    recon = coeff_arr.astype(np.float64, copy=False) @ basis_arr.astype(np.float64, copy=False)
    recon = recon + mean_arr.astype(np.float64, copy=False)[None, :]
    return np.asarray(recon, dtype=np.float32)


def fit_dual_bank_basis(
    *,
    delta_margin_top1: np.ndarray,
    delta_margin_top5: np.ndarray,
    rank: int = 32,
) -> dict[str, np.ndarray]:
    bank_mean_top1, basis_top1, coeff_mean_top1, coeff_std_top1, coeff_top1 = _fit_single_basis(delta_margin_top1, rank=rank)
    bank_mean_top5, basis_top5, coeff_mean_top5, coeff_std_top5, coeff_top5 = _fit_single_basis(delta_margin_top5, rank=rank)
    return {
        "bank_mean_top1": bank_mean_top1,
        "bank_mean_top5": bank_mean_top5,
        "bank_basis_top1": basis_top1,
        "bank_basis_top5": basis_top5,
        "coeff_mean_top1": coeff_mean_top1,
        "coeff_std_top1": coeff_std_top1,
        "coeff_mean_top5": coeff_mean_top5,
        "coeff_std_top5": coeff_std_top5,
        "coeff_top1": coeff_top1,
        "coeff_top5": coeff_top5,
    }


def variance_retained(values: np.ndarray, *, mean: np.ndarray, basis: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    centered = arr.astype(np.float64, copy=False) - np.asarray(mean, dtype=np.float32).astype(np.float64, copy=False)[None, :]
    total = float(np.sum(centered * centered))
    if total <= 1.0e-12:
        return 1.0
    coeff = project_onto_basis(arr, mean=mean, basis=basis)
    recon = reconstruct_from_basis(coeff, mean=mean, basis=basis)
    residual = arr.astype(np.float64, copy=False) - recon.astype(np.float64, copy=False)
    residual_energy = float(np.sum(residual * residual))
    return float(max(0.0, 1.0 - (residual_energy / total)))


def load_exact_baseline(
    *,
    subset_path: Path = DEFAULT_CALIB_SUBSET_PATH,
    baseline_path: Path = DEFAULT_EXACT_BASELINE_PATH,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    subset = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    payload = np.load(Path(baseline_path), allow_pickle=False)
    logits = np.asarray(payload["logits"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if int(logits.shape[0]) != int(labels.shape[0]):
        raise ValueError("Exact baseline logits and labels do not align.")
    return subset, logits, labels


def load_transformer_ame_bank_meta(path: Path) -> TransformerAMEBankMeta:
    payload = np.load(Path(path).resolve(), allow_pickle=False)
    required = (
        "clean_margin_top1",
        "clean_margin_top5",
        "exact_top1_correct_bits",
        "exact_top5_correct_bits",
        "clean_entropy",
        "exact_pred_class",
        "true_label",
        "bank_mean_top1",
        "bank_mean_top5",
        "bank_basis_top1",
        "bank_basis_top5",
        "coeff_mean_top1",
        "coeff_std_top1",
        "coeff_mean_top5",
        "coeff_std_top5",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        preview = ", ".join(missing)
        raise KeyError(f"Bank meta is missing required fields: {preview}")
    return TransformerAMEBankMeta(
        clean_margin_top1=np.asarray(payload["clean_margin_top1"], dtype=np.float32),
        clean_margin_top5=np.asarray(payload["clean_margin_top5"], dtype=np.float32),
        exact_top1_correct_bits=np.asarray(payload["exact_top1_correct_bits"], dtype=np.uint8),
        exact_top5_correct_bits=np.asarray(payload["exact_top5_correct_bits"], dtype=np.uint8),
        clean_entropy=np.asarray(payload["clean_entropy"], dtype=np.float32),
        exact_pred_class=np.asarray(payload["exact_pred_class"], dtype=np.int64),
        true_label=np.asarray(payload["true_label"], dtype=np.int64),
        bank_mean_top1=np.asarray(payload["bank_mean_top1"], dtype=np.float32),
        bank_mean_top5=np.asarray(payload["bank_mean_top5"], dtype=np.float32),
        bank_basis_top1=np.asarray(payload["bank_basis_top1"], dtype=np.float32),
        bank_basis_top5=np.asarray(payload["bank_basis_top5"], dtype=np.float32),
        coeff_mean_top1=np.asarray(payload["coeff_mean_top1"], dtype=np.float32),
        coeff_std_top1=np.asarray(payload["coeff_std_top1"], dtype=np.float32),
        coeff_mean_top5=np.asarray(payload["coeff_mean_top5"], dtype=np.float32),
        coeff_std_top5=np.asarray(payload["coeff_std_top5"], dtype=np.float32),
        basis_ready=bool(int(payload["basis_ready"])) if "basis_ready" in payload else True,
    )
