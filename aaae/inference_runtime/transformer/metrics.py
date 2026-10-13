from __future__ import annotations


import numpy as np

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0.0, 1.0, den)
    return num / den


def _topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(logits, -k, axis=1)[:, -k:]
    order = np.take_along_axis(logits, part, axis=1)
    sorted_idx = np.argsort(order, axis=1)[:, ::-1]
    return np.take_along_axis(part, sorted_idx, axis=1)


def _metric_accumulator_init() -> dict[str, float | int]:
    return {
        "mse_sum": 0.0,
        "output_cos_sum": 0.0,
        "prob_cos_sum": 0.0,
        "kl_sum": 0.0,
        "top1_consistent": 0,
        "top5_consistent": 0,
        "approx_top1_correct": 0,
        "approx_top5_correct": 0,
    }


def _metric_accumulator_update(
    acc: dict[str, float | int],
    *,
    approx_logits: np.ndarray,
    batch_exact: np.ndarray,
    batch_exact_probs: np.ndarray,
    batch_exact_top1: np.ndarray,
    batch_labels: np.ndarray,
) -> None:
    diff = approx_logits - batch_exact
    acc["mse_sum"] = float(acc["mse_sum"]) + float(np.mean(diff * diff, axis=1).sum())
    acc["output_cos_sum"] = float(acc["output_cos_sum"]) + float(_cosine_rows(approx_logits, batch_exact).sum())

    approx_probs = _softmax(approx_logits.astype(np.float64))
    acc["prob_cos_sum"] = float(acc["prob_cos_sum"]) + float(_cosine_rows(approx_probs, batch_exact_probs).sum())
    acc["kl_sum"] = float(acc["kl_sum"]) + float(
        np.sum(
            batch_exact_probs
            * (
                np.log(np.clip(batch_exact_probs, 1e-12, 1.0))
                - np.log(np.clip(approx_probs, 1e-12, 1.0))
            ),
            axis=1,
        ).sum()
    )

    approx_top1 = np.argmax(approx_logits, axis=1)
    approx_top5 = _topk_indices(approx_logits, 5)
    acc["top1_consistent"] = int(acc["top1_consistent"]) + int(np.count_nonzero(approx_top1 == batch_exact_top1))
    acc["top5_consistent"] = int(acc["top5_consistent"]) + int(
        np.count_nonzero(np.any(approx_top5 == batch_exact_top1[:, None], axis=1))
    )
    acc["approx_top1_correct"] = int(acc["approx_top1_correct"]) + int(np.count_nonzero(approx_top1 == batch_labels))
    acc["approx_top5_correct"] = int(acc["approx_top5_correct"]) + int(
        np.count_nonzero(np.any(approx_top5 == batch_labels[:, None], axis=1))
    )


def _metric_accumulator_finalize(
    acc: dict[str, float | int],
    *,
    sample_count: int,
    exact_top1_acc: float,
    exact_top5_acc: float,
) -> dict[str, float]:
    approx_top1_acc = float(int(acc["approx_top1_correct"]) / sample_count)
    approx_top5_acc = float(int(acc["approx_top5_correct"]) / sample_count)
    return {
        "sample_count": sample_count,
        "exact_top1_acc": exact_top1_acc,
        "approx_top1_acc": approx_top1_acc,
        "top1_drop": float(exact_top1_acc - approx_top1_acc),
        "exact_top5_acc": exact_top5_acc,
        "approx_top5_acc": approx_top5_acc,
        "top5_drop": float(exact_top5_acc - approx_top5_acc),
        "top1_consistency": float(int(acc["top1_consistent"]) / sample_count),
        "top5_consistency": float(int(acc["top5_consistent"]) / sample_count),
        "logit_kl": float(float(acc["kl_sum"]) / sample_count),
        "logit_cosine": float(float(acc["prob_cos_sum"]) / sample_count),
        "output_mse": float(float(acc["mse_sum"]) / sample_count),
        "output_cosine": float(float(acc["output_cos_sum"]) / sample_count),
    }


