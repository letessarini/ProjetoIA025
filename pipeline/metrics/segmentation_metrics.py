from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def count_pixels(pred, target, valid=None):
    """Return tp, fp, fn, tn over valid pixels only."""
    p = np.asarray(pred, dtype=bool).ravel()
    a = np.asarray(target, dtype=bool).ravel()
    if valid is None:
        m = np.ones_like(p, dtype=bool)
    else:
        m = np.asarray(valid, dtype=bool).ravel()
    p, a = p[m], a[m]
    return int(np.sum(p & a)), int(np.sum(p & ~a)), int(np.sum(~p & a)), int(np.sum(~p & ~a))



def compute_metrics(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-9) -> Dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    n = tp + fp + fn + tn

    f1 = 2 * precision * recall / (precision + recall + eps)
    f2 = 5 * precision * recall / (4 * precision + recall + eps)

    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": float(f1),
        "f2": float(f2),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": (tp + tn) / (n + eps),
        "prevalence": (tp + fn) / (n + eps),
        "frac_predicted": (tp + fp) / (n + eps),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }



def compute_histogram_metrics(hist, threshold=None, selection_metric="iou"):
    """Compute metrics from negative/positive score bins using a fixed or selected cutoff."""
    neg, pos = hist
    fp = neg[::-1].cumsum()[::-1]
    tp = pos[::-1].cumsum()[::-1]
    positives, negatives = pos.sum(), neg.sum()
    precision = tp / np.maximum(tp + fp, 1)
    iou = tp / np.maximum(positives + fp, 1)
    f2 = 5 * tp / np.maximum(4 * positives + tp + fp, 1)
    steps = len(pos) - 1
    if threshold is None:
        objective = {"iou": iou, "f2": f2}[selection_metric]
        index = 1 + int(np.argmax(objective[1:]))
    else:
        index = round(threshold * steps)
    metrics = compute_metrics(int(tp[index]), int(fp[index]), int(positives - tp[index]), int(negatives - fp[index]))
    metrics["ap_binned"] = float(pos @ precision / positives) if positives else None
    return {"threshold": index / steps, "metrics": metrics}


def compute_metrics_for_prediction(pred, target, valid=None, threshold: float = 0.5) -> Dict[str, float]:
    pred_bin = np.asarray(pred) >= threshold
    tp, fp, fn, tn = count_pixels(pred_bin, target, valid)
    return compute_metrics(tp, fp, fn, tn)


def bootstrap_iou_confidence_interval(
    probs: Sequence[np.ndarray],
    alvos: Sequence[np.ndarray],
    validos: Sequence[np.ndarray],
    threshold: float = 0.5,
    n_bootstrap: int = 300,
    seed: int = 20261022,
) -> Tuple[float, float]:
    """Bootstrap 95% confidence interval for IoU at patch level."""
    if not probs:
        return float("nan"), float("nan")

    scores = []
    for p, a, v in zip(probs, alvos, validos):
        tp, fp, fn, _ = count_pixels(np.asarray(p) >= threshold, a, v)
        scores.append(tp / (tp + fp + fn + 1e-9))

    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(max(1, n_bootstrap)):
        idx = rng.randint(0, len(arr), len(arr))
        sample = arr[idx]
        tp = float(np.sum(sample >= 0.0))
        # patch-level iou bootstrap is approximated by the sampled mean IoU
        vals.append(float(np.mean(sample)))

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    pred = np.array([[0.1, 0.9], [0.8, 0.2]])
    target = np.array([[0, 1], [1, 0]])
    valid = np.array([[1, 1], [1, 1]])
    print(compute_metrics_for_prediction(pred, target, valid, threshold=0.5))
