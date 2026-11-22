"""Evaluation for MediTrust: metrics from additive statistics, and the test-set lock.

Metrics are computed from additive sufficient statistics (protocol, section 1.3): each
client (or the whole validation set when K = 1) returns sums and counts, the server adds
them and computes the metric. This is exact for accuracy, NLL, Brier, macro-F1 and ECE,
and approximate (with a reported bound) for the histogram AUROC. The same function is
used for centralised and federated evaluation.

The test set is only read through `evaluate_test`, which refuses any run that has not
been frozen first (protocol, section 7).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

N_CLASSES = 9
ECE_BINS = 15
AUC_BINS = 1000


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict(model, images, batch_size: int = 512, device=None, amp: bool = True) -> np.ndarray:
    """Logits (float32 numpy, N x C) for uint8 images N x H x W x C (numpy or torch)."""
    import torch

    device = device or next(model.parameters()).device
    use_amp = amp and torch.device(device).type == "cuda"
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(np.ascontiguousarray(batch))
            x = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Additive sufficient statistics
# --------------------------------------------------------------------------- #

def sufficient_stats(logits: np.ndarray, labels: np.ndarray, n_classes: int = N_CLASSES) -> dict:
    """Additive statistics of one set of predictions. Sum them with `merge_stats`."""
    labels = np.asarray(labels).reshape(-1)
    n = len(labels)
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(log_p)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == labels
    onehot = np.eye(n_classes)[labels]

    confusion = np.zeros((n_classes, n_classes), np.int64)
    np.add.at(confusion, (labels, pred), 1)

    b = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
    ece_count = np.bincount(b, minlength=ECE_BINS)
    ece_conf = np.bincount(b, weights=conf, minlength=ECE_BINS)
    ece_correct = np.bincount(b, weights=correct, minlength=ECE_BINS)

    hb = np.minimum((p * AUC_BINS).astype(int), AUC_BINS - 1)  # N x C
    auc_pos = np.zeros((n_classes, AUC_BINS), np.int64)
    auc_neg = np.zeros((n_classes, AUC_BINS), np.int64)
    for c in range(n_classes):
        pos = labels == c
        auc_pos[c] = np.bincount(hb[pos, c], minlength=AUC_BINS)
        auc_neg[c] = np.bincount(hb[~pos, c], minlength=AUC_BINS)

    return {
        "n": n,
        "correct": int(correct.sum()),
        "nll_sum": float(-log_p[np.arange(n), labels].sum()),
        "brier_sum": float(((p - onehot) ** 2).sum()),
        "confusion": confusion,
        "ece_count": ece_count, "ece_conf": ece_conf, "ece_correct": ece_correct,
        "auc_pos": auc_pos, "auc_neg": auc_neg,
    }


def merge_stats(stats: list[dict]) -> dict:
    """Sum the statistics of several clients."""
    return {k: sum(s[k] for s in stats) for k in stats[0]}


def _histogram_auroc(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """AUROC from score histograms (ties within a bin count 1/2) and its error bound."""
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    neg_below = np.concatenate([[0], np.cumsum(neg)[:-1]])
    same_bin = (pos * neg).sum()
    auc = ((pos * neg_below).sum() + 0.5 * same_bin) / (n_pos * n_neg)
    return float(auc), float(0.5 * same_bin / (n_pos * n_neg))


def metrics_from_stats(s: dict, present_only: bool = False) -> dict:
    """Metrics from (merged) sufficient statistics.

    `present_only=True` averages per-class F1 over the classes present in the labels,
    as specified for per-client metrics in E2.
    """
    n = s["n"]
    cm = s["confusion"].astype(np.float64)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    classes = support > 0 if present_only else np.ones_like(support, bool)
    aucs, bounds = zip(*(_histogram_auroc(s["auc_pos"][c], s["auc_neg"][c]) for c in range(len(tp))))
    ece = np.abs(s["ece_correct"] - s["ece_conf"]).sum() / n
    return {
        "n": int(n),
        "accuracy": s["correct"] / n,
        "macro_f1": float(f1[classes].mean()),
        "nll": s["nll_sum"] / n,
        "brier": s["brier_sum"] / n,
        "ece": float(ece),
        "auroc_macro_hist": float(np.nanmean(aucs)),
        "auroc_hist_max_error": float(np.nanmax(bounds)),
        "recall_per_class": recall.tolist(),
        "precision_per_class": precision.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion": s["confusion"].tolist(),
    }


def exact_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Metrics from the stats, plus exact one-vs-rest macro AUROC (scikit-learn)."""
    from sklearn.metrics import roc_auc_score

    m = metrics_from_stats(sufficient_stats(logits, labels))
    m["auroc_macro"] = float(roc_auc_score(labels, softmax(logits), multi_class="ovr", average="macro"))
    return m


def evaluate(model, images, labels, batch_size: int = 512, device=None) -> dict:
    """Validation metrics and inference time (seconds) for one split."""
    t0 = time.perf_counter()
    logits = predict(model, images, batch_size, device)
    elapsed = time.perf_counter() - t0
    m = exact_metrics(logits, np.asarray(labels))
    m["inference_seconds"] = elapsed
    return m


# --------------------------------------------------------------------------- #
# Freeze register and test lock (protocol, section 7)
# --------------------------------------------------------------------------- #

FREEZE_PATH = "results/freeze.json"
TEST_LOG_PATH = "results/test_log.csv"


def _load_freezes(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def freeze(run_id: str, items: dict, commit: str, path=FREEZE_PATH) -> None:
    """Record the frozen decisions for `run_id`. A run can only be frozen once."""
    records = _load_freezes(path)
    if any(r["run_id"] == run_id for r in records):
        raise ValueError(f"{run_id} is already frozen; freezes are immutable")
    records.append({"run_id": run_id, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "commit": commit, "items": items})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)


def is_frozen(run_id: str, path=FREEZE_PATH) -> bool:
    return any(r["run_id"] == run_id for r in _load_freezes(path))


def evaluate_test(run_id: str, model, test_images, test_labels, commit: str,
                  freeze_path=FREEZE_PATH, log_path=TEST_LOG_PATH, device=None) -> dict:
    """The only function allowed to read the test set. Refuses runs that are not frozen."""
    if not is_frozen(run_id, freeze_path):
        raise PermissionError(f"test evaluation refused: {run_id} is not in {freeze_path}")
    m = exact_metrics(predict(model, test_images, device=device), np.asarray(test_labels))
    new = not os.path.exists(log_path)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "run_id", "commit", "accuracy", "macro_f1", "auroc_macro"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), run_id, commit,
                    f"{m['accuracy']:.6f}", f"{m['macro_f1']:.6f}", f"{m['auroc_macro']:.6f}"])
    return m
