"""Tests for src/evaluate.py: metrics, federated aggregation, test lock."""

import numpy as np
import pytest

from src import evaluate as E

rng = np.random.default_rng(0)
N, C = 3000, 9
LABELS = rng.integers(0, C, N)
# Informative but imperfect logits.
LOGITS = (rng.normal(size=(N, C)) + 2.5 * np.eye(C)[LABELS]).astype(np.float32)


def test_metrics_match_reference():
    sk = pytest.importorskip("sklearn.metrics")
    m = E.metrics_from_stats(E.sufficient_stats(LOGITS, LABELS))
    pred = LOGITS.argmax(1)
    p = E.softmax(LOGITS)
    assert m["accuracy"] == pytest.approx(sk.accuracy_score(LABELS, pred))
    assert m["macro_f1"] == pytest.approx(sk.f1_score(LABELS, pred, average="macro"))
    assert np.array_equal(m["confusion"], sk.confusion_matrix(LABELS, pred))
    assert m["recall_per_class"] == pytest.approx(sk.recall_score(LABELS, pred, average=None).tolist())
    assert m["nll"] == pytest.approx(sk.log_loss(LABELS, p))
    assert m["brier"] == pytest.approx(((p - np.eye(C)[LABELS]) ** 2).sum(1).mean())
    # ECE recomputed directly.
    conf, correct = p.max(1), pred == LABELS
    b = np.minimum((conf * 15).astype(int), 14)
    ece = sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(15) if (b == i).any())
    assert m["ece"] == pytest.approx(ece)
    exact = sk.roc_auc_score(LABELS, p, multi_class="ovr", average="macro")
    assert abs(m["auroc_macro_hist"] - exact) <= m["auroc_hist_max_error"] + 1e-12
    assert E.exact_metrics(LOGITS, LABELS)["auroc_macro"] == pytest.approx(exact)


def test_federated_stats_equal_pooled():
    # 4 clients with label skew; one client has no class-0 image.
    order = np.argsort(LABELS + rng.random(N) * 3)
    clients = np.array_split(order, 4)
    stats = [E.sufficient_stats(LOGITS[c], LABELS[c]) for c in clients]
    fed = E.metrics_from_stats(E.merge_stats(stats))
    pooled = E.metrics_from_stats(E.sufficient_stats(LOGITS, LABELS))
    for k in ("accuracy", "macro_f1", "nll", "brier", "ece", "auroc_macro_hist"):
        assert fed[k] == pytest.approx(pooled[k], abs=1e-9), k
    assert fed["confusion"] == pooled["confusion"]
    # Counter-example: the weighted mean of local macro-F1 is not the global macro-F1.
    local = [E.metrics_from_stats(s)["macro_f1"] for s in stats]
    weighted = np.average(local, weights=[len(c) for c in clients])
    assert abs(weighted - pooled["macro_f1"]) > 0.01


def test_present_only_macro_f1():
    keep = LABELS != 0
    s = E.sufficient_stats(LOGITS[keep], LABELS[keep])
    assert E.metrics_from_stats(s, present_only=True)["macro_f1"] > E.metrics_from_stats(s)["macro_f1"]


class _Const:
    """Minimal stand-in for a model in the test-lock tests."""

    def __init__(self):
        import torch
        self.lin = torch.nn.Linear(1, C)

    def parameters(self):
        return self.lin.parameters()

    def eval(self):
        return self

    def __call__(self, x):
        return self.lin(x.mean((1, 2, 3))[:, None])


def test_test_lock(tmp_path):
    pytest.importorskip("torch")
    images = rng.integers(0, 256, (20, 64, 64, 3), dtype=np.uint8)
    labels = np.arange(20) % C
    freeze, log = tmp_path / "freeze.json", tmp_path / "test_log.csv"
    model = _Const()
    with pytest.raises(PermissionError):
        E.evaluate_test("E1-R18-s0", model, images, labels, "abc", freeze, log)
    assert not log.exists()
    E.freeze("E1-R18-s0", {"lr": 0.01}, "abc", freeze)
    with pytest.raises(ValueError):
        E.freeze("E1-R18-s0", {"lr": 0.03}, "abc", freeze)
    m = E.evaluate_test("E1-R18-s0", model, images, labels, "abc", freeze, log)
    assert 0 <= m["accuracy"] <= 1
    rows = log.read_text().strip().splitlines()
    assert len(rows) == 2 and rows[1].split(",")[1] == "E1-R18-s0"
