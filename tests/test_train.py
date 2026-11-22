"""Tests for src/train.py: schedule, checkpoint/resume equivalence, budget estimate (CPU)."""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src import train as T

rng = np.random.default_rng(0)
TRAIN = (rng.integers(0, 256, (48, 64, 64, 3), dtype=np.uint8), np.arange(48) % 9)
VAL = (rng.integers(0, 256, (18, 64, 64, 3), dtype=np.uint8), np.arange(18) % 9)
CFG = {"model": "R18", "lr": 0.01, "epochs": 2, "batch_size": 8, "seed": 0, "pretrained": False}


def test_cosine_lr():
    assert T.cosine_lr(0.1, 0, 10) == pytest.approx(0.1)
    assert T.cosine_lr(0.1, 5, 10) == pytest.approx(0.05)
    assert 0 < T.cosine_lr(0.1, 9, 10) < 0.003


def test_lr_required(tmp_path):
    with pytest.raises(ValueError):
        T.fit({**CFG, "lr": None}, TRAIN, VAL, "x", tmp_path, tmp_path, device="cpu")


def test_interrupt_resume_matches_uninterrupted(tmp_path):
    a = T.fit(CFG, TRAIN, VAL, "run", tmp_path / "ck_a", tmp_path / "res", "cpu",
              stop_after_epoch=1, max_steps_per_epoch=2)
    assert a["status"] == "interrupted" and a["epochs_completed"] == 1
    b = T.fit(CFG, TRAIN, VAL, "run", tmp_path / "ck_a", tmp_path / "res", "cpu", max_steps_per_epoch=2)
    assert b["status"] == "complete" and b["resumed_from_epoch"] == 1 and len(b["history"]) == 2
    ref = T.fit(CFG, TRAIN, VAL, "ref", tmp_path / "ck_b", tmp_path / "res", "cpu", max_steps_per_epoch=2)
    assert ref["history"][1]["train_loss"] == pytest.approx(b["history"][1]["train_loss"], rel=1e-5)
    wa = torch.load(tmp_path / "ck_a" / "run" / "last.pt", weights_only=False)["model"]
    wb = torch.load(tmp_path / "ck_b" / "ref" / "last.pt", weights_only=False)["model"]
    assert all(torch.allclose(wa[k].float(), wb[k].float(), atol=1e-5) for k in wa)
    saved = json.loads((tmp_path / "res" / "run.json").read_text())
    assert saved["status"] == "complete" and saved["checkpoint_mb"] > 0
    # A completed run is not retrained.
    again = T.fit(CFG, TRAIN, VAL, "run", tmp_path / "ck_a", tmp_path / "res", "cpu", max_steps_per_epoch=2)
    assert again["status"] == "complete" and len(again["history"]) == 2


def test_resume_rejects_other_config(tmp_path):
    T.fit(CFG, TRAIN, VAL, "run", tmp_path, tmp_path, "cpu", stop_after_epoch=1, max_steps_per_epoch=1)
    with pytest.raises(ValueError):
        T.fit({**CFG, "lr": 0.03}, TRAIN, VAL, "run", tmp_path, tmp_path, "cpu", max_steps_per_epoch=1)


def test_resume_rng_state_stays_on_cpu(tmp_path, monkeypatch):
    """Regression (Colab T4): `TypeError: RNG state must be a torch.ByteTensor` on resume.

    The checkpoint was loaded with map_location=<cuda device>, which also moved the CPU RNG
    state to the GPU. Without a GPU, the same failure is emulated by moving every tensor of
    the loaded checkpoint to the "meta" device whenever map_location is not "cpu".
    """
    T.fit(CFG, TRAIN, VAL, "run", tmp_path, tmp_path, "cpu", stop_after_epoch=1, max_steps_per_epoch=1)
    saved = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert saved["rng_cpu"].dtype == torch.uint8 and saved["rng_cpu"].device.type == "cpu"

    real_load, seen = torch.load, []

    def load_like_gpu(path, map_location=None, **kw):
        seen.append(map_location)
        state = real_load(path, map_location="cpu", **kw)
        # Any map_location other than the literal "cpu" follows the training device,
        # which on Colab is CUDA: emulate it by moving the RNG state off the CPU.
        if not (isinstance(map_location, str) and map_location == "cpu"):
            state["rng_cpu"] = state["rng_cpu"].to("meta")
        return state

    monkeypatch.setattr(T.torch, "load", load_like_gpu)
    out = T.fit(CFG, TRAIN, VAL, "run", tmp_path, tmp_path, "cpu", max_steps_per_epoch=1)
    assert seen == ["cpu"] and out["status"] == "complete" and out["resumed_from_epoch"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (runs on Colab via notebook 02)")
def test_interrupt_resume_on_gpu(tmp_path):
    """Same scenario as the failing pilot: interrupt after epoch 1 on CUDA, then resume."""
    cfg = {**CFG, "epochs": 2}
    T.fit(cfg, TRAIN, VAL, "run", tmp_path / "a", tmp_path / "r", "cuda", stop_after_epoch=1, max_steps_per_epoch=2)
    state = torch.load(tmp_path / "a" / "run" / "last.pt", weights_only=False)
    assert state["rng_cpu"].device.type == "cpu" and state["rng_cuda"].dtype == torch.uint8
    b = T.fit(cfg, TRAIN, VAL, "run", tmp_path / "a", tmp_path / "r", "cuda", max_steps_per_epoch=2)
    ref = T.fit(cfg, TRAIN, VAL, "ref", tmp_path / "b", tmp_path / "r", "cuda", max_steps_per_epoch=2)
    assert b["status"] == "complete" and b["resumed_from_epoch"] == 1
    wa = torch.load(tmp_path / "a" / "run" / "last.pt", weights_only=False)["model"]
    wb = torch.load(tmp_path / "b" / "ref" / "last.pt", weights_only=False)["model"]
    # GPU kernels (cuDNN, AMP) are not bitwise deterministic: compare with a tolerance.
    assert all(torch.allclose(wa[k].float(), wb[k].float(), atol=1e-3, rtol=1e-3) for k in wa)


def test_budget_estimate():
    est = T.estimate_budget({"R18": 3.0, "DEIT": 2.5}, {"R18": 0.1, "DEIT": 0.1})
    for scenario in est.values():
        lv = scenario["levels"]
        assert lv["N2"]["hours_with_margin"] < lv["N1"]["hours_with_margin"] < lv["N0"]["hours_with_margin"]
        assert scenario["suggested_level"] in ("N0", "N1", "N2", "N3 (needs approval)")
    # N0 units: 2 x 36.4 (LR search, E1, FGSM curves) + 118.8 for M (E2, E4, E3/E6) = 191.6 U.
    assert est["M=R18"]["levels"]["N0"]["units"] == pytest.approx(191.6, abs=0.1)
    assert math.isclose(est["M=R18"]["levels"]["N0"]["hours_raw"] * 1.3,
                        est["M=R18"]["levels"]["N0"]["hours_with_margin"])
