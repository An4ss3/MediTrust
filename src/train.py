"""Centralised training for MediTrust (E1), with checkpoint/resume, timing and JSON summaries.

Configuration follows docs/PROTOCOLE_EXPERIMENTAL.md, section 4: SGD (momentum 0.9,
weight decay 1e-4), cosine learning rate held constant within an epoch, batch 128,
fp16 AMP on GPU, gradient clipping at norm 1.0, last checkpoint kept (no early stopping).
Only the train and validation splits are used here; the test set is never touched.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import augment
from src.evaluate import evaluate
from src.models import build_model, count_parameters

DEFAULT_CONFIG = {
    "model": "R18",
    "lr": None,             # chosen on validation by the LR search (protocol 4.1)
    "epochs": 10,
    "batch_size": 128,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "amp": True,
    "seed": 0,
    "pretrained": True,
}
# Keys that must match when resuming from a checkpoint.
RESUME_KEYS = ("model", "lr", "epochs", "batch_size", "momentum", "weight_decay", "grad_clip", "seed")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def cosine_lr(base_lr: float, epoch: int, epochs: int) -> float:
    """Cosine schedule, constant within an epoch (or a FL round): epoch 0 uses base_lr."""
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))


def env_info(device) -> dict:
    import timm

    info = {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device)}
    if torch.device(device).type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def _save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def fit(config: dict, train, val, run_id: str, ckpt_root, results_dir,
        device=None, meta: dict | None = None, stop_after_epoch: int | None = None,
        max_steps_per_epoch: int | None = None) -> dict:
    """Train one model and evaluate it on the validation split after every epoch.

    train, val: (uint8 images N x 64 x 64 x 3, int64 labels N) from `load_pathmnist`.
    Resumes automatically from `<ckpt_root>/<run_id>/last.pt`. Writes
    `<results_dir>/<run_id>.json` after every epoch.
    `stop_after_epoch` simulates an interruption (used by the pilot and the tests).
    `max_steps_per_epoch` is only for tests on CPU; it must stay None for real runs.
    """
    cfg = {**DEFAULT_CONFIG, **config}
    if cfg["lr"] is None:
        raise ValueError("config['lr'] must be set (chosen on validation by the LR search)")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = cfg["amp"] and device.type == "cuda"
    set_seed(cfg["seed"])

    run_dir = Path(ckpt_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    json_path = Path(results_dir) / f"{run_id}.json"

    model = build_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, history, train_seconds = 0, [], 0.0
    resumed_from = None
    if ckpt_path.exists():
        # Load on CPU: the RNG states must stay CPU ByteTensors (map_location="cuda" would move
        # them to the GPU and torch.set_rng_state would fail). load_state_dict copies the model
        # and optimizer tensors to the device of the parameters.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mismatch = {k: (state["config"][k], cfg[k]) for k in RESUME_KEYS if state["config"][k] != cfg[k]}
        if mismatch:
            raise ValueError(f"checkpoint {ckpt_path} has a different configuration: {mismatch}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, history, train_seconds = state["epoch"], state["history"], state["train_seconds"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and "rng_cuda" in state:  # absent in checkpoints saved before this fix
            torch.cuda.set_rng_state(state["rng_cuda"], device)
        resumed_from = start_epoch

    images = torch.from_numpy(np.ascontiguousarray(train[0])).to(device)  # uint8, ~1.1 GB on GPU
    labels = torch.from_numpy(np.asarray(train[1])).to(device)
    n = len(labels)
    steps = math.ceil(n / cfg["batch_size"])
    if max_steps_per_epoch is not None:
        steps = min(steps, max_steps_per_epoch)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    summary = None
    for epoch in range(start_epoch, cfg["epochs"]):
        lr = cosine_lr(cfg["lr"], epoch, cfg["epochs"])
        for group in opt.param_groups:
            group["lr"] = lr
        # One generator per (seed, epoch): the batch order and augmentations of an epoch
        # do not depend on whether the run was interrupted before it.
        gen = torch.Generator().manual_seed(cfg["seed"] * 100_003 + epoch)
        perm = torch.randperm(n, generator=gen)

        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for step in range(steps):
            idx = perm[step * cfg["batch_size"]:(step + 1) * cfg["batch_size"]].to(device)
            x = images[idx].permute(0, 3, 1, 2).float().div_(255.0)
            x = augment(x, gen)
            y = labels[idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(1) == y).sum()
            seen += len(y)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds

        val_metrics = evaluate(model, val[0], val[1], device=device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        history.append({
            "epoch": epoch + 1, "lr": lr, "steps": steps,
            "train_loss": loss_sum.item() / seen, "train_accuracy": correct.item() / seen,
            "train_seconds": epoch_seconds, "train_images_per_second": seen / epoch_seconds,
            "val_seconds": val_metrics["inference_seconds"], "peak_gpu_mb": peak_mb,
            "val": {k: val_metrics[k] for k in ("accuracy", "macro_f1", "auroc_macro", "nll", "ece")},
        })
        rng = {"rng_cpu": torch.get_rng_state()}
        if device.type == "cuda":
            rng["rng_cuda"] = torch.cuda.get_rng_state(device)  # CPU ByteTensor as well
        _save_atomic({"config": cfg, "epoch": epoch + 1, "history": history, "train_seconds": train_seconds,
                      "model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scaler": scaler.state_dict(), **rng}, ckpt_path)

        done = epoch + 1 == cfg["epochs"]
        interrupted = stop_after_epoch is not None and epoch + 1 == stop_after_epoch and not done
        summary = {
            "run_id": run_id, "status": "complete" if done else ("interrupted" if interrupted else "running"),
            "config": cfg, "epochs_completed": epoch + 1, "resumed_from_epoch": resumed_from,
            "parameters": count_parameters(model), "train_seconds_total": train_seconds,
            "peak_gpu_mb": max((h["peak_gpu_mb"] or 0) for h in history) or None,
            "checkpoint": str(ckpt_path), "checkpoint_mb": ckpt_path.stat().st_size / 2**20,
            "val_final": {k: v for k, v in val_metrics.items()},
            "history": history, "env": env_info(device), "meta": meta or {},
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json(summary, json_path)
        if interrupted:
            break
    if summary is None:  # already complete before this call
        with open(json_path, encoding="utf-8") as f:
            summary = json.load(f)
    return summary


# --------------------------------------------------------------------------- #
# Budget estimate from the measured pilot (protocol, section 9)
# --------------------------------------------------------------------------- #

N_TRAIN, N_VAL, N_TEST = 89996, 10004, 7180


def protocol_workload(n_rounds: int = 10) -> list[dict]:
    """Workload of the mandatory protocol in training units U and validation passes.

    `arch` is "each" (both models) or "M" (pipeline model only). Attack passes are counted
    as one forward+backward each, i.e. like a training step, so their cost is expressed in U.
    """
    val_u = N_VAL / N_TRAIN
    test_u = N_TEST / N_TRAIN
    E = n_rounds
    return [
        {"item": "LR search (3 lr x 2 epochs, includes the pilot run)", "arch": "each", "U": 6, "val_passes": 6},
        {"item": "E1 (3 seeds)", "arch": "each", "U": 3 * E, "val_passes": 3 * E},
        {"item": "E2 IID + DIR05 (3 seeds each)", "arch": "M", "U": 6 * E, "val_passes": 6 * E},
        {"item": "E2 LOCAL (4 clients, 1 seed)", "arch": "M", "U": E, "val_passes": E},
        {"item": "E2 ABL (E_local=2, R=5)", "arch": "M", "U": E, "val_passes": 5},
        {"item": "E4 FGSM-RS federated training (2 passes per step)", "arch": "M", "U": 2 * E, "val_passes": E},
        {"item": "E4 PGD-10 monitoring (1,000 val images per round)", "arch": "M", "U": E * 10 * 1000 / N_TRAIN, "val_passes": 0},
        {"item": "E4 val evaluation for D6 (2 models: FGSM@4 + PGD-10 at 2,4,8)", "arch": "M", "U": 2 * 31 * val_u, "val_passes": 2},
        {"item": "E4 test evaluation (2 models: FGSM + PGD-10 at 1,2,4,8,16)", "arch": "M", "U": 2 * 55 * test_u, "val_passes": 0},
        {"item": "E4 FGSM curves for E1 models (test)", "arch": "each", "U": 5 * test_u, "val_passes": 0},
        {"item": "E3 + E6 (calibration, explanations; inference)", "arch": "M", "U": 2, "val_passes": 10},
    ]


CONTINGENCY = {"item": "E4 strengthened PGD if an alert occurs (2 models, 50 it x 5 starts at 4/255)",
               "U": 2 * 250 * N_VAL / N_TRAIN}


def estimate_budget(minutes_per_u: dict, val_pass_minutes: dict, margin: float = 1.3) -> dict:
    """GPU hours of the protocol for each possible pipeline model M and fallback level.

    minutes_per_u: measured minutes for one training epoch, per architecture ("R18", "DEIT").
    val_pass_minutes: measured minutes for one validation pass, per architecture.
    Levels (protocol 9.1): N1 drops E2 ABL and two IID seeds; N2 also uses 6 instead of 10
    epochs/rounds. N3 (112x112 input) needs a new measurement and is not estimated here.
    """
    out = {}
    for m in minutes_per_u:
        levels = {}
        for level in ("N0", "N1", "N2"):
            rows = protocol_workload(n_rounds=6 if level == "N2" else 10)
            if level in ("N1", "N2"):
                rows = [r for r in rows if not r["item"].startswith("E2 ABL")]
                for r in rows:
                    if r["item"].startswith("E2 IID"):
                        r["U"] = r["U"] * 4 / 6  # IID 1 seed + DIR05 3 seeds
                        r["val_passes"] = r["val_passes"] * 4 / 6
            minutes, units = 0.0, 0.0
            for r in rows:
                archs = list(minutes_per_u) if r["arch"] == "each" else [m]
                for a in archs:
                    minutes += r["U"] * minutes_per_u[a] + r["val_passes"] * val_pass_minutes[a]
                    units += r["U"]
            levels[level] = {"units": round(units, 1), "hours_raw": minutes / 60, "hours_with_margin": margin * minutes / 60}
        contingency_h = margin * CONTINGENCY["U"] * minutes_per_u[m] / 60
        # Protocol 9.1: the level is chosen from the projection P of the full plan (N0).
        p = levels["N0"]["hours_with_margin"]
        chosen = "N0" if p <= 15 else "N1" if p <= 25 else "N2" if p <= 40 else "N3 (needs approval)"
        out[f"M={m}"] = {"levels": levels, "suggested_level": chosen,
                         "contingency_hours": contingency_h}
    return out
