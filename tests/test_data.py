"""Tests for src/data.py.

Synthetic tests always run. Tests marked by `needs_data` run only if
pathmnist_64.npz is available (MEDITRUST_DATA env var, default ./data).
Tests needing PyTorch are skipped when it is not installed.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src import data as D

DATA_ROOT = Path(os.environ.get("MEDITRUST_DATA", Path(__file__).resolve().parents[1] / "data"))
HAS_DATA = (DATA_ROOT / D.FILENAME).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"{D.FILENAME} not found in {DATA_ROOT}")


def synthetic_labels(n=20000, seed=123):
    rng = np.random.default_rng(seed)
    # Unbalanced classes, like PathMNIST.
    p = rng.dirichlet(np.full(D.N_CLASSES, 5.0))
    return rng.choice(D.N_CLASSES, size=n, p=p).astype(np.int64)


TRAIN = synthetic_labels(20000, 1)
VAL = synthetic_labels(2500, 2)


# ----------------------------- constants ---------------------------------- #

def test_official_constants():
    assert D.N_CLASSES == 9
    assert sum(D.N_SAMPLES.values()) == 107180
    assert D.N_SAMPLES == {"train": 89996, "val": 10004, "test": 7180}


def test_validate_split_rejects_wrong_shapes():
    n = D.N_SAMPLES["test"]
    images = np.zeros((n, 64, 64, 3), np.uint8)
    D.validate_split(images, np.zeros(n, np.int64), "test")
    with pytest.raises(ValueError):
        D.validate_split(images[:-1], np.zeros(n - 1, np.int64), "test")
    with pytest.raises(ValueError):
        D.validate_split(images, np.full(n, 9, np.int64), "test")


# ----------------------------- partitions --------------------------------- #

@pytest.mark.parametrize("scheme", ["IID", "DIR05"])
def test_partition_disjoint_and_exhaustive(scheme):
    part = D.make_partition(TRAIN, VAL, scheme, min_train=1000)
    D.check_partition(part["train"], len(TRAIN))
    D.check_partition(part["val"], len(VAL))
    assert len(part["train"]) == 4


@pytest.mark.parametrize("scheme", ["IID", "DIR05"])
def test_partition_reproducible(scheme):
    a = D.make_partition(TRAIN, VAL, scheme, seed=0, min_train=1000)
    b = D.make_partition(TRAIN, VAL, scheme, seed=0, min_train=1000)
    c = D.make_partition(TRAIN, VAL, scheme, seed=1, min_train=1000)
    assert D.partition_hash(a) == D.partition_hash(b)
    assert D.partition_hash(a) != D.partition_hash(c)


def test_iid_class_distribution_is_balanced():
    part = D.make_partition(TRAIN, VAL, "IID")
    for split, labels in (("train", TRAIN), ("val", VAL)):
        counts = D.partition_class_counts(part[split], labels)
        # Each class split into 4 parts differing by at most one image.
        assert (counts.max(0) - counts.min(0) <= 1).all()
        assert (counts.sum(0) == D.class_counts(labels)).all()


def test_dirichlet_class_distribution_follows_proportions():
    part = D.make_partition(TRAIN, VAL, "DIR05", min_train=1000)
    p = part["proportions"]
    assert p.shape == (9, 4) and np.allclose(p.sum(1), 1)
    for split, labels in (("train", TRAIN), ("val", VAL)):
        counts = D.partition_class_counts(part[split], labels)  # k x C
        expected = p.T * D.class_counts(labels)[None, :]
        assert np.abs(counts - expected).max() <= 1.0
    sizes = [len(x) for x in part["train"]]
    assert min(sizes) >= 1000
    # Label skew: client class mixes differ much more than in IID.
    shares = D.partition_class_counts(part["train"], TRAIN) / np.array(sizes)[:, None]
    assert np.abs(shares - shares.mean(0)).max() > 0.05


def test_dirichlet_min_train_unreachable_raises():
    with pytest.raises(RuntimeError):
        D.make_partition(TRAIN, VAL, "DIR05", min_train=6000, max_draws=5)


def test_save_load_roundtrip(tmp_path):
    part = D.make_partition(TRAIN, VAL, "DIR05", min_train=1000)
    path = tmp_path / "p.npz"
    D.save_partition(part, path)
    loaded = D.load_partition(path)
    assert D.partition_hash(loaded) == D.partition_hash(part)
    assert loaded["scheme"] == "DIR05" and loaded["n_draws"] == part["n_draws"]


# ----------------------------- real dataset ------------------------------- #

@pytest.fixture(scope="module")
def dataset():
    return D.load_pathmnist(DATA_ROOT, check_md5=True)


@needs_data
def test_dataset_shapes_counts_classes(dataset):
    for split in D.SPLITS:
        images, labels = dataset[split]
        assert images.shape == (D.N_SAMPLES[split], 64, 64, 3)
        assert images.dtype == np.uint8
        assert set(np.unique(labels)) == set(range(9))


@needs_data
def test_dataset_pixel_range(dataset):
    for split in D.SPLITS:
        images, _ = dataset[split]
        x = images[:2000].astype(np.float32) / 255.0
        assert 0.0 <= x.min() and x.max() <= 1.0
        assert images.max() > 200 and images.min() < 50  # not blank or constant


@needs_data
def test_real_partitions_valid(dataset):
    train, val = dataset["train"][1], dataset["val"][1]
    for scheme in ("IID", "DIR05"):
        part = D.make_partition(train, val, scheme)
        D.check_partition(part["train"], len(train))
        D.check_partition(part["val"], len(val))
        assert min(len(x) for x in part["train"]) >= 2000


# ----------------------------- torch preprocessing ------------------------ #

def test_to_tensor_and_augment():
    torch = pytest.importorskip("torch")
    images = np.random.default_rng(0).integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
    x = D.to_tensor(images)
    assert x.shape == (8, 3, 64, 64) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert torch.allclose(x[0, :, 5, 7] * 255, torch.tensor(images[0, 5, 7], dtype=torch.float32))
    g = torch.Generator().manual_seed(0)
    y = D.augment(x, g)
    assert y.shape == x.shape
    # Flips and 90-degree rotations only permute pixels.
    assert torch.allclose(y.flatten(1).sort(1).values, x.flatten(1).sort(1).values)
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(y, D.augment(x, g2))
