"""PathMNIST+ 64x64 data pipeline for MediTrust.

Loading, integrity checks, preprocessing and federated partitions, as specified in
docs/PROTOCOLE_EXPERIMENTAL.md (sections 1 and 2).

The loading and partition functions only need NumPy. The tensor functions import
PyTorch lazily, so the module can be tested without PyTorch installed.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

# Official metadata (medmnist/info.py and Zenodo record 10519652, checked 21/11/2026).
FILENAME = "pathmnist_64.npz"
URL = "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1"
MD5 = "55aa9c1e0525abe5a6b9d8343a507616"
SPLITS = ("train", "val", "test")
N_SAMPLES = {"train": 89996, "val": 10004, "test": 7180}
IMAGE_SHAPE = (64, 64, 3)
CLASS_NAMES = (
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
)
N_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Download and integrity
# --------------------------------------------------------------------------- #

def md5sum(path: str | os.PathLike, chunk_size: int = 1 << 22) -> str:
    """Return the MD5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: str | os.PathLike) -> Path:
    """Download pathmnist_64.npz (about 1.1 GB) into `root` and check its MD5.

    The file is first written as `.part` and only renamed once the checksum matches.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / FILENAME
    if path.exists():
        return path
    tmp = path.with_suffix(".npz.part")
    urllib.request.urlretrieve(URL, tmp)
    found = md5sum(tmp)
    if found != MD5:
        raise RuntimeError(f"MD5 mismatch for downloaded file: {found} != {MD5}")
    tmp.replace(path)
    return path


def validate_split(images: np.ndarray, labels: np.ndarray, split: str) -> None:
    """Raise ValueError if one split does not match the official specification."""
    n = N_SAMPLES[split]
    if images.shape != (n, *IMAGE_SHAPE):
        raise ValueError(f"{split}: images shape {images.shape}, expected {(n, *IMAGE_SHAPE)}")
    if images.dtype != np.uint8:
        raise ValueError(f"{split}: images dtype {images.dtype}, expected uint8")
    if labels.shape != (n,):
        raise ValueError(f"{split}: labels shape {labels.shape}, expected {(n,)}")
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError(f"{split}: labels outside [0, {N_CLASSES - 1}]")


def load_pathmnist(
    root: str | os.PathLike = "data",
    download_if_missing: bool = False,
    check_md5: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the official splits as {split: (images uint8 NxHxWxC, labels int64 N)}.

    Checks the file MD5, then shapes, dtype, sample counts and label range.
    """
    path = Path(root) / FILENAME
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found (use download_if_missing=True)")
        download(root)
    if check_md5:
        found = md5sum(path)
        if found != MD5:
            raise RuntimeError(f"MD5 mismatch for {path}: {found} != {MD5}")
    data = {}
    with np.load(path) as npz:
        for split in SPLITS:
            images = npz[f"{split}_images"]
            labels = npz[f"{split}_labels"].reshape(-1).astype(np.int64)
            validate_split(images, labels, split)
            data[split] = (images, labels)
    return data


def class_counts(labels: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Number of samples per class."""
    return np.bincount(labels, minlength=n_classes)


# --------------------------------------------------------------------------- #
# Federated partitions (protocol section 1.2)
# --------------------------------------------------------------------------- #

def _block_sizes(count: int, p: np.ndarray) -> np.ndarray:
    """Sizes of the k blocks obtained when `count` indices are cut by proportions p."""
    cuts = np.round(np.cumsum(p)[:-1] * count).astype(int)
    return np.diff(np.concatenate([[0], cuts, [count]]))


def split_by_proportions(
    labels: np.ndarray, proportions: np.ndarray, rng: np.random.Generator
) -> list[np.ndarray]:
    """Split indices class by class according to `proportions` (n_classes x k).

    For each class c, its indices are shuffled and cut into k consecutive blocks whose
    sizes follow proportions[c]. Returns one sorted index array per client.
    """
    n_classes, k = proportions.shape
    parts: list[list[np.ndarray]] = [[] for _ in range(k)]
    for c in range(n_classes):
        idx = rng.permutation(np.flatnonzero(labels == c))
        cuts = np.cumsum(_block_sizes(len(idx), proportions[c]))[:-1]
        for client, block in enumerate(np.split(idx, cuts)):
            parts[client].append(block)
    return [np.sort(np.concatenate(p)).astype(np.int64) for p in parts]


def make_partition(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    scheme: str,
    k: int = 4,
    alpha: float = 0.5,
    seed: int = 0,
    min_train: int = 2000,
    max_draws: int = 1000,
) -> dict:
    """Build the `IID` or `DIR05` partition of the train and val splits.

    IID: every class is split into k equal parts.
    DIR05: for each class, client proportions ~ Dirichlet(alpha * 1_k) (Hsu et al., 2019).
    The draw is repeated while a client has fewer than `min_train` train images.
    The same proportion matrix is used for train and val, so each client has a local
    validation set with the same class mix, and the union of local sets is the full split.
    """
    n_classes = int(max(train_labels.max(), val_labels.max())) + 1
    rng = np.random.default_rng(seed)
    if scheme == "IID":
        proportions = np.full((n_classes, k), 1.0 / k)
        n_draws = 0
    elif scheme == "DIR05":
        for n_draws in range(1, max_draws + 1):
            proportions = rng.dirichlet(np.full(k, alpha), size=n_classes)
            counts = class_counts(train_labels, n_classes)
            train_sizes = sum(_block_sizes(counts[c], proportions[c]) for c in range(n_classes))
            if train_sizes.min() >= min_train:
                break
        else:
            raise RuntimeError(f"no Dirichlet draw with >= {min_train} images per client")
    else:
        raise ValueError(f"unknown scheme {scheme!r} (expected 'IID' or 'DIR05')")
    return {
        "scheme": scheme,
        "k": k,
        "alpha": alpha if scheme == "DIR05" else None,
        "seed": seed,
        "n_draws": n_draws,
        "proportions": proportions,
        "train": split_by_proportions(train_labels, proportions, rng),
        "val": split_by_proportions(val_labels, proportions, rng),
    }


def check_partition(parts: list[np.ndarray], n: int) -> None:
    """Raise ValueError unless the parts are disjoint and cover range(n) exactly."""
    all_idx = np.concatenate(parts)
    if len(all_idx) != n:
        raise ValueError(f"partition has {len(all_idx)} indices, expected {n}")
    if not np.array_equal(np.sort(all_idx), np.arange(n)):
        raise ValueError("partition parts overlap or do not cover all indices")


def partition_class_counts(parts: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """Matrix (k x n_classes) of class counts per client."""
    n_classes = int(labels.max()) + 1
    return np.stack([class_counts(labels[p], n_classes) for p in parts])


def partition_hash(partition: dict) -> str:
    """SHA-256 of the index arrays (train then val), to record in run files."""
    digest = hashlib.sha256()
    for split in ("train", "val"):
        for p in partition[split]:
            digest.update(np.ascontiguousarray(p, dtype=np.int64).tobytes())
    return digest.hexdigest()


def save_partition(partition: dict, path: str | os.PathLike) -> None:
    """Save a partition as a small .npz file (indices and metadata)."""
    arrays = {f"{split}_{i}": p for split in ("train", "val") for i, p in enumerate(partition[split])}
    np.savez_compressed(
        path,
        proportions=partition["proportions"],
        scheme=np.array(partition["scheme"]),
        k=np.array(partition["k"]),
        seed=np.array(partition["seed"]),
        n_draws=np.array(partition["n_draws"]),
        alpha=np.array(np.nan if partition["alpha"] is None else partition["alpha"]),
        **arrays,
    )


def load_partition(path: str | os.PathLike) -> dict:
    """Load a partition saved by save_partition (no pickle needed)."""
    with np.load(path, allow_pickle=False) as f:
        k = int(f["k"])
        alpha = float(f["alpha"])
        return {
            "scheme": str(f["scheme"]), "k": k, "seed": int(f["seed"]), "n_draws": int(f["n_draws"]),
            "alpha": None if np.isnan(alpha) else alpha, "proportions": f["proportions"],
            "train": [f[f"train_{i}"] for i in range(k)],
            "val": [f[f"val_{i}"] for i in range(k)],
        }


# --------------------------------------------------------------------------- #
# Preprocessing (protocol section 2). Requires PyTorch.
# Upsampling to 224 and ImageNet normalisation are done inside the model (Phase 2).
# --------------------------------------------------------------------------- #

def to_tensor(images: np.ndarray):
    """uint8 N x H x W x C array -> float32 torch tensor N x C x H x W in [0, 1]."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)


def augment(x, generator=None):
    """Training augmentation on a float batch N x C x H x W.

    Independent random horizontal flip (p=0.5), vertical flip (p=0.5) and rotation by
    k * 90 degrees (k uniform in {0,1,2,3}) for each image. These preserve tissue labels.
    """
    import torch

    n = x.shape[0]
    hflip = torch.rand(n, generator=generator, device="cpu") < 0.5
    vflip = torch.rand(n, generator=generator, device="cpu") < 0.5
    rot = torch.randint(0, 4, (n,), generator=generator, device="cpu")
    hflip, vflip, rot = hflip.to(x.device), vflip.to(x.device), rot.to(x.device)
    x = torch.where(hflip[:, None, None, None], x.flip(3), x)
    x = torch.where(vflip[:, None, None, None], x.flip(2), x)
    out = x.clone()
    for k in (1, 2, 3):
        mask = rot == k
        if mask.any():
            out[mask] = torch.rot90(x[mask], k, dims=(2, 3))
    return out
