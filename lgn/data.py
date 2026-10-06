"""Datasets, binarized with thermometer encoding.

A logic network only sees bits, so each pixel becomes one bit per threshold:
with thresholds (4, 8, 12) a pixel of value 9 becomes 1, 1, 0.
"""

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]
MNIST_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


@dataclass
class Dataset:
    name: str
    x_train: np.ndarray  # raw pixel values, (N, H*W)
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    image_shape: tuple
    max_value: float
    n_classes: int = 10


def thermometer(x: np.ndarray, thresholds) -> np.ndarray:
    """(N, D) values -> (N, D * len(thresholds)) bits, threshold-major order."""
    return np.concatenate([(x > t) for t in thresholds], axis=1).astype(np.float32)


def load_digits(seed: int = 0, test_fraction: float = 0.2) -> Dataset:
    """sklearn's 8x8 handwritten digits (1,797 images, pixel values 0-16)."""
    from sklearn.datasets import load_digits as _load
    from sklearn.model_selection import train_test_split

    d = _load()
    x_tr, x_te, y_tr, y_te = train_test_split(
        d.data.astype(np.float32), d.target, test_size=test_fraction,
        random_state=seed, stratify=d.target)
    return Dataset("digits", x_tr, y_tr, x_te, y_te, (8, 8), 16.0)


def _download(fname: str) -> Path:
    path = DATA_DIR / "mnist" / fname
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for base in MNIST_MIRRORS:
        try:
            print(f"downloading {base}{fname}")
            urllib.request.urlretrieve(base + fname, path)
            return path
        except Exception as e:  # try the next mirror
            print(f"  failed: {e}")
            path.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {fname}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, _, ndim = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist() -> Dataset:
    arr = {k: _read_idx(_download(f)) for k, f in MNIST_FILES.items()}
    x_tr = arr["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    x_te = arr["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    return Dataset("mnist", x_tr, arr["y_train"].astype(np.int64),
                   x_te, arr["y_test"].astype(np.int64), (28, 28), 1.0)


DEFAULT_THRESHOLDS = {
    "digits": [4.0, 8.0, 12.0],
    "mnist": [0.5],
}


def load(name: str) -> Dataset:
    if name == "digits":
        return load_digits()
    if name == "mnist":
        return load_mnist()
    raise ValueError(f"unknown dataset {name!r}")
