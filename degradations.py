"""
Data-degradation utilities used by the §7.8 experiments.

Implements the perturbation functions described in the dissertation:

    * Fully-uncertain features  -> additive uniform noise on a random subset
                                   of features (clipped to the feature range).
    * Fully-distrusted features -> replace by uniform random samples drawn
                                   from the feature range.
    * Fully-uncertain labels    -> random relabelling on a fraction of rows.
    * Fully-distrusted labels   -> complete random relabelling.
    * Patch injection           -> for image-shaped inputs, paste a constant
                                   patch in the top-left corner of a subset
                                   of rows AND flip a pair of labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


def add_uniform_noise(X: np.ndarray, *, eta_frac: float = 0.3,
                      prob: float = 0.3, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """§7.8.1 "uncertain features" model."""
    rng = np.random.default_rng(seed)
    X_out = X.astype(np.float32).copy()
    x_max, x_min = float(np.max(X)), float(np.min(X))
    eta = eta_frac * max(abs(x_max), abs(x_min), 1.0)
    mask = rng.uniform(size=X.shape) < prob
    noise = rng.uniform(-eta, eta, size=X.shape).astype(np.float32) * mask
    X_out = np.clip(X_out + noise, x_min, x_max)
    return X_out, mask


def replace_with_uniform(X: np.ndarray, *, prob: float = 1.0,
                         seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """§7.8.1 "distrusted features" model."""
    rng = np.random.default_rng(seed)
    X_out = X.astype(np.float32).copy()
    x_max, x_min = float(np.max(X)), float(np.min(X))
    mask = rng.uniform(size=X.shape) < prob
    sub = rng.uniform(x_min, x_max, size=X.shape).astype(np.float32)
    return np.where(mask, sub, X_out), mask


def add_label_noise(y: np.ndarray, n_classes: int, *, prob: float = 0.3,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Uncertain labels: random relabelling on a fraction of rows."""
    rng = np.random.default_rng(seed)
    y_out = y.astype(np.int64).copy()
    mask = rng.uniform(size=y.shape) < prob
    repl = rng.integers(0, n_classes, size=y.shape)
    return np.where(mask, repl, y_out).astype(np.int64), mask


def replace_labels_random(y: np.ndarray, n_classes: int, *, seed: int = 0
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Distrusted labels: every label replaced by uniform random."""
    rng = np.random.default_rng(seed)
    repl = rng.integers(0, n_classes, size=y.shape)
    return repl.astype(np.int64), np.ones_like(y, dtype=bool)


@dataclass
class PoisoningInfo:
    poisoned_rows: np.ndarray
    patch_pixels:  np.ndarray
    label_flipped: np.ndarray


def inject_patch(X: np.ndarray, y: np.ndarray, *,
                 image_shape: Tuple[int, int],
                 patch_size: int,
                 frac_poisoned: float = 1.0 / 3.0,
                 source_label: int = 6, target_label: int = 9,
                 patch_value: float | None = None,
                 seed: int = 0
                 ) -> Tuple[np.ndarray, np.ndarray, PoisoningInfo]:
    """Poisoned MNIST attack (§7.8.1 Experiment 3)."""
    rng = np.random.default_rng(seed)
    H, W = image_shape
    if patch_value is None:
        patch_value = float(np.max(X))

    rr, cc = np.meshgrid(np.arange(patch_size), np.arange(patch_size),
                         indexing="ij")
    patch_idx = (rr * W + cc).ravel()
    patch_mask = np.zeros(H * W, dtype=bool)
    patch_mask[patch_idx] = True

    target_rows = np.isin(y, [source_label, target_label])
    cand = np.where(target_rows)[0]
    n_poison = min(int(frac_poisoned * len(y)), len(cand))
    poison_idx = rng.choice(cand, size=n_poison, replace=False)

    poisoned_rows = np.zeros(len(y), dtype=bool)
    poisoned_rows[poison_idx] = True
    X_out = X.astype(np.float32).copy()
    X_out[poison_idx[:, None], patch_idx[None, :]] = patch_value
    y_out = y.astype(np.int64).copy()
    flip = {source_label: target_label, target_label: source_label}
    for r in poison_idx:
        y_out[r] = flip[int(y[r])]

    return X_out, y_out, PoisoningInfo(poisoned_rows=poisoned_rows,
                                       patch_pixels=patch_mask,
                                       label_flipped=poisoned_rows.copy())
