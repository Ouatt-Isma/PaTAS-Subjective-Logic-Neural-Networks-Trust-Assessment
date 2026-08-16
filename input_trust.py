"""input_trust.py — data-conditioned per-feature input trust opinions.

Turns a raw test sample into a per-feature subjective-logic opinion so the
PaTAS/IPTA feedforward propagates *input-side* trust instead of the constant
fully-trusted opinion used historically (which made the per-sample IPTA score
a constant rescaling of softmax confidence — no per-sample information).

Construction (per feature j, with statistics learned from the training set):

    z_j      = |x_j − μ_j| / max(σ_j, floor)          standardized deviation
    excess_j = max(z_j − slack, 0)                    typical variation is free
    g_j      = exp(−excess_j² / 2)                    conformity in (0, 1]
    r_j      = N·g_j,  s_j = N·(1 − g_j)              evidence split
    ω_j      = BPQ(r_j, s_j) = (r, s, W) / (r+s+W)    opinion  (b, d, u)

The evidence→opinion mapping is subjective_logic.bpq_vec — the *same* mapping
PaTAS uses to turn gradient-stability counts into weight opinions
(TensorTO.theta_given_y), so input trust and model trust live on one scale.

μ/σ are marginal per-feature statistics of the (clean) training features; a
feature that deviates from anything ever seen in training is evidence the
input is corrupted/atypical, graded smoothly rather than thresholded.  The
``floor`` is expressed as a fraction of the feature span, so the model works
unchanged for [0,1]-scaled data (GTSRB, CIFAR-10) and standardized data
(MNIST, range ≈ [−0.42, 2.82]).
"""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_here, "patas_module")
for _p in (_here, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from patas_module.subjective_logic import bpq_vec


class InputTrustModel:
    """Per-feature conformity → subjective-opinion mapping.

    Parameters
    ----------
    floor_frac : float
        Minimum per-feature σ as a fraction of the (robust) feature span.
        Prevents near-constant features (MNIST background) from producing
        infinite z on the smallest perturbation while still flagging them
        quickly — they are exactly the features where corruption is most
        detectable.
    slack : float
        Deviations up to ``slack`` standard deviations are considered fully
        conforming (g = 1), so in-distribution variation between samples is
        not penalized; only the excess beyond it erodes conformity.
    evidence : float
        Total evidence mass N assigned per feature. With BPQ weight W=2 the
        per-feature uncertainty is u = W/(N+W) regardless of conformity —
        i.e. how much a single feature is allowed to commit the opinion.
    W : float
        Non-informative prior weight of the BPQ mapping (2 is the SL
        convention used throughout patas_module).
    base_rate : float
        Base rate a used for projected probability p = b + a·u.
    """

    def __init__(self, floor_frac: float = 0.05, slack: float = 1.0,
                 evidence: float = 50.0, W: float = 2.0,
                 base_rate: float = 0.5):
        if floor_frac <= 0:
            raise ValueError("floor_frac must be > 0")
        if evidence <= 0:
            raise ValueError("evidence must be > 0")
        self.floor_frac = float(floor_frac)
        self.slack = float(slack)
        self.evidence = float(evidence)
        self.W = float(W)
        self.base_rate = float(base_rate)
        self.mu: np.ndarray | None = None
        self.sigma_eff: np.ndarray | None = None
        self.span: float | None = None

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "InputTrustModel":
        """Learn per-feature μ and effective σ from training features
        (n, d). Uses a robust span (0.5th–99.5th percentile) so a handful of
        extreme values can't inflate the σ floor."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected (n, d) features, got shape {X.shape}")
        self.mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        lo, hi = np.percentile(X, [0.5, 99.5])
        self.span = float(max(hi - lo, 1e-6))
        floor = self.floor_frac * self.span
        self.sigma_eff = np.maximum(sigma, floor).astype(np.float32)
        return self

    def _check_fitted(self):
        if self.mu is None:
            raise RuntimeError("InputTrustModel.fit() must be called first")

    # ------------------------------------------------------------------ #

    def conformity(self, X: np.ndarray) -> np.ndarray:
        """Per-feature conformity g ∈ (0, 1], shape (n, d)."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        z = np.abs(X - self.mu) / self.sigma_eff
        excess = np.clip(z - self.slack, 0.0, None)
        return np.exp(-0.5 * excess * excess).astype(np.float32)

    def opinions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature trust opinions, shape (n, d, 3) float32 [b, d, u]."""
        g = self.conformity(X)
        r = self.evidence * g
        s = self.evidence * (1.0 - g)
        return bpq_vec(r, s, W=self.W).astype(np.float32)

    def sample_trust(self, X: np.ndarray) -> np.ndarray:
        """Per-sample input-trust scalar, shape (n,): the projected
        probability b̄ + a·ū of the feature-averaged opinion (average fusion
        across features, matching PTAS.aggregation semantics)."""
        ops = self.opinions(X)
        m = ops.mean(axis=1)                      # (n, 3)
        return (m[:, 0] + self.base_rate * m[:, 2]).astype(np.float32)

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        self._check_fitted()
        return (f"InputTrustModel(floor_frac={self.floor_frac:g}, "
                f"slack={self.slack:g}, evidence={self.evidence:g}, "
                f"W={self.W:g}, span={self.span:.3f}, "
                f"median σ_eff={float(np.median(self.sigma_eff)):.4f})")
