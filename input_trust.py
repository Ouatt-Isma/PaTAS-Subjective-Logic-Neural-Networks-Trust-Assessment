"""input_trust.py — data-conditioned per-feature input trust opinions.

Turns a raw test sample into a per-feature subjective-logic opinion so the
PaTAS/IPTA feedforward propagates *input-side* trust instead of the constant
fully-trusted opinion used historically (which made the per-sample IPTA score
a constant rescaling of softmax confidence — no per-sample information).

Construction (per feature j, with statistics learned from the training set):

    z_j      = |x_j − μ_j| / max(σ_j, floor)          standardized deviation
    excess_j = max(z_j − slack, 0)                    typical variation is free
    g_j      = exp(−excess_j² / 2)                    conformity in (0, 1]
    w_j      = min((σ_ref/σ_eff_j)^α, w_cap)          witness reliability
    r_j      = N·w_j·g_j,  s_j = N·w_j·(1 − g_j)      evidence split
    ω_j      = BPQ(r_j, s_j) = (r, s, W) / (r+s+W)    opinion  (b, d, u)

and the per-sample input trust pools the evidence of all features
(cumulative fusion of independent evidence sources — evidence masses add):

    P_input  = pp( BPQ(Σ_j r_j, Σ_j s_j) )

The evidence→opinion mapping is subjective_logic.bpq_vec — the *same* mapping
PaTAS uses to turn gradient-stability counts into weight opinions
(TensorTO.theta_given_y), so input trust and model trust live on one scale.

μ/σ are marginal per-feature statistics of the (clean) training features; a
feature that deviates from anything ever seen in training is evidence the
input is corrupted/atypical, graded smoothly rather than thresholded.  The
``floor`` is expressed as a fraction of the feature span, so the model works
unchanged for [0,1]-scaled data (GTSRB, CIFAR-10) and standardized data
(MNIST, range ≈ [−0.42, 2.82]).

The reliability weight w_j is inverse-variance (precision) weighting cast
in evidence terms: a feature whose training distribution is tight is a
reliable witness — corruption there is unambiguous — so its conformity
carries proportionally more evidence mass, while a feature that varies
wildly across training samples (e.g. central MNIST pixels, where a stroke
may or may not pass) is a weak witness either way.  Without it, an
*unweighted* mean over features caps the corruption signal at (fraction of
reliable features) — on MNIST the ~20 % of near-constant border pixels are
the only strong responders to uniform noise, so the unweighted sample trust
moved by ≤0.1 even under 60 % corruption.
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
        detectable.  The default 0.02 keeps a uniform noise bump of
        magnitude ≥ ~6 % of the feature span on a near-constant feature
        far outside the slack zone (strong distrust) while ≤ 1σ of real
        training variation stays fully conforming; the earlier 0.05
        default let about half of the ±0.3·span corruption bumps land
        inside the slack zone, compressing the clean-vs-corrupted
        separation to a few percent.
    slack : float
        Deviations up to ``slack`` standard deviations are considered fully
        conforming (g = 1), so in-distribution variation between samples is
        not penalized; only the excess beyond it erodes conformity.
    evidence : float
        Base evidence mass N per median-reliability feature; feature j
        contributes N·w_j.  With BPQ weight W=2 the per-feature uncertainty
        is u_j = W/(N·w_j + W).
    alpha : float
        Precision-weighting exponent: w_j = (σ_ref/σ_eff_j)^α with σ_ref
        the median effective σ.  α = 2 (default) is inverse-variance
        weighting; α = 0 disables weighting (every feature an equal
        witness — the ablation that caps the corruption signal at the
        fraction of reliable features).
    weight_cap : float
        Upper bound on w_j so a handful of ultra-tight features cannot
        monopolize the pooled evidence when the σ floor is set very low.
    W : float
        Non-informative prior weight of the BPQ mapping (2 is the SL
        convention used throughout patas_module).
    base_rate : float
        Base rate a used for projected probability p = b + a·u.
    """

    def __init__(self, floor_frac: float = 0.02, slack: float = 1.0,
                 evidence: float = 50.0, alpha: float = 2.0,
                 weight_cap: float = 64.0, W: float = 2.0,
                 base_rate: float = 0.5):
        if floor_frac <= 0:
            raise ValueError("floor_frac must be > 0")
        if evidence <= 0:
            raise ValueError("evidence must be > 0")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if weight_cap < 1:
            raise ValueError("weight_cap must be >= 1")
        self.floor_frac = float(floor_frac)
        self.slack = float(slack)
        self.evidence = float(evidence)
        self.alpha = float(alpha)
        self.weight_cap = float(weight_cap)
        self.W = float(W)
        self.base_rate = float(base_rate)
        self.mu: np.ndarray | None = None
        self.sigma_eff: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.span: float | None = None

    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "InputTrustModel":
        """Learn per-feature μ, effective σ and reliability weights from
        training features (n, d). Uses a robust span (0.5th–99.5th
        percentile) so a handful of extreme values can't inflate the σ
        floor."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected (n, d) features, got shape {X.shape}")
        self.mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        lo, hi = np.percentile(X, [0.5, 99.5])
        self.span = float(max(hi - lo, 1e-6))
        floor = self.floor_frac * self.span
        self.sigma_eff = np.maximum(sigma, floor).astype(np.float32)
        sigma_ref = float(np.median(self.sigma_eff))
        w = (sigma_ref / self.sigma_eff) ** self.alpha
        self.weights = np.clip(w, None, self.weight_cap).astype(np.float32)
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
        """Per-feature trust opinions, shape (n, d, 3) float32 [b, d, u].
        Reliable features (large w_j) carry more evidence, hence less
        per-feature uncertainty."""
        g = self.conformity(X)
        n_j = self.evidence * self.weights            # (d,)
        r = n_j * g
        s = n_j * (1.0 - g)
        return bpq_vec(r, s, W=self.W).astype(np.float32)

    def sample_trust(self, X: np.ndarray) -> np.ndarray:
        """Per-sample input-trust scalar, shape (n,): evidence of all
        features pooled (cumulative fusion of independent evidence sources
        — masses add) and mapped through BPQ; the projected probability of
        the pooled opinion.  Equals the reliability-weighted mean
        conformity up to the vanishing prior-weight term."""
        g = self.conformity(X)
        n_j = self.evidence * self.weights            # (d,)
        R = (n_j * g).sum(axis=1)                     # (n,)
        S = (n_j * (1.0 - g)).sum(axis=1)
        pooled = bpq_vec(R, S, W=self.W)              # (n, 3)
        return (pooled[..., 0]
                + self.base_rate * pooled[..., 2]).astype(np.float32)

    # ------------------------------------------------------------------ #

    def describe(self) -> str:
        self._check_fitted()
        w = self.weights
        return (f"InputTrustModel(floor_frac={self.floor_frac:g}, "
                f"slack={self.slack:g}, evidence={self.evidence:g}, "
                f"alpha={self.alpha:g}, weight_cap={self.weight_cap:g}, "
                f"W={self.W:g}, span={self.span:.3f}, "
                f"median σ_eff={float(np.median(self.sigma_eff)):.4f}, "
                f"weights median/max={float(np.median(w)):.2f}/"
                f"{float(w.max()):.1f}, "
                f"top-decile evidence share="
                f"{float(np.sort(w)[-len(w)//10:].sum() / w.sum()):.2f})")
