"""Unit tests for the data-conditioned input-trust machinery.

Fast, deterministic, numpy-only at the core (no torch, no datasets, no
sockets) — safe to run before spending cluster time:

    pytest tests/test_input_trust.py -q

Covers:
  * InputTrustModel — opinion validity, precision weighting, pooled sample
    trust, corruption response ordering, and the unweighted (α=0) ablation
    being strictly weaker than α=2.
  * GenIPTA propagation (numpy backend) — data-conditioned input opinions
    produce ordered path trust.
  * The belief-anchored discounted-confidence score and the FPR@95 metric
    (skipped automatically when torch / scikit-learn are unavailable, since
    importing run_uq_comparison / uq_methods pulls them in).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_v2_dir, os.path.join(_v2_dir, "patas_module")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from input_trust import InputTrustModel  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data: standardized MNIST-like feature profile (mix of near-dead,
# quiet and active features), calibrated so the unweighted model reproduces
# the behaviour measured on the real cluster runs.
# ─────────────────────────────────────────────────────────────────────────────

N_DEAD, N_QUIET, N_ACTIVE = 40, 100, 60
LO, HI = -0.4243, 2.82


def make_profile(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    sig = np.concatenate([
        np.full(N_DEAD, 0.02),
        rng.uniform(0.15, 0.60, N_QUIET),
        rng.uniform(0.70, 1.40, N_ACTIVE),
    ])
    mu = np.concatenate([
        np.full(N_DEAD, LO),
        np.full(N_QUIET, -0.30),
        np.full(N_ACTIVE, 0.40),
    ])
    X = mu + sig * rng.standard_normal((n, len(sig)))
    return np.clip(X, LO, HI).astype(np.float32)


def bump(X: np.ndarray, p: float, seed: int = 1,
         scale: float = 0.3) -> np.ndarray:
    """Range-aware Bernoulli-uniform corruption (mirrors
    uq_methods.apply_feature_noise without importing torch)."""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    span = float(X.max() - X.min())
    mask = rng.random(X.shape) < p
    Xn[mask] += ((rng.random(int(mask.sum())) * 2 - 1)
                 * scale * span).astype(np.float32)
    return np.clip(Xn, X.min(), X.max())


@pytest.fixture(scope="module")
def fitted():
    X = make_profile()
    return X, InputTrustModel().fit(X)


# ─────────────────────────────────────────────────────────────────────────────
# InputTrustModel
# ─────────────────────────────────────────────────────────────────────────────

def test_opinions_are_valid(fitted):
    X, itm = fitted
    ops = itm.opinions(X[:32])
    assert ops.shape == (32, X.shape[1], 3)
    assert np.allclose(ops.sum(-1), 1.0, atol=1e-5)
    assert ops.min() >= -1e-6 and ops.max() <= 1 + 1e-6


def test_reliable_features_get_more_evidence(fitted):
    X, itm = fitted
    ops = itm.opinions(X[:8])
    u = ops[..., 2]
    # dead features carry more evidence → less per-feature uncertainty
    assert u[:, :N_DEAD].mean() < u[:, -N_ACTIVE:].mean()


def test_sample_trust_orders_corruption_levels(fitted):
    X, itm = fitted
    s0 = itm.sample_trust(X[:100]).mean()
    s3 = itm.sample_trust(bump(X[:100], 0.3)).mean()
    s6 = itm.sample_trust(bump(X[:100], 0.6)).mean()
    assert s0 > 0.9, "clean data must keep high input trust"
    assert s0 > s3 > s6, "trust must degrade monotonically with corruption"
    assert s0 - s6 > 0.1, "the corruption response must be substantial"


def test_precision_weighting_beats_unweighted(fitted):
    X, itm = fitted
    itm0 = InputTrustModel(alpha=0.0).fit(X)
    gap = (itm.sample_trust(X[:100]).mean()
           - itm.sample_trust(bump(X[:100], 0.6)).mean())
    gap0 = (itm0.sample_trust(X[:100]).mean()
            - itm0.sample_trust(bump(X[:100], 0.6)).mean())
    assert gap > 1.5 * gap0


def test_weight_cap_respected():
    X = make_profile()
    itm = InputTrustModel(weight_cap=4.0).fit(X)
    assert float(itm.weights.max()) <= 4.0 + 1e-6


def test_fit_input_validation():
    with pytest.raises(ValueError):
        InputTrustModel(floor_frac=0.0)
    with pytest.raises(ValueError):
        InputTrustModel(alpha=-1.0)
    with pytest.raises(RuntimeError):
        InputTrustModel().sample_trust(np.zeros((2, 3), dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# GenIPTA propagation (numpy backend, no torch required)
# ─────────────────────────────────────────────────────────────────────────────

def test_ipta_propagation_orders_input_trust(fitted):
    import io
    import contextlib
    from NN.PTAStemplate import PTAS
    from concrete.TensorTO import TensorArrayTO, fill as tfill

    X, itm = fitted
    d, hidden, k = X.shape[1], 8, 5
    om = []
    for shape in ((d + 1, hidden), (hidden + 1, k)):
        v = np.zeros(shape + (3,), dtype=np.float32)
        v[..., 0], v[..., 2] = 0.97, 0.03
        om.append(TensorArrayTO(v))
    ptas = PTAS(om, operator_mapping=None, nn_interface=None,
                trust_assessment_func=None, structure=[d, hidden, k],
                epsilon_low=0.05, eval=False)
    path = [[1] * (hidden // 2) + [0] * (hidden - hidden // 2)]
    with contextlib.redirect_stdout(io.StringIO()):
        ipta = ptas.GenIPTA(path)
        pp = []
        for Xi in (None, X[:1], bump(X[:1], 0.6)):
            Tx = (TensorArrayTO(tfill((1, d), method="trust")) if Xi is None
                  else TensorArrayTO(itm.opinions(Xi)))
            v = ipta(Tx).to_numpy()
            assert v.shape == (1, k, 3)
            assert np.allclose(v.sum(-1), 1.0, atol=1e-4)
            pp.append(float(v[0, 0, 0] + 0.5 * v[0, 0, 2]))
    p_const, p_clean, p_noised = pp
    assert p_const >= p_clean - 1e-6 > p_noised


# ─────────────────────────────────────────────────────────────────────────────
# Score / metric helpers (need torch / sklearn via their home modules)
# ─────────────────────────────────────────────────────────────────────────────

def test_discounted_confidence_endpoints():
    pytest.importorskip("torch")
    from run_uq_comparison import _discounted_confidence
    conf = np.array([0.95, 0.60])
    assert np.allclose(_discounted_confidence(np.ones(2), conf, 10), conf)
    assert np.allclose(_discounted_confidence(np.zeros(2), conf, 10), 0.1)
    lo = _discounted_confidence(np.array([0.4]), np.array([0.9]), 10)
    hi = _discounted_confidence(np.array([0.9]), np.array([0.9]), 10)
    assert hi[0] > lo[0]


def test_fpr_at_tpr_separable():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from uq_methods import fpr_at_tpr
    scores = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
    labels = np.r_[np.ones(50), np.zeros(50)]
    assert fpr_at_tpr(scores, labels) == 0.0
