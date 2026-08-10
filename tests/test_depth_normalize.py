"""
Unit tests for depth-normalized trust opinions.

Run standalone (no pytest needed):
    python tests/test_depth_normalize.py

Run with pytest:
    pytest tests/test_depth_normalize.py -s

Depth normalization undoes the geometric depth decay of PaTAS trust
propagation: each layer scales belief and disbelief by the same discount
factor p <= 1, so the committed mass m = b + d decays geometrically with
depth L while the b:d ratio stays depth-invariant.  The normalized opinion

    m' = (b + d)^(1/L),  b' = m'*b/m,  d' = m'*d/m,  u' = 1 - m'

is therefore comparable across architectures of different depth
("trust retention per processing step").
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap (works with or without pip install) ────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # triggers patas_module/__init__.py path bootstrap
from patas_module.subjective_logic import (
    Opinion,
    depth_normalize,
    depth_normalize_vec,
)

try:
    import torch
except ImportError:
    torch = None

ATOL = 1e-6


def test_identity_at_L1():
    w = Opinion(0.5, 0.3, 0.2)
    out = depth_normalize(w, 1)
    assert abs(out.b - 0.5) < ATOL
    assert abs(out.d - 0.3) < ATOL
    assert abs(out.u - 0.2) < ATOL


def test_worked_example_L3():
    # b=0.5, d=0.3, u=0.2, L=3:  m'=0.8^(1/3), split at the original 5:3 ratio
    out = depth_normalize(Opinion(0.5, 0.3, 0.2), 3)
    m_norm = 0.8 ** (1.0 / 3.0)
    assert abs(out.b - m_norm * 0.5 / 0.8) < ATOL
    assert abs(out.d - m_norm * 0.3 / 0.8) < ATOL
    assert abs(out.u - (1.0 - m_norm)) < ATOL


def test_vacuous_stays_vacuous():
    for L in (1, 3, 10):
        out = depth_normalize(Opinion(0.0, 0.0, 1.0), L)
        assert abs(out.b) < ATOL
        assert abs(out.d) < ATOL
        assert abs(out.u - 1.0) < ATOL


def test_depth_invariance():
    # A network retaining p per layer yields raw b = p^L from a trusted
    # input; the normalized belief must recover p regardless of depth.
    p = 0.9
    for L in (1, 3, 10, 50):
        raw = Opinion(p ** L, 0.0, 1.0 - p ** L)
        out = depth_normalize(raw, L)
        assert abs(out.b - p) < ATOL, f"L={L}: {out.b} != {p}"


def test_bd_ratio_preserved():
    w = Opinion(0.10, 0.05, 0.85)
    out = depth_normalize(w, 10)
    assert abs(out.b / out.d - w.b / w.d) < 1e-4
    # Balanced conflict must stay balanced (the flaw in b-only rooting)
    conflicted = depth_normalize(Opinion(0.1, 0.1, 0.8), 10)
    assert abs(conflicted.b - conflicted.d) < ATOL


def test_valid_opinion():
    rng = np.random.default_rng(0)
    for _ in range(100):
        b, d = rng.dirichlet([1, 1, 1])[:2]
        for L in (1, 2, 5, 20):
            out = depth_normalize(Opinion(b, d, 1.0 - b - d), L)
            assert min(out.b, out.d, out.u) >= -ATOL
            assert abs(out.b + out.d + out.u - 1.0) < 1e-5


def test_invalid_L_raises():
    for fn, arg in ((depth_normalize, Opinion(0.5, 0.3, 0.2)),
                    (depth_normalize_vec, np.array([0.5, 0.3, 0.2]))):
        try:
            fn(arg, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted L=0")


def _example_grid() -> np.ndarray:
    return np.array(
        [[[0.5, 0.3, 0.2], [0.0, 0.0, 1.0]],
         [[0.9, 0.1, 0.0], [0.01, 0.02, 0.97]]],
        dtype=np.float64,
    )


def test_vec_matches_scalar():
    ops = _example_grid()
    for L in (1, 3, 10):
        out = depth_normalize_vec(ops, L)
        assert out.shape == ops.shape
        for idx in np.ndindex(ops.shape[:-1]):
            b, d, u = ops[idx]
            expected = depth_normalize(Opinion(b, d, u), L)
            got = out[idx]
            assert abs(got[0] - expected.b) < ATOL, (idx, L)
            assert abs(got[1] - expected.d) < ATOL, (idx, L)
            assert abs(got[2] - expected.u) < ATOL, (idx, L)


def test_vec_torch_matches_numpy():
    if torch is None:
        print("torch not installed — skipping torch backend test")
        return
    ops = _example_grid()
    for L in (1, 3, 10):
        out_np = depth_normalize_vec(ops, L)
        out_t = depth_normalize_vec(torch.tensor(ops), L)
        assert isinstance(out_t, torch.Tensor)
        assert np.allclose(out_t.numpy(), out_np, atol=ATOL)


def test_reexported_via_tensorto():
    from concrete.TensorTO import depth_normalize_vec as via_tensorto
    assert via_tensorto is depth_normalize_vec


ALL_TESTS = [
    test_identity_at_L1,
    test_worked_example_L3,
    test_vacuous_stays_vacuous,
    test_depth_invariance,
    test_bd_ratio_preserved,
    test_valid_opinion,
    test_invalid_L_raises,
    test_vec_matches_scalar,
    test_vec_torch_matches_numpy,
    test_reexported_via_tensorto,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(ALL_TESTS)} tests passed.")
