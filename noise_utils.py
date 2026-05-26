"""noise_utils.py
Noise injection and trust-opinion mapping for the 5G use-case evaluation.

Design rationale
----------------
Measurement noise (features) reduces certainty without implying active
deception -> uncertainty (u) grows, disbelief (d) stays small.

Label mislabelling actively contradicts ground truth
-> disbelief (d) grows, uncertainty (u) stays small.

Both use a saturating map  x / (x + alpha)  that is monotone on [0, inf)
and maps to [0, 1).
"""
from __future__ import annotations

import numpy as np
from patas_module.concrete.TrustOpinion import TrustOpinion

_FEAT_SAT   = 0.30   # half-saturation for feature noise
_LABEL_SAT  = 0.30   # half-saturation for label flip rate
_FEAT_D_K   = 0.10   # disbelief fraction for feature uncertainty
_LABEL_U_K  = 0.20   # uncertainty fraction for label disbelief


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def add_feature_noise(
    X: np.ndarray,
    sigma_relative: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled by each feature's empirical std.

    Parameters
    ----------
    X              : (n, d) float32 feature matrix
    sigma_relative : noise std as a fraction of each feature's std (SNR proxy)
    rng            : optional Generator for reproducibility
    """
    if sigma_relative == 0.0:
        return X.copy()
    if rng is None:
        rng = np.random.default_rng(0)
    per_feat_std = X.std(axis=0) + 1e-8
    noise = rng.normal(0.0, sigma_relative * per_feat_std, X.shape)
    return (X + noise).astype(np.float32)


def add_label_noise(
    y_int: np.ndarray,
    flip_rate: float,
    n_classes: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly relabel *flip_rate* fraction of samples to a wrong class.

    The replacement class is drawn uniformly from all classes excluding the
    true class, ensuring the label is always incorrect.

    Parameters
    ----------
    y_int      : (n,) integer class labels
    flip_rate  : fraction of samples to mislabel in [0, 1)
    n_classes  : total number of classes
    rng        : optional Generator for reproducibility
    """
    if flip_rate == 0.0:
        return y_int.copy()
    if rng is None:
        rng = np.random.default_rng(0)
    y_noisy = y_int.copy()
    n_flip = int(len(y_int) * flip_rate)
    flip_idx = rng.choice(len(y_int), n_flip, replace=False)
    for i in flip_idx:
        wrong = [c for c in range(n_classes) if c != int(y_int[i])]
        y_noisy[i] = int(rng.choice(wrong))
    return y_noisy


# ---------------------------------------------------------------------------
# Trust-opinion mapping
# ---------------------------------------------------------------------------

def trust_quant(x: float) -> TrustOpinion:
    """Map a quantile in [0, 1] to a trust opinion with belief = quantile."""

    return TrustOpinion(1-x, 0.0, x)

def feature_noise_to_trust(sigma_relative: float) -> TrustOpinion:
    """Map a relative noise level to an uncertainty-dominated trust opinion.

    Formula:  u = sigma_relative^2 / (1+sigma_relative^2),  r = 1-sigma_relative,  s = sigma_relative
    Returns TrustOpinion(1, 0, 0) for sigma = 0 (clean features).
    """
    if sigma_relative == 0.0:
        return TrustOpinion(1.0, 0.0, 0.0)
    
    u = 2*(sigma_relative**2 / (1 + sigma_relative**2))
    r = 1 - sigma_relative
    s = sigma_relative
    gamma = (1-u)/(r+s)
    b = r * gamma
    d = s * gamma
    return TrustOpinion(b, d, u)


def label_noise_to_trust(flip_rate: float) -> TrustOpinion:
    """Map a label flip rate to a disbelief-dominated trust opinion."""
    if flip_rate == 0.0:
        return TrustOpinion(1.0, 0.0, 0.0)
    p = flip_rate
    return TrustOpinion(1-p, p, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trust_str(opinion) -> str:
    """Compact, filename-safe label for a TrustOpinion or string type."""
    if isinstance(opinion, str):
        return opinion
    return f"b{opinion.t:.3f}_d{opinion.d:.3f}_u{opinion.u:.3f}"


def trust_components(opinion) -> tuple[float, float, float]:
    """Return (b, d, u) tuple from a TrustOpinion or a named string."""
    if isinstance(opinion, str):
        _MAP = {
            "trusted":    (1.0, 0.0, 0.0),
            "ftrust":     (1.0, 0.0, 0.0),
            "vacuous":    (0.0, 0.0, 1.0),
            "distrusted": (0.0, 1.0, 0.0),
            "fdistrust":  (0.0, 1.0, 0.0),
            "percal":     (float("nan"), float("nan"), float("nan")),
        }
        return _MAP.get(opinion, (1.0, 0.0, 0.0))
    return float(opinion.t), float(opinion.d), float(opinion.u)


# ---------------------------------------------------------------------------
# Per-sample noise injection and trust calibration
# ---------------------------------------------------------------------------

def add_feature_noise_per_sample(
    X: np.ndarray,
    sigma_max: float,
    rng: "np.random.Generator | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Add per-sample Gaussian noise with independently sampled noise levels.

    Each sample i receives noise scaled by σ_i ~ Uniform(0, sigma_max).

    Returns
    -------
    X_noisy   : noisy feature matrix, same shape as X, dtype float32
    sigma_arr : per-sample noise levels σ_i in [0, sigma_max], shape (n,)
    """
    if sigma_max == 0.0:
        return X.copy(), np.zeros(len(X), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(0)
    n = X.shape[0]
    sigma_arr = rng.uniform(0.0, sigma_max, size=n).astype(np.float32)
    per_feat_std = X.std(axis=0) + 1e-8
    noise = (rng.normal(0.0, 1.0, X.shape).astype(np.float32)
             * (sigma_arr[:, None] * per_feat_std))
    return (X + noise).astype(np.float32), sigma_arr


def build_per_sample_feature_trust_generator(sigma_arr: np.ndarray):
    """Return a (indices_or_n, dim) -> ArrayTO generator with per-sample trust.

    The NN client sends batch *indices* (not actual feature values) in the
    TRAINING_FEEDFORWARD message, so sigma_arr is indexed directly.

    Parameters
    ----------
    sigma_arr : per-sample noise levels (n_train,) from add_feature_noise_per_sample.

    Returns
    -------
    gen(indices_or_n, dim) -> ArrayTO
        Each row gets the trust opinion calibrated to its actual noise level.
        Falls back to fully-trusted when given a plain int (legacy path).
    """
    _sigma = np.asarray(sigma_arr, dtype=np.float64)

    def gen(indices_or_n, dim: int):
        from patas_module.concrete.ArrayTO import ArrayTO

        if isinstance(indices_or_n, (int, np.integer)):
            n = int(indices_or_n)
            return ArrayTO(TrustOpinion.fill(shape=(n, dim), method="ftrust"))

        indices = np.asarray(indices_or_n, dtype=int)
        n = len(indices)
        opinions = np.empty((n, dim), dtype=object)
        for i, idx in enumerate(indices):
            op = feature_noise_to_trust(float(_sigma[idx]))
            for j in range(dim):
                opinions[i, j] = op
        return ArrayTO(opinions)

    return gen
