"""
Subjective Logic (SL) operators and binomial opinion data structure.

NOTE: The **canonical** implementation of this module lives at
``patas_module/subjective_logic.py``.  This root-level copy is kept only for
scripts (e.g. patas.py) that pre-date the integration of SL into patas_module.
If you are writing new code, import from the package instead:

    from patas_module.subjective_logic import Opinion, fuse_many, ...
    # or
    from patas_module import Opinion, fuse_many, ...

Implements the operators required by PaTAS-TP (Chapter 7 of the dissertation):
    * Trust discounting          (⊗)
    * Cumulative belief fusion   (⊕ cumulative)    [CBF / aCBF]
    * Averaging belief fusion    (⊕ averaging)     [ABF]  — used for NN sums per §7.8.1
    * Weighted belief fusion     (⊕ weighted)      [WBF]  — Definition 4 of [van der Heijden 2018]
    * Consensus & Compromise     (⊕ ccf)           [CCF]  — Definition 5 of [van der Heijden 2018]
    * Trust revision             (⊖)
    * Binomial multiplication    (⊙)
    * Conservative division      (⊘) — Eq. (7.7)
    * Inferential deduction      (⊚)
    * BPQ / EWQ / CUQ quantification schemes (Eq. 6.3-6.5)

All opinions are *binomial* and stored as a 4-tuple (b, d, u, a) with
b + d + u = 1 and a ∈ [0, 1] being the base rate (default 0.5).

Multi-source fusion (fuse_many / consensus_compromise_fusion / weighted_belief_fusion)
-----------------------------------------------------------------------
All multi-source operators follow the *direct* N-source formulas from:

    A. Jøsang, D. Wang, J. Zhang, "Multi-Source Fusion in Subjective Logic,"
    FUSION 2017.

and the corrections / extensions in:

    R. W. van der Heijden, H. Kopp, F. Kargl,
    "Multi-Source Fusion Operations in Subjective Logic," arXiv 1805.01388, 2018.

Key correctness notes
---------------------
* ABF is **not** associative, so sequential binary application is WRONG for N>2
  sources; this module uses the direct N-source formula (verified against
  Table I of [van der Heijden 2018]).
* CBF *is* effectively associative in the non-dogmatic case, and the direct
  N-source formula is used here for consistency.
* Dogmatic-handling edge cases follow the corrections in §III-A of
  [van der Heijden 2018]: when ≥2 sources are dogmatic (u=0) and ≥1 is
  non-dogmatic, the non-dogmatic opinions are discarded and only the
  dogmatic ones are averaged.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
import numpy as np

EPS = 1e-12


@dataclass
class Opinion:
    """Binomial subjective opinion ω = (b, d, u, a)."""
    b: float = 0.0
    d: float = 0.0
    u: float = 1.0
    a: float = 0.5

    def __post_init__(self):
        # numerical safety + normalisation
        self.b = max(0.0, min(1.0, float(self.b)))
        self.d = max(0.0, min(1.0, float(self.d)))
        self.u = max(0.0, min(1.0, float(self.u)))
        s = self.b + self.d + self.u
        if s > 0 and abs(s - 1.0) > 1e-6:
            self.b, self.d, self.u = self.b / s, self.d / s, self.u / s

    # projected probability (probability expectation)
    @property
    def P(self) -> float:
        return self.b + self.a * self.u

    def as_tuple(self):
        return (self.b, self.d, self.u, self.a)

    def __repr__(self):
        return f"ω(b={self.b:.3f}, d={self.d:.3f}, u={self.u:.3f}, a={self.a:.2f})"


# ------------------------------------------------------------------ #
#  Canonical opinion shortcuts                                       #
# ------------------------------------------------------------------ #
def trusted(a: float = 0.5)    -> Opinion: return Opinion(1.0, 0.0, 0.0, a)
def distrusted(a: float = 0.5) -> Opinion: return Opinion(0.0, 1.0, 0.0, a)
def vacuous(a: float = 0.5)    -> Opinion: return Opinion(0.0, 0.0, 1.0, a)


# ------------------------------------------------------------------ #
#  Quantification schemes (Chapter 6, Eq. 6.3-6.5)                   #
# ------------------------------------------------------------------ #
def bpq(r: float, s: float, W: float = 2.0, a: float = 0.5) -> Opinion:
    """Baseline-Prior Quantification."""
    denom = W + r + s
    return Opinion(r / denom, s / denom, W / denom, a)


def ewq(r: float, s: float, w: float, a: float = 0.5) -> Opinion:
    """Evidence-Weighted Quantification."""
    denom = w + r + s + EPS
    return Opinion(r / denom, s / denom, w / denom, a)


def cuq(r: float, s: float, U: float, a: float = 0.5) -> Opinion:
    """Constant-Uncertainty Quantification."""
    U = max(EPS, min(1.0, U))
    total = r + s
    if total <= 0:
        return Opinion(0.0, 0.0, 1.0, a)
    gamma = (1.0 - U) / total
    return Opinion(gamma * r, gamma * s, U, a)


# ------------------------------------------------------------------ #
#  Trust discounting  (⊗) -- referral discounting                    #
#  ω_x^{A;B} = ω_A^B ⊗ ω_x^B                                         #
# ------------------------------------------------------------------ #
def discount(omega_A: Opinion, omega_x: Opinion) -> Opinion:
    """
    Standard probability-sensitive trust discounting.
    Trust mass of A scales the (b,d) of x; the rest goes to uncertainty.
    """
    p = omega_A.b  # use trust mass of A as the discount factor
    b = p * omega_x.b
    d = p * omega_x.d
    u = 1.0 - b - d
    return Opinion(b, d, u, omega_x.a)


# ------------------------------------------------------------------ #
#  Binary fusion operators (two-source, exact)                       #
# ------------------------------------------------------------------ #
def cumulative_fusion(w1: Opinion, w2: Opinion) -> Opinion:
    """
    Aleatory cumulative belief fusion (aCBF) for two sources [Jøsang].

    Correct per [van der Heijden 2018] §III-A for the binary case:
    - Both dogmatic (u1=u2=0): average.
    - Otherwise: standard CBF formula.
    """
    u1, u2 = w1.u, w2.u
    if u1 < EPS and u2 < EPS:                      # Case I: both dogmatic
        b = 0.5 * (w1.b + w2.b)
        d = 0.5 * (w1.d + w2.d)
        return Opinion(b, d, max(0.0, 1.0 - b - d), 0.5 * (w1.a + w2.a))
    denom = u1 + u2 - u1 * u2                      # Case II: ≥1 non-dogmatic
    b = (w1.b * u2 + w2.b * u1) / denom
    d = (w1.d * u2 + w2.d * u1) / denom
    u = u1 * u2 / denom
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


def averaging_fusion(w1: Opinion, w2: Opinion) -> Opinion:
    """
    Averaging belief fusion (ABF) for two sources [Jøsang].

    Correct formula:
    - Both dogmatic (u1=u2=0): average.
    - Otherwise: b=(b1u2+b2u1)/(u1+u2), u=2u1u2/(u1+u2).

    Note: the denominator for BOTH b and u is (u1+u2), NOT (u1+u2−2u1u2).
    The latter is the WBF denominator and is a common mis-transcription.
    """
    u1, u2 = w1.u, w2.u
    if (u1 + u2) < EPS:                            # Case I: both dogmatic
        b = 0.5 * (w1.b + w2.b)
        d = 0.5 * (w1.d + w2.d)
        return Opinion(b, d, max(0.0, 1.0 - b - d), 0.5 * (w1.a + w2.a))
    b = (w1.b * u2 + w2.b * u1) / (u1 + u2)       # Case II: ≥1 non-dogmatic
    d = (w1.d * u2 + w2.d * u1) / (u1 + u2)
    u = 2.0 * u1 * u2 / (u1 + u2)
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


# ------------------------------------------------------------------ #
#  Multi-source fusion helpers                                       #
# ------------------------------------------------------------------ #
def _prod_except(us: List[float], k: int) -> float:
    """∏_{j ≠ k} us[j]  (product of all elements except index k)."""
    result = 1.0
    for j, u in enumerate(us):
        if j != k:
            result *= u
    return result


def _fuse_cbf_multi(opinions: List[Opinion]) -> Opinion:
    """
    Direct N-source aleatory CBF (van der Heijden 2018, §III-A corrected).

    Three cases, matching the corrected multi-source definition:
    - Case I  (all dogmatic, u=0):            average all opinions.
    - Case II (≥1 dogmatic, ≥1 non-dogmatic): discard non-dogmatic, average dogmatic.
    - Case III (all non-dogmatic):            direct N-source formula.

    Formula (Case III, binomial):
        denom = Σ_A [∏_{A'≠A} u_A'] − (N−1)·∏_A u_A
        b     = Σ_A [b_A · ∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A · ∏_{A'≠A} u_A'] / denom
        u     = ∏_A u_A / denom
    """
    n = len(opinions)
    us = [w.u for w in opinions]

    # Case I: all dogmatic
    if all(u < EPS for u in us):
        b = sum(w.b for w in opinions) / n
        d = sum(w.d for w in opinions) / n
        a = sum(w.a for w in opinions) / n
        return Opinion(b, d, max(0.0, 1.0 - b - d), a)

    # Case II: mixed dogmatic/non-dogmatic — discard non-dogmatic
    dog_ops = [w for w in opinions if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b = sum(w.b for w in dog_ops) / nd
        d = sum(w.d for w in dog_ops) / nd
        a = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case III: all non-dogmatic
    prod_u      = float(np.prod(us))
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    denom       = sum(prods_excl) - (n - 1) * prod_u
    if denom < EPS:
        return vacuous()
    b = sum(opinions[k].b * prods_excl[k] for k in range(n)) / denom
    d = sum(opinions[k].d * prods_excl[k] for k in range(n)) / denom
    u = prod_u / denom
    a = sum(w.a for w in opinions) / n
    return Opinion(b, d, max(0.0, u), a)


def _fuse_abf_multi(opinions: List[Opinion]) -> Opinion:
    """
    Direct N-source ABF (van der Heijden 2018, §III-A corrected).

    ABF is *not* associative, so sequential binary application gives wrong
    results for N > 2.  This uses the correct direct formula.

    Three cases (same structure as CBF):
    - Case I  (all dogmatic):                 average all.
    - Case II (≥1 dogmatic, ≥1 non-dogmatic): discard non-dogmatic, average dogmatic.
    - Case III (all non-dogmatic):            direct N-source formula.

    Formula (Case III, binomial):
        denom = Σ_A [∏_{A'≠A} u_A']
        b     = Σ_A [b_A · ∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A · ∏_{A'≠A} u_A'] / denom
        u     = N · ∏_A u_A / denom
        a     = (1/N) Σ_A a_A          (simple average)
    """
    n = len(opinions)
    us = [w.u for w in opinions]

    # Case I: all dogmatic
    if all(u < EPS for u in us):
        b = sum(w.b for w in opinions) / n
        d = sum(w.d for w in opinions) / n
        a = sum(w.a for w in opinions) / n
        return Opinion(b, d, max(0.0, 1.0 - b - d), a)

    # Case II: mixed — discard non-dogmatic
    dog_ops = [w for w in opinions if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b = sum(w.b for w in dog_ops) / nd
        d = sum(w.d for w in dog_ops) / nd
        a = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case III: all non-dogmatic
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    denom       = sum(prods_excl)
    if denom < EPS:
        return vacuous()
    b = sum(opinions[k].b * prods_excl[k] for k in range(n)) / denom
    d = sum(opinions[k].d * prods_excl[k] for k in range(n)) / denom
    u = n * float(np.prod(us)) / denom
    a = sum(w.a for w in opinions) / n
    return Opinion(b, d, max(0.0, u), a)


# ------------------------------------------------------------------ #
#  Weighted Belief Fusion  (WBF) — Definition 4                      #
# ------------------------------------------------------------------ #
def weighted_belief_fusion(opinions: Iterable[Opinion]) -> Opinion:
    """
    Direct N-source weighted belief fusion (WBF) — Definition 4 of
    [van der Heijden 2018].

    Three cases:
    - Case 1 (all u_A ≠ 0, ∃A: u_A ≠ 1):  direct N-source formula.
    - Case 2 (∃A: u_A = 0):                combine only dogmatic opinions
                                            with equal weights (γ_A = 1/|A^dog|).
    - Case 3 (∀A: u_A = 1):                fully vacuous result.

    Formula (Case 1, binomial):
        denom = [Σ_A ∏_{A'≠A} u_A'] − N · ∏_A u_A
        b     = Σ_A [b_A·(1−u_A)·∏_{A'≠A} u_A'] / denom
        d     = Σ_A [d_A·(1−u_A)·∏_{A'≠A} u_A'] / denom
        u     = (N − Σ_A u_A) · ∏_A u_A / denom
        a     = Σ_A [a_A·(1−u_A)] / (N − Σ_A u_A)   (confidence-weighted)

    Verified against Table I of [van der Heijden 2018] (3-source example):
        A1=(0.10,0.30,0.60), A2=(0.40,0.20,0.40), A3=(0.70,0.10,0.20)
        → WBF: b≈0.562, d≈0.146, u≈0.292  ✓
    """
    ops = list(opinions)
    n   = len(ops)
    if n == 0:
        return vacuous()
    if n == 1:
        return ops[0]

    us = [w.u for w in ops]

    # Case 3: all vacuous
    if all(abs(u - 1.0) < EPS for u in us):
        return Opinion(0.0, 0.0, 1.0, sum(w.a for w in ops) / n)

    # Case 2: ≥1 dogmatic — combine dogmatic opinions with equal weights
    dog_ops = [w for w in ops if w.u < EPS]
    if dog_ops:
        nd = len(dog_ops)
        b  = sum(w.b for w in dog_ops) / nd
        d  = sum(w.d for w in dog_ops) / nd
        a  = sum(w.a for w in dog_ops) / nd
        return Opinion(b, d, 0.0, a)

    # Case 1: all non-dogmatic, not all vacuous
    prod_u      = float(np.prod(us))
    prods_excl  = [_prod_except(us, k) for k in range(n)]
    sum_prods   = sum(prods_excl)
    denom       = sum_prods - n * prod_u

    if abs(denom) < EPS:
        # Degenerate: all uncertainties identical → fall back to ABF
        return _fuse_abf_multi(ops)

    bw = [(1.0 - us[k]) * prods_excl[k] for k in range(n)]
    b  = sum(ops[k].b * bw[k] for k in range(n)) / denom
    d  = sum(ops[k].d * bw[k] for k in range(n)) / denom
    u  = (n - sum(us)) * prod_u / denom

    # Confidence-weighted base rate
    conf_sum = n - sum(us)   # = Σ(1 − u_A)
    a = sum(ops[k].a * (1.0 - us[k]) for k in range(n)) / max(EPS, conf_sum)

    return Opinion(b, d, max(0.0, u), a)


# ------------------------------------------------------------------ #
#  Consensus & Compromise Fusion  (CCF) — Definition 5              #
# ------------------------------------------------------------------ #
def consensus_compromise_fusion(opinions: Iterable[Opinion]) -> Opinion:
    """
    Direct N-source consensus & compromise fusion (CCF) for *binomial*
    opinions — Definition 5 of [van der Heijden 2018].

    For binomial opinions with domain X = {x, x̄}, the CC-fusion reduces
    to a three-step procedure operating on (b, d, u) directly.

    Step 1 — Consensus:
        b_cons = min_A b_A            (minimum common positive belief)
        d_cons = min_A d_A            (minimum common negative belief)
        b_res_A = b_A − b_cons        (residual positive belief of A)
        d_res_A = d_A − d_cons        (residual negative belief of A)

    Step 2 — Compromise (derived for binomial X = {x, x̄}):
        b_comp(x)  = Σ_A b_res_A·∏_{A'≠A} u_A' + ∏_A b_res_A
        b_comp(x̄) = Σ_A d_res_A·∏_{A'≠A} u_A' + ∏_A d_res_A
        b_comp(X)  = ∏_A(b_res_A+d_res_A) − ∏_A b_res_A − ∏_A d_res_A
                     (composite belief → transferred to uncertainty)
        u_pre      = ∏_A u_A

    Step 3 — Normalization:
        b_comp_total = b_comp(x) + b_comp(x̄) + b_comp(X)
        η  = (1 − b_cons_total − u_pre) / b_comp_total
        b  = b_cons + η·b_comp(x)
        d  = d_cons + η·b_comp(x̄)
        u  = u_pre  + η·b_comp(X)

    Verified against Table I of [van der Heijden 2018] (3-source example):
        A1=(0.10,0.30,0.60), A2=(0.40,0.20,0.40), A3=(0.70,0.10,0.20)
        → CCF: b≈0.629, d≈0.182, u≈0.189  ✓
    """
    ops = list(opinions)
    n   = len(ops)
    if n == 0:
        return vacuous()
    if n == 1:
        return ops[0]

    us   = [w.u for w in ops]
    bs   = [w.b for w in ops]
    ds   = [w.d for w in ops]
    a    = sum(w.a for w in ops) / n   # simple average of base rates

    # ---- Step 1: Consensus ----
    b_cons = min(bs)
    d_cons = min(ds)
    b_cons_total = b_cons + d_cons

    b_res = [bs[k] - b_cons for k in range(n)]
    d_res = [ds[k] - d_cons for k in range(n)]

    # ---- Step 2: Compromise ----
    prods_excl = [_prod_except(us, k) for k in range(n)]

    # b_comp(x): component 1 (one source residual, others uncertain)
    #          + component 2 (all sources provide residual simultaneously, ∩=x)
    # For binomial: component 2 = ∏_A b_res_A  (since a(x|x)=1)
    b_comp_x   = (sum(b_res[k] * prods_excl[k] for k in range(n))
                  + float(np.prod(b_res)))

    b_comp_xbar = (sum(d_res[k] * prods_excl[k] for k in range(n))
                   + float(np.prod(d_res)))

    # b_comp(X): composite belief (all mixed tuples with ≥1 x and ≥1 x̄)
    # = ∏_A(b_res_A+d_res_A) − ∏_A b_res_A − ∏_A d_res_A
    prod_sum_res = float(np.prod([b_res[k] + d_res[k] for k in range(n)]))
    prod_b_res   = float(np.prod(b_res))
    prod_d_res   = float(np.prod(d_res))
    b_comp_X     = max(0.0, prod_sum_res - prod_b_res - prod_d_res)

    u_pre        = float(np.prod(us))
    b_comp_total = b_comp_x + b_comp_xbar + b_comp_X

    # ---- Step 3: Normalization ----
    if b_comp_total < EPS:
        # No residual compromise — consensus opinion dominates
        b_out = b_cons
        d_out = d_cons
        u_out = max(0.0, 1.0 - b_out - d_out)
        return Opinion(b_out, d_out, u_out, a)

    eta   = (1.0 - b_cons_total - u_pre) / b_comp_total
    b_out = b_cons + eta * b_comp_x
    d_out = d_cons + eta * b_comp_xbar
    u_out = u_pre  + eta * b_comp_X

    return Opinion(b_out, d_out, max(0.0, u_out), a)


# ------------------------------------------------------------------ #
#  Unified multi-source fuse_many                                    #
# ------------------------------------------------------------------ #
def fuse_many(opinions: Iterable[Opinion], how: str = "averaging") -> Opinion:
    """
    Fuse an arbitrary number of opinions using the specified operator.

    Parameters
    ----------
    opinions : iterable of Opinion
    how      : one of
        "averaging"  — N-source averaging belief fusion (ABF).
                       NOTE: ABF is not associative, so the direct N-source
                       formula is used (not sequential binary application).
        "cumulative" — N-source cumulative belief fusion (CBF).
        "weighted"   — N-source weighted belief fusion (WBF).
        "ccf"        — N-source consensus & compromise fusion (CCF).

    Returns
    -------
    Opinion
        The fused opinion. Returns vacuous() for an empty iterable.
    """
    ops = list(opinions)
    if not ops:
        return vacuous()
    if len(ops) == 1:
        return ops[0]

    if how == "cumulative":
        return _fuse_cbf_multi(ops)
    elif how == "weighted":
        return weighted_belief_fusion(ops)
    elif how == "ccf":
        return consensus_compromise_fusion(ops)
    else:   # "averaging" (default)
        return _fuse_abf_multi(ops)


# ------------------------------------------------------------------ #
#  Trust revision (⊖)                                                #
#  Used to refine parameter trust: ω' = ω ⊖ evidence                 #
#  Implemented as conservative averaging fusion                      #
# ------------------------------------------------------------------ #
def revise(omega: Opinion, evidence: Opinion) -> Opinion:
    return averaging_fusion(omega, evidence)


# ------------------------------------------------------------------ #
#  Binomial multiplication (⊙) and conservative division (⊘)         #
# ------------------------------------------------------------------ #
def multiply(w1: Opinion, w2: Opinion) -> Opinion:
    """Binomial multiplication (AND) - Jøsang."""
    a1, a2 = w1.a, w2.a
    a = a1 * a2
    b = w1.b * w2.b + (
        ((1.0 - a1) * a2 * w1.b * w2.u + (1.0 - a2) * a1 * w1.u * w2.b)
        / max(EPS, 1.0 - a)
    )
    d = w1.d + w2.d - w1.d * w2.d
    u = max(0.0, 1.0 - b - d)
    return Opinion(b, d, u, a)


def conservative_div(w1: Opinion, w2: Opinion) -> Opinion:
    """Eq. (7.7) - conservative trust 'division' for parameter updates."""
    b = min(w1.b, w2.b)
    d = max(w1.d, w2.d)
    u = max(0.0, 1.0 - (b + d))
    return Opinion(b, d, u, 0.5 * (w1.a + w2.a))


# ------------------------------------------------------------------ #
#  Inferential deduction (⊚)                                         #
#  ω_{n||y} = ω_y ⊚ (ω_{n|y}, ω_{n|¬y})                              #
# ------------------------------------------------------------------ #
def deduce(omega_y: Opinion,
           omega_n_given_y: Opinion,
           omega_n_given_not_y: Opinion) -> Opinion:
    """
    Simplified binomial inferential deduction.
    Returns the trust in n marginalised over y / ¬y.
    """
    Py     = omega_y.P
    notPy  = 1.0 - Py
    b = Py * omega_n_given_y.b + notPy * omega_n_given_not_y.b
    d = Py * omega_n_given_y.d + notPy * omega_n_given_not_y.d
    u = Py * omega_n_given_y.u + notPy * omega_n_given_not_y.u
    s = b + d + u
    if s > 0:
        b, d, u = b / s, d / s, u / s
    return Opinion(b, d, u, omega_y.a)


# ------------------------------------------------------------------ #
#  Convenience: NumPy <-> Opinion grid helpers                       #
# ------------------------------------------------------------------ #
def opinions_to_array(grid) -> np.ndarray:
    """Convert a nested list / array of Opinions to a stacked ndarray
    of shape (..., 4) holding (b, d, u, a)."""
    if isinstance(grid, Opinion):
        return np.array(grid.as_tuple())
    arr = np.array([opinions_to_array(g) for g in grid])
    return arr
