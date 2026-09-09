"""attribution.py — attribution-weighted parameter trust for PaTAS.

The Parameter-Trust Update of the framework conditions every weight in a
layer on the same batch-aggregated opinion: the per-sample, per-feature
input opinions are pooled into one scalar and fused across the batch before
they reach a parameter.  The trust state therefore carries no record of
*which* data shaped *which* weight, and a corruption that the optimizer
fits smoothly (a trigger-patch backdoor) leaves no trace in it.

This module restores that record.  Each parameter's opinion is built from
the opinions of the samples that actually formed it, weighted by their share
of the gradient that formed it:

    for weight theta_ij of layer l, with input activation a_j and error d_i,
        mass_ij = sum_n |a_j^n| |d_i^n|                       (influence)
        r_ij    = N * sum_n |a_j^n| |d_i^n| b_j^n  / mass_ij  (positive evidence)
        s_ij    = N * sum_n |a_j^n| |d_i^n| d_j^n  / mass_ij  (negative evidence)
        omega_ij = BPQ(r_ij, s_ij, W)

with (b_j^n, d_j^n) the opinion held about feature j of sample n.  For the
input layer these are the per-feature opinions (no pooling); for hidden
layers and the bias row they are the sample's own opinion, since the
provenance of an activation is the provenance of the sample that produced
it.  A weight formed entirely by untrusted data inherits that distrust; a
weight formed by trusted data does not, whatever the gradient magnitudes.

The computation is an offline replay of the converged model over the
training set, the same mode in which every cached PaTAS experiment is
produced, so it needs no change to the NN/PTAS protocol.  Results are
written as a standard PTAS cache directory (omega_arrays.pkl + at.pkl) with
the threshold slot naming the trust source, so every downstream tool reads
them unchanged.

Usage
-----
    python attribution.py --dataset mnist --arch 128 --poisoned-patch 4 \
        --trust-source provenance --untrusted-tail 0.3333
    python attribution.py --dataset mnist --arch 512 --trust-source conformity
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module"), os.path.join(_here, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# ---------------------------------------------------------------------------
# Trust sources: what opinion is held about feature j of sample n
# ---------------------------------------------------------------------------

class TrustSource:
    """Returns per-batch (belief, disbelief) for input features and samples."""
    name = "base"

    def features(self, idx, xb):
        """(belief, disbelief) per input feature, each (batch, d)."""
        raise NotImplementedError

    def samples(self, idx, xb):
        """(belief, disbelief) per sample, each (batch,)."""
        raise NotImplementedError


class TrustedSource(TrustSource):
    """Everything fully trusted: the control that must produce no signal."""
    name = "trusted"

    def features(self, idx, xb):
        return np.ones_like(xb), np.zeros_like(xb)

    def samples(self, idx, xb):
        return np.ones(len(xb), np.float32), np.zeros(len(xb), np.float32)


class ProvenanceSource(TrustSource):
    """Source-level knowledge only: a contiguous tail of the training set came
    from an untrusted source.  Nothing is known about which samples in it are
    corrupted, about any trigger, or about the affected classes."""
    name = "provenance"

    def __init__(self, n_total: int, tail_frac: float, opinion=(0.0, 0.0, 1.0)):
        self.untrusted = np.zeros(n_total, bool)
        self.untrusted[int(round((1.0 - tail_frac) * n_total)):] = True
        self.b_u, self.d_u = float(opinion[0]), float(opinion[1])

    def samples(self, idx, xb):
        u = self.untrusted[idx]
        return (np.where(u, self.b_u, 1.0).astype(np.float32),
                np.where(u, self.d_u, 0.0).astype(np.float32))

    def features(self, idx, xb):
        b, d = self.samples(idx, xb)
        return (np.repeat(b[:, None], xb.shape[1], 1),
                np.repeat(d[:, None], xb.shape[1], 1))


class MultiSourceProvenance(TrustSource):
    """Several data sources carrying DIFFERENT AMOUNTS of evidence.

    The single-source case cannot separate the opinion algebra from a plain
    influence ratio: with one untrusted source, belief and disbelief are both
    functions of the untrusted share of a parameter's influence, so the mapped
    opinion is a monotone function of that share and the two rankings must
    coincide.  They can only diverge when sources differ in how much evidence
    they carry, e.g. a verified source (1,0,0), a source of unknown provenance
    (0,0,1) and a source known to be compromised (0,1,0).  A scalar ratio sees
    "not verified" and cannot tell the last two apart; an opinion keeps
    vacuity and disbelief on separate axes.
    """
    name = "multisource"

    def __init__(self, assignment, opinions):
        self.assign = np.asarray(assignment, dtype=int)
        self.ops = np.asarray(opinions, dtype=np.float32)   # (n_sources, 3)

    def samples(self, idx, xb):
        o = self.ops[self.assign[idx]]
        return o[:, 0].astype(np.float32), o[:, 1].astype(np.float32)

    def features(self, idx, xb):
        b, d = self.samples(idx, xb)
        return (np.repeat(b[:, None], xb.shape[1], 1),
                np.repeat(d[:, None], xb.shape[1], 1))


class ConformitySource(TrustSource):
    """Estimation from data: per-feature conformity opinions from the
    framework's own InputTrustModel, un-pooled."""
    name = "conformity"

    def __init__(self, X_train):
        from input_trust import InputTrustModel
        self.itm = InputTrustModel().fit(X_train)

    def features(self, idx, xb):
        ops = self.itm.opinions(xb)                    # (batch, d, 3)
        return ops[..., 0], ops[..., 1]

    def samples(self, idx, xb):
        from subjective_logic import bpq_vec
        g = self.itm.conformity(xb)
        n_j = self.itm.evidence * self.itm.weights
        pooled = bpq_vec((n_j * g).sum(1), (n_j * (1.0 - g)).sum(1), W=self.itm.W)
        return pooled[..., 0].astype(np.float32), pooled[..., 1].astype(np.float32)


class OracleSource(TrustSource):
    """The assumption the published experiment made: the analyst knows the
    trigger location and which samples carry it.  Kept for comparison only."""
    name = "oracle"

    def __init__(self, patch_idx, patched_mask):
        self.patch_idx, self.patched = np.asarray(patch_idx), patched_mask

    def features(self, idx, xb):
        b = np.ones_like(xb); d = np.zeros_like(xb)
        p = self.patched[idx]
        b[np.ix_(p, self.patch_idx)] = 0.0
        d[np.ix_(p, self.patch_idx)] = 1.0
        return b, d

    def samples(self, idx, xb):
        p = self.patched[idx]
        return (np.where(p, 0.0, 1.0).astype(np.float32),
                np.where(p, 1.0, 0.0).astype(np.float32))


# ---------------------------------------------------------------------------
# The attribution replay
# ---------------------------------------------------------------------------

def accumulate(Ws, bs, X, Y, source: TrustSource, batch: int = 512,
               verbose: bool = True):
    """Replay the training set and accumulate, per parameter, the influence
    mass and the belief/disbelief-weighted mass of the samples that produced
    it.  Separated from ``attribute`` so that baselines which do not use the
    opinion algebra (a plain influence ratio) can share the same replay."""
    L = len(Ws)
    mass = [np.zeros((W.shape[0] + 1, W.shape[1]), np.float64) for W in Ws]
    R = [np.zeros_like(m) for m in mass]
    S = [np.zeros_like(m) for m in mass]
    n = len(X)
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(s0 + batch, n))
        xb, yb = X[idx], Y[idx]
        acts, zs = [xb], []
        a = xb
        for l in range(L):
            z = a @ Ws[l] + bs[l]
            zs.append(z)
            if l == L - 1:
                e = np.exp(z - z.max(1, keepdims=True)); a = e / e.sum(1, keepdims=True)
            else:
                a = np.maximum(z, 0.0)
            acts.append(a)
        dz = (acts[-1] - yb) / len(idx)
        b_s, d_s = source.samples(idx, xb)
        for l in range(L - 1, -1, -1):
            a_prev = np.abs(acts[l]); adz = np.abs(dz)
            if l == 0:
                b_f, d_f = source.features(idx, xb)
            else:
                b_f = np.repeat(b_s[:, None], a_prev.shape[1], 1)
                d_f = np.repeat(d_s[:, None], a_prev.shape[1], 1)
            mass[l][:-1] += a_prev.T @ adz
            R[l][:-1] += (a_prev * b_f).T @ adz
            S[l][:-1] += (a_prev * d_f).T @ adz
            mass[l][-1] += adz.sum(0)
            R[l][-1] += (b_s[:, None] * adz).sum(0)
            S[l][-1] += (d_s[:, None] * adz).sum(0)
            if l > 0:
                dz = (dz @ Ws[l].T) * (zs[l - 1] > 0)
        if verbose and (s0 // batch) % 20 == 0:
            print(f"    replay {min(s0 + batch, n):>6}/{n}", flush=True)
    return mass, R, S


def attribute(Ws, bs, X, Y, source: TrustSource, evidence: float = 50.0,
              W_prior: float = 2.0, batch: int = 512, verbose: bool = True):
    """Replay the training set through the converged model and build one
    opinion per parameter from the samples that formed it.

    Returns (omegas, mass) with omegas a list of (in+1, out, 3) arrays in the
    same layout as the PTAS omega_arrays.pkl, and mass the per-layer
    influence matrices (used for reporting which parameters are live)."""
    from subjective_logic import bpq_vec
    L = len(Ws)
    mass, R, S = accumulate(Ws, bs, X, Y, source, batch=batch, verbose=verbose)
    omegas = []
    for l in range(L):
        m = mass[l]
        r = np.divide(evidence * R[l], m, out=np.zeros_like(m), where=m > 1e-12)
        s = np.divide(evidence * S[l], m, out=np.zeros_like(m), where=m > 1e-12)
        omegas.append(bpq_vec(r, s, W=W_prior).astype(np.float32))
    return omegas, mass


def flag_features(score, live, k: float = 8.0):
    """Distribution-relative flagging of suspect parameters.

    The absolute level of attributed trust shifts from one training run to
    the next (medians of 0.46 to 0.61 across seeds of the same setup), so a
    fixed threshold flags everything on one model and nothing on another.
    The separation does not shift: corrupted features sit many robust
    deviations below the body of the distribution.  Flag a live feature when
    it lies more than ``k`` scaled MADs below the median, which needs no
    per-dataset calibration and stays silent when there is no tail.

    Returns (indices, z) with z the robust deviation of every feature.
    """
    lv = np.asarray(score)[live]
    med = float(np.median(lv))
    mad = float(np.median(np.abs(lv - med)))
    scale = max(mad * 1.4826, 1e-9)
    z = (med - np.asarray(score)) / scale
    return np.where((z > k) & live)[0], z


def feedforward_trusted(omegas, input_dim):
    """Output-class opinions under a fully trusted input (the at.pkl object)."""
    from concrete.TensorTO import TensorArrayTO, fill as tfill
    cur = TensorArrayTO(tfill((1, input_dim), method="trust"))
    one = tfill((1, 1), method="one")           # vacuous bias column, as in PTAS
    for om in omegas:
        with_bias = TensorArrayTO(np.concatenate([np.asarray(cur.value), one], axis=1))
        cur = TensorArrayTO.dot(with_bias, TensorArrayTO(om))
    return cur


def projected(omega):
    """Projected probability b + 0.5 u of every parameter opinion."""
    return omega[..., 0] + 0.5 * omega[..., 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "gtsrb"])
    p.add_argument("--arch", type=int, nargs="+", default=[128])
    p.add_argument("--poisoned-patch", type=int, default=None,
                   help="Patch size of the cached poisoned scenario (omit for a clean model)")
    p.add_argument("--poison-mode", choices=["both", "flip", "patch"], default="both")
    p.add_argument("--trust-source", default="provenance",
                   choices=["provenance", "conformity", "oracle", "trusted"])
    p.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0,
                   help="Fraction of the training set (from the end) held to come "
                        "from an untrusted source (provenance source only)")
    p.add_argument("--untrusted-opinion", default="0,0,1",
                   help="Opinion b,d,u held about the untrusted source "
                        "(default 0,0,1 = no knowledge; 0,1,0 = known compromised)")
    p.add_argument("--evidence", type=float, default=50.0)
    p.add_argument("--x-trust", default="trust")
    p.add_argument("--y-trust", default="trust")
    p.add_argument("--noise-level", type=float, default=None)
    p.add_argument("--tag", default=None, help="Override the cache-directory tag")
    return p.parse_args()


def main():
    args = parse_args()
    from NN.datasets import load_data
    from main import nn_cache_dir, ptas_cache_dir
    from concrete.TensorTO import TensorArrayTO

    arch = tuple(args.arch)
    arch_str = "_".join(str(h) for h in arch)
    patch_tag = None
    if args.poisoned_patch:
        patch_tag = str(args.poisoned_patch)
        if args.poison_mode != "both":
            patch_tag += f"pm{args.poison_mode}"

    nn_dir = nn_cache_dir(args.dataset, arch_str, args.x_trust, args.y_trust,
                          patch=patch_tag, noise_level=args.noise_level)
    nn_path = os.path.join(nn_dir, "nn_model.pkl")
    if not os.path.exists(nn_path):
        raise SystemExit(f"trained model not found: {nn_path}\n"
                         f"Run the corresponding scenario first.")
    with open(nn_path, "rb") as fh:
        wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1))
        i += 1
    print(f"[attr] loaded {nn_path}  layers {[w.shape for w in Ws]}")

    y_how = "clean" if args.noise_level is None else "noise"
    kw = {} if args.noise_level is None else {"noise_level": args.noise_level}
    X, X_test, Y, y_test, _ = load_data(
        args.dataset, "clean", y_how,
        poisoned_patch=args.poisoned_patch, poison_mode=args.poison_mode, **kw)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    print(f"[attr] training data {X.shape}  poisoned_patch={args.poisoned_patch}")

    # --- trust source -------------------------------------------------------
    patch_idx = patched = None
    if args.poisoned_patch:
        img = int(round(X.shape[1] ** 0.5))
        patch_idx = np.array([img * r + c for r in range(args.poisoned_patch)
                              for c in range(args.poisoned_patch)])
        lit = X[:, patch_idx].min(1) > 0.5      # samples actually carrying the trigger
        patched = lit
        print(f"[attr] trigger present in {patched.sum()} / {len(X)} samples "
              f"({patched.mean()*100:.1f}%)")
    if args.trust_source == "provenance":
        op = tuple(float(v) for v in args.untrusted_opinion.split(","))
        src = ProvenanceSource(len(X), args.untrusted_tail, op)
        print(f"[attr] provenance: last {args.untrusted_tail*100:.1f}% of the training "
              f"set held untrusted, opinion {op}; no per-sample or per-pixel knowledge")
    elif args.trust_source == "conformity":
        src = ConformitySource(X)
    elif args.trust_source == "oracle":
        if patched is None:
            raise SystemExit("--trust-source oracle requires --poisoned-patch")
        src = OracleSource(patch_idx, patched)
    else:
        src = TrustedSource()

    omegas, mass = attribute(Ws, bs, X, Y, src, evidence=args.evidence)

    # --- write a standard PTAS cache directory ------------------------------
    tag = args.tag or f"attr-{src.name}"
    out_dir = ptas_cache_dir(args.dataset, arch_str, args.x_trust, args.y_trust,
                             tag, patch=patch_tag, noise_level=args.noise_level)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "omega_arrays.pkl"), "wb") as fh:
        pickle.dump([np.asarray(o) for o in omegas], fh)
    at = feedforward_trusted(omegas, Ws[0].shape[0])
    with open(os.path.join(out_dir, "at.pkl"), "wb") as fh:
        pickle.dump(TensorArrayTO(np.asarray(at.to_numpy())), fh)
    print(f"[attr] saved {out_dir}/omega_arrays.pkl and at.pkl")

    # --- diagnostics --------------------------------------------------------
    pp0 = projected(omegas[0][:-1])                    # (in, hidden) input-layer weights
    live = mass[0][:-1].sum(1) > np.percentile(mass[0][:-1].sum(1), 20)
    feat_min = pp0.min(1)
    suspect = np.where((feat_min < 0.25) & live)[0]
    at_np = np.asarray(at.to_numpy()).reshape(-1, 3)
    summary = {
        "dataset": args.dataset, "arch": arch_str, "trust_source": src.name,
        "poisoned_patch": args.poisoned_patch, "poison_mode": args.poison_mode,
        "untrusted_tail": args.untrusted_tail if src.name == "provenance" else None,
        "evidence": args.evidence,
        "n_live_features": int(live.sum()),
        "n_suspect_features": int(len(suspect)),
        "min_feature_trust": float(feat_min[live].min()),
        "mean_feature_trust": float(feat_min[live].mean()),
        "class_trust": [float(v) for v in (at_np[:, 0] + 0.5 * at_np[:, 2])],
    }
    if patch_idx is not None:
        from sklearn.metrics import roc_auc_score
        pm = np.zeros(len(feat_min), bool); pm[patch_idx] = True
        found = int(np.isin(patch_idx, suspect).sum())
        auroc = float(roc_auc_score(pm[live], -feat_min[live])) if pm[live].any() else float("nan")
        summary.update(trigger_pixels=int(pm.sum()), trigger_pixels_flagged=found,
                       trigger_identification_auroc=auroc,
                       trigger_mean_trust=float(feat_min[patch_idx].mean()),
                       other_mean_trust=float(feat_min[live & ~pm].mean()))
        print(f"[attr] trigger pixels flagged {found}/{int(pm.sum())}   "
              f"identification AUROC {auroc:.4f}   "
              f"trigger trust {summary['trigger_mean_trust']:.4f} vs "
              f"other {summary['other_mean_trust']:.4f}")
    print(f"[attr] suspect input features: {len(suspect)} of {int(live.sum())} live")
    print(f"[attr] per-class output trust under a trusted input: "
          f"{[round(v,3) for v in summary['class_trust']]}")
    with open(os.path.join(out_dir, "attribution_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[attr] saved {out_dir}/attribution_summary.json")


if __name__ == "__main__":
    main()
