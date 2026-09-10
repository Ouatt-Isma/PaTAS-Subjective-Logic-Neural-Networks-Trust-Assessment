"""Figure 1 of the paper: (a) the trust distribution the decision rule acts on,
(b) attack success under the adaptive adversary for every criterion.

Both panels are produced from the same pipeline and the same provenance
setting as the tables, so the numbers annotated on the figure are the numbers
in the tables.

Usage
-----
    python make_fig_trust.py --out ../PaTAS-SaTML-v2/fig_trust
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module"), os.path.join(_here, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def scores_for(model_path, poisoned, opinion, evidence=50.0, tail=1.0 / 3.0):
    """Per-feature attributed trust of a cached MNIST model, exactly as
    eval_localisation.py computes it."""
    import pickle
    from NN.datasets import load_data
    from attribution import ProvenanceSource, accumulate, flag_features
    from subjective_logic import bpq_vec
    if poisoned:
        X, _, Y, _, _ = load_data("mnist", "clean", "clean", poisoned_patch=4)
    else:
        X, _, Y, _, _ = load_data("mnist", "clean", "clean")
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    with open(model_path, "rb") as fh:
        wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1)); i += 1
    src = ProvenanceSource(len(X), tail, opinion)
    mu = X[~src.untrusted].mean(0)
    infl, mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=False, center=mu)
    m0 = mass[0][:-1]
    live = infl[0][:-1].sum(1) > np.percentile(infl[0][:-1].sum(1), 20)
    r = np.divide(evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    s = np.divide(evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    om = bpq_vec(r, s, W=2.0)
    score = (om[..., 0] + 0.5 * om[..., 2]).min(1)
    sel, _ = flag_features(score, live, 8.0, rule="otsu")
    return score, live, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_trust")
    ap.add_argument("--models", default="results/Localisation_mnist_otsu/models")
    ap.add_argument("--multisource", default="results/MultiSource_mnist_128")
    ap.add_argument("--opinion", default="0,0,1")
    ap.add_argument("--show", type=int, default=140)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    op = tuple(float(v) for v in args.opinion.split(","))
    pidx = np.array([28 * r + c for r in range(4) for c in range(4)])

    # ---- panel (a): sorted attributed trust, backdoored vs clean ------------
    sb, lb, selb = scores_for(os.path.join(args.models, "backdoored_128_seed0.pkl"), True, op)
    sc, lc, selc = scores_for(os.path.join(args.models, "clean128_128_seed0.pkl"), False, op)
    ob = np.argsort(np.where(lb, sb, np.inf))[:int(lb.sum())]
    oc = np.argsort(np.where(lc, sc, np.inf))[:int(lc.sum())]
    n = args.show
    is_trig = np.isin(ob[:n], pidx)
    # the rule's cut: between the highest flagged and the lowest unflagged score
    if len(selb):
        hi, lo = sb[selb].max(), sb[[j for j in ob if j not in set(selb)]].min()
        cut = 0.5 * (hi + lo)
    else:
        cut = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    xs = np.arange(n)
    ax1.plot(xs, sc[oc[:n]], color="0.55", lw=1.4, label=f"clean model ({len(selc)} flagged)")
    ax1.plot(xs, sb[ob[:n]], color="#1f4e79", lw=1.4, label=f"backdoored model ({len(selb)} flagged)")
    ax1.scatter(xs[is_trig], sb[ob[:n]][is_trig], s=22, color="#c0392b", zorder=3,
                label=f"trigger features ({int(is_trig.sum())} of 16 shown)")
    if cut is not None:
        ax1.axhline(cut, ls=":", color="k", lw=1)
        ax1.text(n * 0.98, cut, "rule cuts here", ha="right", va="bottom", fontsize=8)
    ax1.set_xlabel(f"live input features, sorted by attributed trust (lowest {n} of {int(lb.sum())})")
    ax1.set_ylabel("attributed trust  $b + u/2$")
    ax1.set_title("(a) what the rule acts on", fontsize=10, loc="left")
    ax1.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax1.grid(alpha=0.25)

    # ---- panel (b): attack success under leakage, all criteria ------------
    leaks, series = [], {"opinion@thr": [], "scalar@thr": [], "scalar-proj@thr": []}
    base = os.path.join(args.multisource, "summary_uc4-7_cf0.25_p4")
    files = [(0.0, base + ".json")] + sorted(
        (float(f.split("_leak")[1][:-5]), f) for f in glob.glob(base + "_leak*.json"))
    for lk, f in files:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        if not any(r["criterion"] == "opinion@thr" for r in d["rows"]):
            continue                       # a stale single-seed sweep file
        leaks.append(lk * 100)
        for c in series:
            v = [r["asr"] * 100 for r in d["rows"] if r["criterion"] == c]
            series[c].append((np.mean(v), np.std(v)) if v else (np.nan, np.nan))
    styles = {"opinion@thr": ("#1f4e79", "o", "opinion (three states, ratio of evidence)"),
              "scalar@thr": ("#c0392b", "s", "verified share (two states)"),
              "scalar-proj@thr": ("#2e8b57", "^", "mean projected probability (three states)")}
    for c, vals in series.items():
        m = np.array([v[0] for v in vals]); sd = np.array([v[1] for v in vals])
        col, mk, lbl = styles[c]
        ax2.errorbar(leaks, m, yerr=sd, color=col, marker=mk, ms=4.5, lw=1.4,
                     capsize=2.5, label=lbl, alpha=0.95 if c != "scalar-proj@thr" else 0.8,
                     ls="-" if c != "scalar-proj@thr" else "--")
    ax2.set_xlabel("poison hidden in the audited source (%)")
    ax2.set_ylabel("attack success after removal (%)")
    ax2.set_title("(b) the adaptive adversary", fontsize=10, loc="left")
    ax2.set_ylim(-4, 104); ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(args.out + ".pdf"); fig.savefig(args.out + ".png", dpi=200)
    print(f"[fig] panel (a): backdoored flagged {len(selb)} ({int(np.isin(pidx, selb).sum())}/16 trigger), "
          f"clean flagged {len(selc)}, live {int(lb.sum())}; cut at {cut}")
    print(f"[fig] panel (b): leaks {leaks}")
    for c, vals in series.items():
        print(f"      {c:<16}" + "  ".join(f"{m:6.2f}±{s:5.2f}" for m, s in vals))
    print(f"[fig] wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
