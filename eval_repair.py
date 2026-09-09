"""eval_repair.py — provenance-guided removal of an implanted backdoor.

Localising the parameters that untrusted data shaped is a diagnosis; this
script asks whether it supports a cure.  The defender is told only that one
data source is untrusted.  Nothing is known about the trigger, which samples
carry it, or which classes it targets.  Parameters whose attributed trust
falls below a threshold are removed, and we measure what happens to the
implanted behaviour and to clean accuracy.

Four criteria are compared under the same knowledge, so the contribution of
each ingredient is separable:

  attribution   the subjective-logic opinion per parameter (this work)
  influence     the raw untrusted share of the influence mass, no opinion
                algebra -- the ablation that asks whether the calculus earns
                its place over a plain influence ratio
  finepruning   hidden units with the lowest mean activation on the TRUSTED
                subset, the standard activation-based defence, given the same
                provenance split as its clean reference set
  random        features removed at random, the control that separates a real
                localisation from generic robustness

Usage
-----
    python eval_repair.py --dataset mnist --arch 128 --poisoned-patch 4
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


def forward(Ws, bs, X):
    a = X
    for l in range(len(Ws)):
        z = a @ Ws[l] + bs[l]
        a = np.maximum(z, 0.0) if l < len(Ws) - 1 else z
    return a


def evaluate(Ws, bs, X_test, y_test, patch_idx, pois_pair=(6, 9)):
    """Clean accuracy, accuracy on the targeted classes, and attack success."""
    pred = forward(Ws, bs, X_test).argmax(1)
    acc = float(np.mean(pred == y_test))
    a, b = pois_pair
    asr, pair_acc = [], []
    for src, dst in ((a, b), (b, a)):
        Xs = X_test[y_test == src]
        if not len(Xs):
            continue
        pair_acc.append(float(np.mean(forward(Ws, bs, Xs).argmax(1) == src)))
        Xq = Xs.copy(); Xq[:, patch_idx] = PATCH_VALUE
        asr.append(float(np.mean(forward(Ws, bs, Xq).argmax(1) == dst)))
    return acc, float(np.mean(pair_acc)), float(np.mean(asr))


def prune_features(Ws, bs, feats):
    """Remove input features: zero their outgoing first-layer weights."""
    W = [w.copy() for w in Ws]
    W[0][np.asarray(feats, dtype=int), :] = 0.0
    return W, bs


def prune_units(Ws, bs, units):
    """Remove hidden units of the first hidden layer (fine-pruning)."""
    W = [w.copy() for w in Ws]; B = [b.copy() for b in bs]
    u = np.asarray(units, dtype=int)
    W[0][:, u] = 0.0; B[0][u] = 0.0
    return W, B


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "gtsrb"])
    ap.add_argument("--arch", type=int, nargs="+", default=[128])
    ap.add_argument("--poisoned-patch", type=int, default=4)
    ap.add_argument("--poison-mode", choices=["both", "flip", "patch"], default="both")
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--untrusted-opinion", default="0,1,0")
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.35],
                    help="Fraction of the layer removed, swept for every criterion")
    ap.add_argument("--seeds", type=int, default=20, help="Random-control repetitions")
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import nn_cache_dir, DATASET_META
    from attribution import accumulate, ProvenanceSource
    from subjective_logic import bpq_vec

    global PATCH_VALUE
    arch_str = "_".join(str(h) for h in args.arch)
    patch_tag = str(args.poisoned_patch)
    if args.poison_mode != "both":
        patch_tag += f"pm{args.poison_mode}"
    nn_path = os.path.join(nn_cache_dir(args.dataset, arch_str, "trust", "trust",
                                        patch=patch_tag), "nn_model.pkl")
    with open(nn_path, "rb") as fh:
        wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1)); i += 1
    print(f"[repair] model {nn_path}  {[w.shape for w in Ws]}")

    X, X_test, Y, y_test_oh, _ = load_data(args.dataset, "clean", "clean",
                                           poisoned_patch=args.poisoned_patch,
                                           poison_mode=args.poison_mode)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    X_test = np.asarray(X_test, np.float32); y_test = np.asarray(y_test_oh).argmax(1)
    meta = DATASET_META[args.dataset]
    img = meta["img_size"]
    patch_idx = np.array([img * r + c for r in range(args.poisoned_patch)
                          for c in range(args.poisoned_patch)])
    PATCH_VALUE = meta["scale_patch"](1.0)
    args.pois_pair = list(meta["pois_pair"])
    print(f"[repair] trigger {args.poisoned_patch}x{args.poisoned_patch} at value "
          f"{PATCH_VALUE:.4f} (dataset scale)")

    op = tuple(float(v) for v in args.untrusted_opinion.split(","))
    src = ProvenanceSource(len(X), args.untrusted_tail, op)
    print(f"[repair] defender knows only: the last {args.untrusted_tail*100:.0f}% of the "
          f"training data comes from an untrusted source, opinion {op}")
    mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=True)

    m0 = mass[0][:-1]                       # (in, hidden) influence per weight
    live = m0.sum(1) > np.percentile(m0.sum(1), 20)
    r = np.divide(args.evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    s = np.divide(args.evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    om = bpq_vec(r, s, W=2.0)
    attribution_score = (om[..., 0] + 0.5 * om[..., 2]).min(1)      # low = suspicious
    untrusted_share = np.divide(S[0][:-1], R[0][:-1] + S[0][:-1],
                                out=np.zeros_like(m0), where=(R[0][:-1] + S[0][:-1]) > 1e-12)
    influence_score = -untrusted_share.max(1)                        # low = suspicious
    trusted = ~src.untrusted
    act_trusted = np.maximum(X[trusted] @ Ws[0] + bs[0], 0.0).mean(0)  # per hidden unit

    acc0, pair0, asr0 = evaluate(Ws, bs, X_test, y_test, patch_idx, tuple(args.pois_pair))
    print(f"\n[repair] undefended model: clean acc {acc0*100:.2f}%  "
          f"target-class acc {pair0*100:.2f}%  attack success {asr0*100:.2f}%")

    rng = np.random.default_rng(0)
    n_feat, n_unit = Ws[0].shape
    rows = []
    for frac in args.budgets:
        k_f, k_u = max(1, int(round(frac * n_feat))), max(1, int(round(frac * n_unit)))
        cand = {}
        order = np.argsort(np.where(live, attribution_score, np.inf))
        cand["attribution"] = ("feat", order[:k_f])
        order = np.argsort(np.where(live, influence_score, np.inf))
        cand["influence"] = ("feat", order[:k_f])
        cand["finepruning"] = ("unit", np.argsort(act_trusted)[:k_u])
        for name, (kind, sel) in cand.items():
            W2_, b2_ = (prune_features(Ws, bs, sel) if kind == "feat"
                        else prune_units(Ws, bs, sel))
            acc, pair, asr = evaluate(W2_, b2_, X_test, y_test, patch_idx, tuple(args.pois_pair))
            rec = (float(np.isin(patch_idx, sel).mean()) if kind == "feat" else float("nan"))
            rows.append(dict(method=name, budget=frac, n_removed=len(sel),
                             clean_acc=acc, target_acc=pair, asr=asr, trigger_recall=rec))
        accs, asrs = [], []
        for t in range(args.seeds):
            sel = rng.choice(np.where(live)[0], k_f, replace=False)
            W2_, b2_ = prune_features(Ws, bs, sel)
            a, _, z = evaluate(W2_, b2_, X_test, y_test, patch_idx, tuple(args.pois_pair)); accs.append(a); asrs.append(z)
        rows.append(dict(method="random", budget=frac, n_removed=k_f,
                         clean_acc=float(np.mean(accs)), target_acc=float("nan"),
                         asr=float(np.mean(asrs)), trigger_recall=float(k_f) / n_feat))

    print(f"\n{'method':<13} {'budget':>7} {'removed':>8} {'clean acc':>10} "
          f"{'target acc':>11} {'attack succ':>12} {'trigger found':>14}")
    for d in rows:
        print(f"{d['method']:<13} {d['budget']*100:>6.0f}% {d['n_removed']:>8} "
              f"{d['clean_acc']*100:>9.2f}% {d['target_acc']*100:>10.2f}% "
              f"{d['asr']*100:>11.2f}% {d['trigger_recall']*100:>13.1f}%")

    out_dir = f"results/Repair_{args.dataset}_{arch_str}_patch{patch_tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=args.dataset, arch=arch_str, patch=args.poisoned_patch,
                       poison_mode=args.poison_mode, untrusted_tail=args.untrusted_tail,
                       untrusted_opinion=op, undefended=dict(clean_acc=acc0,
                       target_acc=pair0, asr=asr0), rows=rows), fh, indent=2)
    print(f"\n[repair] saved {out_dir}/summary.json")


if __name__ == "__main__":
    main()
