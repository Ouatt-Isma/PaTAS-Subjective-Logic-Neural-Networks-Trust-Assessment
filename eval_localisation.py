"""Localisation: what the defence flags, across conditions.

The point of this table is negative control.  A defence that flags the trigger
on a backdoored model is only interesting if it flags nothing on a model with
nothing to find, and if the signal comes from the defender's provenance rather
than from the attribution machinery producing suspicious-looking numbers on
its own.

Every condition here trains its own network and runs the same decision rule as
the removal experiment, so the two tables are produced by one pipeline.  The
earlier version of this table used a fixed absolute cutoff on projected trust,
which is a different rule from the one the rest of the paper uses.

Usage
-----
    python eval_localisation.py --seeds 0 1 2
"""
from __future__ import annotations

import os
import sys
import json
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module"), os.path.join(_here, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# Each condition names the dataset it builds and what the defence should do.
#   patch       : trigger size, or None for no trigger
#   poison_mode : "both" (trigger + label swap), "patch" (trigger, correct
#                 labels), "flip" (label swap, no trigger)
#   arch        : hidden layer sizes
#   provenance  : False runs the no-knowledge control, where every sample is
#                 given the same opinion so there is nothing to attribute
CONDITIONS = [
    dict(key="backdoored",   label="Backdoored",
         patch=4, poison_mode="both", arch=[128], provenance=True),
    dict(key="clean128",     label="Clean, 784-128-10",
         patch=None, poison_mode="both", arch=[128], provenance=True),
    dict(key="clean512",     label="Clean, 784-512-10",
         patch=None, poison_mode="both", arch=[512], provenance=True),
    dict(key="fliponly",     label="Label flip only, no trigger",
         patch=4, poison_mode="flip", arch=[128], provenance=True),
    dict(key="patchonly",    label="Trigger only, labels correct",
         patch=4, poison_mode="patch", arch=[128], provenance=True),
    dict(key="labelnoise",   label="Label-noise trained",
         patch=None, poison_mode="both", arch=[128], provenance=True,
         noise_level=0.3),
    dict(key="noprovenance", label="No provenance used (control)",
         patch=4, poison_mode="both", arch=[128], provenance=False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--rule", choices=["otsu", "mad"], default="otsu",
                    help="Decision rule; must match the removal experiment.")
    ap.add_argument("--untrusted-opinion", default="0,1,0",
                    help="Opinion held about the untrusted source. The removal "
                         "experiment states the source is believed "
                         "compromised, (0,1,0), and this table has to match it "
                         "or the two are not comparable.")
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import DATASET_META
    from attribution import (ProvenanceSource, accumulate, flag_features)
    from subjective_logic import bpq_vec
    from sklearn.metrics import roc_auc_score
    import influence_baseline as IB
    import eval_repair as ER

    meta = DATASET_META[args.dataset]
    img = meta["img_size"]
    ER.PATCH_VALUE = meta["scale_patch"](1.0)
    out = f"results/Localisation_{args.dataset}_{args.rule}"
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    rows = []

    for cond in CONDITIONS:
        patch_idx = None
        if cond["patch"]:
            patch_idx = np.array([img * r + c
                                  for r in range(cond["patch"])
                                  for c in range(cond["patch"])])
            X, X_test, Y, Y_test, _ = load_data(
                args.dataset, "clean", "clean",
                poisoned_patch=cond["patch"], poison_mode=cond["poison_mode"])
        elif cond.get("noise_level") is not None:
            X, X_test, Y, Y_test, _ = load_data(
                args.dataset, "clean", "noise", noise_level=cond["noise_level"])
        else:
            X, X_test, Y, Y_test, _ = load_data(args.dataset, "clean", "clean")
        X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
        X_test = np.asarray(X_test, np.float32)
        y_test = np.asarray(Y_test).argmax(1)
        n = len(X)

        # The control gets a source that asserts the same thing about every
        # sample, so no parameter can differ from any other by provenance.
        opinion = tuple(float(v) for v in args.untrusted_opinion.split(","))
        src = ProvenanceSource(n, args.untrusted_tail, opinion)
        if not cond["provenance"]:
            src.untrusted[:] = True          # everything equally unverified

        arch_str = "_".join(str(h) for h in cond["arch"])
        for seed in args.seeds:
            path = os.path.join(out, "models",
                                f"{cond['key']}_{arch_str}_seed{seed}.pkl")

            Ws, bs = IB.train_model(X, Y, X_test, Y_test, cond["arch"],
                                    args.epochs, seed, path)
            mu = X[~src.untrusted].mean(0) if (~src.untrusted).any() else X.mean(0)
            infl, mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=False,
                                          center=mu)
            m0 = mass[0][:-1]
            live = infl[0][:-1].sum(1) > np.percentile(infl[0][:-1].sum(1), 20)
            r = np.divide(args.evidence * R[0][:-1], m0,
                          out=np.zeros_like(m0), where=m0 > 1e-12)
            s_ = np.divide(args.evidence * S[0][:-1], m0,
                           out=np.zeros_like(m0), where=m0 > 1e-12)
            om = bpq_vec(r, s_, W=2.0)
            score = (om[..., 0] + 0.5 * om[..., 2]).min(1)
            sel, _ = flag_features(score, live, args.k, rule=args.rule)
            # What the flags cost.  On a model with nothing to find this is
            # the false-alarm price, and it is the number that decides whether
            # a non-zero count matters at all.
            acc0 = float(np.mean(ER.forward(Ws, bs, X_test).argmax(1) == y_test))
            if len(sel):
                W_, b_ = ER.prune_features(Ws, bs, sel)
                acc1 = float(np.mean(ER.forward(W_, b_, X_test).argmax(1) == y_test))
            else:
                acc1 = acc0

            auroc = float("nan"); found = None; ranks = None
            if patch_idx is not None and cond["poison_mode"] != "flip":
                pm = np.zeros(len(score), bool); pm[patch_idx] = True
                found = int(np.isin(patch_idx, sel).sum())
                if pm[live].any() and (~pm[live]).any():
                    auroc = float(roc_auc_score(pm[live], -score[live]))
                order = np.argsort(np.where(live, score, np.inf))
                rk = [int(np.where(order == p)[0][0]) for p in patch_idx if live[p]]
                ranks = (min(rk), max(rk)) if rk else None
            rows.append(dict(condition=cond["label"], key=cond["key"], seed=seed,
                             clean_acc=acc0, masked_acc=acc1,
                             flagged=int(len(sel)), trigger_found=found,
                             n_trigger=(len(patch_idx) if patch_idx is not None
                                        else None),
                             auroc=auroc, ranks=ranks))
            au_txt = f"{auroc:.3f}" if auroc == auroc else "--"
            tf_txt = "--" if found is None else str(found)
            print(f"[loc] {cond['label']:<30s} seed {seed}: "
                  f"flagged {len(sel):4d}  trigger {tf_txt}  AUROC {au_txt}"
                  f"  acc {acc0*100:.2f} -> {acc1*100:.2f}")

    print(f"\n  {'condition':<32}{'flagged':>10}{'trigger':>12}{'AUROC':>9}"
          f"{'acc cost':>10}")
    for cond in CONDITIONS:
        rs = [r for r in rows if r["key"] == cond["key"]]
        fl = np.mean([r["flagged"] for r in rs])
        tf = [r["trigger_found"] for r in rs if r["trigger_found"] is not None]
        au = [r["auroc"] for r in rs if r["auroc"] == r["auroc"]]
        cost = np.mean([r["clean_acc"] - r["masked_acc"] for r in rs]) * 100
        tf_s = f"{np.mean(tf):.1f}/{rs[0]['n_trigger']}" if tf else "---"
        au_s = f"{np.mean(au):.3f}" if au else "---"
        print(f"  {cond['label']:<32}{fl:>10.1f}{tf_s:>12}{au_s:>9}{cost:>10.2f}")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=args.dataset, rule=args.rule, k=args.k,
                       untrusted_opinion=args.untrusted_opinion,
                       epochs=args.epochs, seeds=args.seeds, rows=rows), fh,
                  indent=2, default=str)
    print(f"\n[loc] saved {out}/summary.json")


if __name__ == "__main__":
    main()
