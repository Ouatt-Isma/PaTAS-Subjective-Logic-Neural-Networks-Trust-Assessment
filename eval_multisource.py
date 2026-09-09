"""eval_multisource.py — does the opinion algebra beat a scalar influence ratio?

With a SINGLE untrusted source the two are provably equivalent: belief and
disbelief are both functions of the untrusted share of a parameter's influence
mass, so the mapped opinion is a monotone function of that share and the two
rankings coincide.  Reporting the defence as a subjective-logic result would
therefore be unearned.

They can only diverge when sources carry different AMOUNTS of evidence.  This
script builds that setting, which is also the realistic supply-chain one:

    A  verified      (1,0,0)   audited data
    B  unknown       (0,0,1)   an unvetted source, benign, and the ONLY
                               provider of some classes
    C  compromised   (0,1,0)   carries a trigger-patch backdoor

A scalar criterion sees "not verified" and cannot separate B from C, so it
spends its pruning budget on harmless parameters.  An opinion keeps vacuity
and disbelief on separate axes and reaches the backdoor first.

Usage
-----
    python eval_multisource.py --seeds 0 1 2
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


def build_sources(X, y_oh, args, patch_idx, patch_value):
    """Assign sources and poison.  ``--poison-leak f`` is the adaptive
    adversary: a fraction f of the poisoned samples is placed in the VERIFIED
    source instead, modelling an attacker who also compromised part of the
    audited pipeline, or provenance labels that are simply wrong."""
    """Assign every training sample to a source and poison source C."""
    y = y_oh.argmax(1)
    n = len(X)
    src = np.zeros(n, dtype=int)                                   # A verified
    src[int(round((1.0 - args.compromised_frac) * n)):] = 2        # C compromised
    src[(src == 0) & np.isin(y, args.unknown_classes)] = 1         # B unknown
    Xp, yp = X.copy(), y.copy()
    a, b = args.pois_pair
    victims = [i for i in np.where(src == 2)[0] if y[i] in (a, b)]
    leak = getattr(args, "poison_leak", 0.0)
    if leak > 0:
        rng = np.random.default_rng(12345)
        cand = np.array([i for i in np.where(src == 0)[0] if y[i] in (a, b)])
        k = int(round(leak * len(victims)))
        if k and len(cand):
            victims += list(rng.choice(cand, min(k, len(cand)), replace=False))
    for i in victims:
        Xp[i, patch_idx] = patch_value
        yp[i] = b if y[i] == a else a
    return src, Xp, np.eye(y_oh.shape[1], dtype=np.float32)[yp], len(victims)


def get_model(Xp, Yp, X_test, y_test_oh, seed, args, cache_dir):
    """Train (or reuse) the network for one seed."""
    leak = getattr(args, "poison_leak", 0.0)
    path = os.path.join(cache_dir,
                        f"nn_seed{seed}" + (f"_leak{leak:g}" if leak else "") + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            wd = pickle.load(fh)
    else:
        os.environ["PATAS_SEED"] = str(seed)
        from NN.primaryNN import NeuralNetwork
        from main import get_lr_mnist
        nn = NeuralNetwork(Xp.shape[1], hidden_sizes=list(args.arch),
                           output_size=Yp.shape[1], ptas=False, operation=False)
        nn.train(Xp, Yp, X_test, y_test_oh, epochs=args.epochs,
                 batch_size=128, shuffle=True, lr_scheduler=get_lr_mnist)
        os.makedirs(cache_dir, exist_ok=True)
        nn.save_model(path)
        with open(path, "rb") as fh:
            wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1)); i += 1
    return Ws, bs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arch", type=int, nargs="+", default=[128])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--pois-pair", type=int, nargs=2, default=[6, 9])
    ap.add_argument("--compromised-frac", type=float, default=0.25)
    ap.add_argument("--unknown-classes", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--poison-leak", type=float, default=0.0,
                    help="Adaptive adversary: fraction of the poison placed in "
                         "the VERIFIED source, so provenance is partly wrong")
    ap.add_argument("--k", type=float, default=8.0,
                    help="Robust deviations below the median at which a "
                         "feature is flagged; distribution-relative, so it "
                         "needs no per-dataset or per-run calibration")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="Attributed-trust threshold below which a feature is "
                         "flagged; validated on clean models to flag nothing")
    args = ap.parse_args()

    import eval_repair as ER
    from NN.datasets import load_data, mnist_get_scaling
    from attribution import accumulate, MultiSourceProvenance, flag_features
    from subjective_logic import bpq_vec

    X, X_test, Y, y_test_oh, _ = load_data("mnist", "clean", "clean")
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    X_test = np.asarray(X_test, np.float32); y_test = np.asarray(y_test_oh).argmax(1)
    img = int(round(X.shape[1] ** 0.5))
    patch_idx = np.array([img * r + c for r in range(args.patch_size)
                          for c in range(args.patch_size)])
    ER.PATCH_VALUE = mnist_get_scaling(1.0)

    src, Xp, Yp, n_pois = build_sources(X, Y, args, patch_idx, ER.PATCH_VALUE)
    print(f"[multi] A verified {np.mean(src==0)*100:.0f}%   "
          f"B unknown (owns classes {args.unknown_classes}) {np.mean(src==1)*100:.0f}%   "
          f"C compromised {np.mean(src==2)*100:.0f}%   poisoned samples {n_pois}")
    cache = f"results/MultiSource_mnist_{'_'.join(map(str,args.arch))}"
    # The training data depends only on the compromised source, so models are
    # shared across scenarios; summaries are not, and must not collide.
    scen = (f"uc{'-'.join(map(str,args.unknown_classes))}"
            f"_cf{args.compromised_frac:g}_p{args.patch_size}"
            + (f"_leak{args.poison_leak:g}" if args.poison_leak else ""))
    src_obj = MultiSourceProvenance(src, [(1, 0, 0), (0, 0, 1), (0, 1, 0)])

    rows = []
    for seed in args.seeds:
        Ws, bs = get_model(Xp, Yp, X_test, y_test_oh, seed, args, cache)
        acc0, pair0, asr0 = ER.evaluate(Ws, bs, X_test, y_test, patch_idx,
                                        tuple(args.pois_pair))
        print(f"\n[multi] seed {seed}: undefended clean {acc0*100:.2f}%  "
              f"attack success {asr0*100:.2f}%")
        mass, R, S = accumulate(Ws, bs, Xp, Yp, src_obj, verbose=False)
        m0 = mass[0][:-1]
        live = m0.sum(1) > np.percentile(m0.sum(1), 20)
        r = np.divide(args.evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
        s = np.divide(args.evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
        om = bpq_vec(r, s, W=2.0)
        crit = {
            # the opinion: vacuity and disbelief on separate axes
            "opinion": (om[..., 0] + 0.5 * om[..., 2]).min(1),
            # the scalar ablation: verified share of the influence mass
            "scalar": np.divide(R[0][:-1], m0, out=np.zeros_like(m0),
                                where=m0 > 1e-12).min(1),
        }
        n_feat = Ws[0].shape[0]
        for frac in args.budgets:
            k = max(1, int(round(frac * n_feat)))
            for name, score in crit.items():
                sel = np.argsort(np.where(live, score, np.inf))[:k]
                W_, b_ = ER.prune_features(Ws, bs, sel)
                acc, pair, asr = ER.evaluate(W_, b_, X_test, y_test, patch_idx,
                                             tuple(args.pois_pair))
                m = np.isin(y_test, args.unknown_classes)
                bacc = float(np.mean(ER.forward(W_, b_, X_test[m]).argmax(1) == y_test[m]))
                rows.append(dict(seed=seed, criterion=name, budget=frac,
                                 clean_acc=acc, asr=asr, unknown_class_acc=bacc,
                                 trigger_recall=float(np.isin(patch_idx, sel).mean())))
        # Operating rule: the opinion flags what falls below its threshold and
        # that count becomes the budget BOTH criteria are given, so the
        # comparison stays matched while the defender never picks a budget.
        k_thr = int(len(flag_features(crit["opinion"], live, args.k)[0]))
        if k_thr > 0:
            for name, score in crit.items():
                sel = np.argsort(np.where(live, score, np.inf))[:k_thr]
                W_, b_ = ER.prune_features(Ws, bs, sel)
                acc, pair, asr = ER.evaluate(W_, b_, X_test, y_test, patch_idx,
                                             tuple(args.pois_pair))
                m = np.isin(y_test, args.unknown_classes)
                bacc = float(np.mean(ER.forward(W_, b_, X_test[m]).argmax(1) == y_test[m]))
                rows.append(dict(seed=seed, criterion=f"{name}@thr", budget=-1.0,
                                 n_selected=k_thr, clean_acc=acc, asr=asr,
                                 unknown_class_acc=bacc,
                                 trigger_recall=float(np.isin(patch_idx, sel).mean())))
            print(f"[multi] seed {seed}: threshold {args.threshold} flags "
                  f"{k_thr} of {int(live.sum())} live features")
        else:
            print(f"[multi] seed {seed}: threshold {args.threshold} flags nothing "
                  f"(correct when no corruption is localised in input space)")
        rows.append(dict(seed=seed, criterion="undefended", budget=0.0,
                         clean_acc=acc0, asr=asr0,
                         unknown_class_acc=float(np.mean(
                             ER.forward(Ws, bs, X_test[np.isin(y_test, args.unknown_classes)]).argmax(1)
                             == y_test[np.isin(y_test, args.unknown_classes)])),
                         trigger_recall=0.0))

    def agg(crit, frac, key):
        v = [d[key] for d in rows if d["criterion"] == crit and d["budget"] == frac]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2
    print(f"\n{'criterion':<12}{'budget':>7}{'clean acc':>16}{'attack success':>18}"
          f"{'acc on B classes':>19}{'trigger':>9}")
    for frac in [0.0, -1.0] + list(args.budgets):
        for crit in (["undefended"] if frac == 0.0
                     else ["opinion@thr", "scalar@thr"] if frac == -1.0
                     else ["opinion", "scalar"]):
            a, ad = agg(crit, frac, "clean_acc"); z, zd = agg(crit, frac, "asr")
            b, _ = agg(crit, frac, "unknown_class_acc"); t, _ = agg(crit, frac, "trigger_recall")
            lbl = "thr" if frac == -1.0 else f"{frac*100:.0f}%"
            print(f"{crit:<14}{lbl:>6}{a*100:>11.2f}±{ad*100:<4.2f}"
                  f"{z*100:>13.2f}±{zd*100:<4.2f}{b*100:>14.2f}%{t*100:>9.0f}%")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, f"summary_{scen}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(arch=args.arch, epochs=args.epochs, seeds=args.seeds,
                       scenario=scen, threshold=args.threshold,
                       compromised_frac=args.compromised_frac,
                       unknown_classes=args.unknown_classes,
                       patch_size=args.patch_size, rows=rows), fh, indent=2)
    print(f"\n[multi] saved {cache}/summary_{scen}.json")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
