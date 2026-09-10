"""conv_multisource.py: the adaptive adversary at input-position level.

The dense multi-source experiment (eval_multisource.py) is where the opinion
representation is claimed to matter: three sources carrying different amounts
of evidence, and an attacker who also hides poison in the source the defender
believes audited.  This runs the same adversary against the convolutional form
of the method, which attributes to input positions rather than parameters and
neutralises flagged positions at inference.  Same source assignment, same
leakage construction, same criteria, same matched-budget rule.

Criteria, all scored per input position and all given the budget the opinion's
own rule selects:
    opinion      b + u/2 of the position's opinion (ratio of evidence, keeps u)
    scalar       verified share of influence, R / m ("not verified" is one state)
    scalar-proj  influence-weighted mean projected probability,
                 0.5 + 0.5 (R - S) / m  (three states in one number)

Usage
-----
    python conv_multisource.py --poison-leak 0.3 --seeds 0 1 2 3 4 --device cuda
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
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion", "gtsrb"])
    ap.add_argument("--arch", choices=["small", "resnet18"], default="resnet18")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--pois-pair", type=int, nargs=2, default=[6, 9])
    ap.add_argument("--compromised-frac", type=float, default=0.25)
    ap.add_argument("--unknown-classes", type=int, nargs="+", default=[4, 7])
    ap.add_argument("--poison-leak", type=float, default=0.0)
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--k", type=float, default=8.0)
    ap.add_argument("--rule", choices=["mad", "otsu"], default="mad",
                    help="The convolutional experiments use the fixed-deviation rule.")
    ap.add_argument("--live-pct", type=float, default=0.0)
    ap.add_argument("--with-drop", action="store_true",
                    help="Also retrain without the compromised source.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import DATASET_META
    from attribution import flag_features
    from subjective_logic import bpq_vec
    from eval_multisource import build_sources
    from conv_attribution import (ResNet18, SmallCNN, train, accuracy,
                                  input_attribution)

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[cmulti] device {dev}")
    meta = DATASET_META[args.dataset]
    img = meta["img_size"]; pv = meta["scale_patch"](1.0)
    pidx = np.array([img * r + c for r in range(args.patch_size)
                     for c in range(args.patch_size)])
    X, X_test, Y, Y_test, _ = load_data(args.dataset, "clean", "clean")
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    X_test = np.asarray(X_test, np.float32); y_test = np.asarray(Y_test).argmax(1)

    # identical adversary to the dense experiment
    src, Xp, Yp, n_pois = build_sources(X, Y, args, pidx, pv)
    ops = np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0)], np.float32)
    b_n, d_n = ops[src, 0], ops[src, 1]
    print(f"[cmulti] {args.dataset}: sources A {int((src==0).sum())} audited, "
          f"B {int((src==1).sum())} unvetted (classes {args.unknown_classes}), "
          f"C {int((src==2).sum())} compromised; {n_pois} poisoned, "
          f"leak {args.poison_leak:g}")
    mu = Xp[src == 0].mean(0)
    unk = np.isin(y_test, args.unknown_classes)

    out = (f"results/ConvMulti_{args.dataset}_{args.arch}"
           + ("_aug" if args.augment else "") + f"_cf{args.compromised_frac:g}")
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    tag = f"_leak{args.poison_leak:g}" if args.poison_leak else ""

    def new_model():
        Net = ResNet18 if args.arch == "resnet18" else SmallCNN
        w = args.width if args.arch == "small" else max(args.width, 64)
        return Net(img, Y.shape[1], 1, w).to(dev)

    def get_model(Xt, Yt, seed, name):
        path = os.path.join(out, "models", f"{name}_seed{seed}{tag}.pt")
        model = new_model()
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=dev))
        else:
            train(model, Xt, Yt, args.epochs, args.batch, seed, dev,
                  aug=args.augment, img=img, in_ch=1, lr=args.lr)
            torch.save(model.state_dict(), path)
        model.eval()
        return model

    @torch.no_grad()
    def evaluate(model, sel=None):
        """Clean accuracy, accuracy on the unvetted source's classes, and
        attack success, with the flagged positions masked to the trusted mean."""
        def run(Xq):
            Xq = Xq.copy()
            if sel is not None and len(sel):
                Xq[:, sel] = mu[sel]
            return np.concatenate([model(torch.as_tensor(Xq[i:i + 1024], device=dev))
                                   .argmax(1).cpu().numpy()
                                   for i in range(0, len(Xq), 1024)])
        pred = run(X_test)
        acc = float(np.mean(pred == y_test))
        bacc = float(np.mean(pred[unk] == y_test[unk]))
        a, b = args.pois_pair; zs = []
        for s_, d_ in ((a, b), (b, a)):
            Xs = X_test[y_test == s_].copy(); Xs[:, pidx] = pv
            zs.append(float(np.mean(run(Xs) == d_)))
        return acc, bacc, float(np.mean(zs))

    rows = []
    for seed in args.seeds:
        model = get_model(Xp, Yp, seed, "nn")
        acc0, bacc0, asr0 = evaluate(model)
        print(f"\n[cmulti] seed {seed}: undefended clean {acc0*100:.2f}%  "
              f"B-class {bacc0*100:.2f}%  attack {asr0*100:.2f}%")
        rows.append(dict(seed=seed, criterion="undefended", clean_acc=acc0,
                         unknown_class_acc=bacc0, asr=asr0, trigger_recall=0.0))

        infl, mass, R, S = input_attribution(model, Xp, Yp, b_n, d_n, mu, dev)
        live = infl > np.percentile(infl, args.live_pct)
        r = np.divide(args.evidence * R, mass, out=np.zeros_like(mass), where=mass > 1e-12)
        s = np.divide(args.evidence * S, mass, out=np.zeros_like(mass), where=mass > 1e-12)
        om = bpq_vec(r, s, W=2.0)
        crit = {
            "opinion": om[..., 0] + 0.5 * om[..., 2],
            "scalar": np.divide(R, mass, out=np.zeros_like(mass), where=mass > 1e-12),
            "scalar-proj": 0.5 + 0.5 * np.divide(R - S, mass, out=np.zeros_like(mass),
                                                 where=mass > 1e-12),
        }
        k_thr = int(len(flag_features(crit["opinion"], live, args.k, rule=args.rule)[0]))
        print(f"[cmulti] seed {seed}: the opinion's rule selects {k_thr} of "
              f"{int(live.sum())} live positions; every criterion gets that budget")
        for name, score in crit.items():
            if k_thr == 0:
                acc, bacc, asr, rec = acc0, bacc0, asr0, 0.0
                order = np.argsort(np.where(live, score, np.inf))
            else:
                order = np.argsort(np.where(live, score, np.inf))
                sel = order[:k_thr]
                acc, bacc, asr = evaluate(model, sel)
                rec = float(np.isin(pidx, sel).mean())
            rk = [int(np.where(order == p)[0][0]) for p in pidx if live[p]]
            rows.append(dict(seed=seed, criterion=f"{name}@thr", n_selected=k_thr,
                             clean_acc=acc, unknown_class_acc=bacc, asr=asr,
                             trigger_recall=rec,
                             trigger_ranks=[min(rk), max(rk)] if rk else None))
            print(f"[cmulti]   {name:<12} clean {acc*100:.2f}%  B-class {bacc*100:.2f}%  "
                  f"attack {asr*100:.2f}%  trigger {rec*100:.0f}%  "
                  f"ranks {min(rk) if rk else -1}-{max(rk) if rk else -1}")

        if args.with_drop:
            keep = src != 2
            md = get_model(Xp[keep], Yp[keep], seed, "drop-compromised")
            acc, bacc, asr = evaluate(md)
            rows.append(dict(seed=seed, criterion="drop-compromised", clean_acc=acc,
                             unknown_class_acc=bacc, asr=asr, trigger_recall=float("nan")))
            print(f"[cmulti]   drop-compromised (retrained) clean {acc*100:.2f}%  "
                  f"attack {asr*100:.2f}%")

    def agg(c, key):
        v = [d[key] for d in rows if d["criterion"] == c and d[key] == d[key]]
        return (float(np.mean(v)) * 100, float(np.std(v)) * 100) if v else (float("nan"),) * 2
    print(f"\n  leak {args.poison_leak:g}   {'criterion':<18}{'clean':>14}{'attack':>16}{'trigger':>9}")
    for c in ["undefended", "opinion@thr", "scalar@thr", "scalar-proj@thr"] \
             + (["drop-compromised"] if args.with_drop else []):
        a, ad = agg(c, "clean_acc"); z, zd = agg(c, "asr"); t, _ = agg(c, "trigger_recall")
        print(f"  {'':10}{c:<18}{a:>9.2f}±{ad:<4.2f}{z:>11.2f}±{zd:<4.2f}{t:>8.0f}%")
    with open(os.path.join(out, f"summary{tag or '_leak0'}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(vars(args), rows=rows), fh, indent=2, default=str)
    print(f"[cmulti] saved {out}/summary{tag or '_leak0'}.json")


if __name__ == "__main__":
    main()
