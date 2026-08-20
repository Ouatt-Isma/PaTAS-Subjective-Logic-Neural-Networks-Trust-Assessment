"""eval_dead_units.py — is the model-trust audit a dead-unit artifact?

Reviewer concern (round 3, Major 1): the flat trust drop under label noise
could be driven by inactive ("dead") hidden units whose weights receive few
confident gradient updates and therefore keep near-vacuous opinions; the
audit would then measure network deadness rather than training corruption.

For the same cached models as eval_model_trust.py (one per label-flip rate,
identical settings so no retraining happens), this script measures:

  1. per-hidden-unit activation frequency on the audit's test subset
     (fraction of inputs with positive post-ReLU output);
  2. per-unit trust: the mean projected probability b + a*u of the unit's
     incoming weight opinions (columns of the layer's opinion matrix);
  3. the dead-unit share per rate, the trust split between dead and active
     units, and the frequency-trust correlation;
  4. the audit signal recomputed with activity-weighted opinion feedforward
     (TensorArrayTO.weighted_dot: each unit's contribution weighted by its
     own activation frequency, bias weight 1), which removes dead units'
     contribution from the aggregate by construction.

If the clean-vs-corrupted trust margin survives (4), and active units'
trust itself drops with the flip rate, the audit signal lives in the live
paths of the network and the dead-unit confound is refuted.

Usage
-----
    python eval_dead_units.py --dataset mnist --arch 512
    (defaults mirror eval_model_trust.py so the cached sweep is reused)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import multiprocessing

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.abspath(__file__))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # noqa: F401 — path bootstrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_uq_comparison import (
    DATASET_CFG, _arch_str, _DEFAULT_EPS,
    CLEAN_CONDITION, label_noise_condition,
    get_base_mlp, ensure_patas_cache, load_offline_ptas,
)
from eval_model_trust import pearson, spearman


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def unit_activation_freqs(base, X: np.ndarray, chunk: int = 1000) -> list[np.ndarray]:
    """Per-hidden-layer activation frequency: fraction of the inputs on
    which each unit's post-ReLU output is strictly positive."""
    sums, n = None, 0
    for i in range(0, len(X), chunk):
        base.forward(X[i:i + chunk])
        hs = base._activations[1:-1]           # post-ReLU hidden layers
        act = [(np.asarray(h.detach().cpu().numpy()) > 0).sum(0) for h in hs]
        sums = act if sums is None else [s + a for s, a in zip(sums, act)]
        n += len(X[i:i + chunk])
    return [(s / n).astype(np.float64) for s in sums]


def unit_trusts(ptas) -> list[np.ndarray]:
    """Per-hidden-layer unit trust: mean projected probability of the
    incoming weight opinions (columns of omega, bias row included)."""
    out = []
    for om in ptas.omega_thetas[:-1]:          # all layers feeding hidden units
        v = np.asarray(om.to_numpy(), dtype=np.float64)    # (in+1, units, 3)
        pp = v[..., 0] + 0.5 * v[..., 2]
        out.append(pp.mean(axis=0))
    return out


def feedforward_trust(ptas, input_dim: int,
                      freqs: list[np.ndarray] | None = None) -> float:
    """Projected probability of the aggregated output opinion under a fully
    trusted input. With ``freqs``, layer l >= 1 uses activity-weighted
    opinion propagation (unit rows weighted by their own activation
    frequency, bias row weight 1); with freqs=None this reproduces the
    standard audit feedforward exactly (weighted_dot with unit weights
    equals dot)."""
    from NN.PTAStemplate import PTAS
    from concrete.TensorTO import TensorArrayTO, fill as tfill, to_numpy

    omegas = [TensorArrayTO(np.asarray(om.to_numpy(), dtype=np.float32))
              for om in ptas.omega_thetas]
    cur = TensorArrayTO(tfill((1, input_dim), method="trust"))
    one = tfill((1, 1), method="one")
    for l, om in enumerate(omegas):
        vb = TensorArrayTO(np.concatenate([np.asarray(cur.value), one], axis=1))
        if freqs is not None and l >= 1:
            row_w = np.concatenate([freqs[l - 1].astype(np.float32),
                                    np.ones(1, dtype=np.float32)])
            cur = TensorArrayTO.weighted_dot(vb, om, row_w)
        else:
            cur = TensorArrayTO.dot(vb, om)
    agg = to_numpy(PTAS.aggregation(cur))
    return float(agg[0] + 0.5 * agg[2])


DEAD_THR = 0.01      # active on < 1% of inputs: dead
LOW_THR = 0.05       # active on < 5% of inputs: low-activity


def eval_rate(dataset: str, rate: float, args) -> dict | None:
    from NN.datasets import load_data

    cond = CLEAN_CONDITION if rate == 0 else label_noise_condition(rate)
    cfgd = DATASET_CFG[dataset]
    arch = cfgd["arch"]
    print(f"\n{'='*70}\n  Dead-unit audit — {dataset} ({_arch_str(arch)})  "
          f"label-flip rate p={rate:g}\n{'='*70}")

    y_how = "clean" if rate == 0 else "noise"
    _load_kwargs = {} if cond.noise_level is None else {"noise_level": cond.noise_level}
    X_train, X_test, y_train, y_test, _ = load_data(dataset, "clean", y_how,
                                                     **_load_kwargs)
    rng = np.random.default_rng(args.seed)
    n_sub = min(args.subset, len(X_test))
    idx = np.sort(rng.choice(len(X_test), size=n_sub, replace=False))
    Xs, ys = X_test[idx], y_test.argmax(1)[idx]

    base = get_base_mlp(dataset, arch, X_train, y_train, X_test, y_test,
                        args.epochs, args.train_missing,
                        cond.x_trust, cond.y_trust,
                        noise_level=cond.noise_level)
    if not ensure_patas_cache(dataset, arch, args.eps, args.epochs,
                              args.train_missing, fuse_method=args.fuse_method,
                              x_trust=cond.x_trust, y_trust=cond.y_trust,
                              noise_level=cond.noise_level,
                              y_dataset=y_how if rate > 0 else None):
        print("  [DeadUnits] PaTAS cache unavailable — skipping rate.")
        return None
    ptas = load_offline_ptas(dataset, arch, args.eps,
                             fuse_method=args.fuse_method,
                             x_trust=cond.x_trust, y_trust=cond.y_trust,
                             noise_level=cond.noise_level)

    probs = base.forward(Xs)
    acc = float(np.mean(probs.argmax(1) == ys))

    freqs = unit_activation_freqs(base, Xs)
    trusts = unit_trusts(ptas)
    f_all = np.concatenate(freqs)
    t_all = np.concatenate(trusts)
    dead = f_all < DEAD_THR
    low = f_all < LOW_THR

    row = {
        "rate": rate, "test_acc": acc,
        "n_units": int(f_all.size),
        "dead_frac": float(dead.mean()),
        "lowact_frac": float(low.mean()),
        "mean_freq": float(f_all.mean()),
        "trust_dead": float(t_all[dead].mean()) if dead.any() else float("nan"),
        "trust_active": float(t_all[~low].mean()) if (~low).any() else float("nan"),
        "corr_freq_trust_pearson": pearson(f_all, t_all),
        "corr_freq_trust_spearman": spearman(f_all, t_all),
        "ff_trust": feedforward_trust(ptas, cfgd["input_dim"]),
        "ff_trust_actweighted": feedforward_trust(ptas, cfgd["input_dim"],
                                                  freqs=freqs),
        "unit_freqs": f_all.round(4).tolist(),
        "unit_trusts": t_all.round(4).tolist(),
    }
    print(f"  p={rate:g}: acc={acc*100:.2f}%  dead(<{DEAD_THR:g})="
          f"{row['dead_frac']*100:.1f}%  low(<{LOW_THR:g})="
          f"{row['lowact_frac']*100:.1f}%  trust dead/active="
          f"{row['trust_dead']:.4f}/{row['trust_active']:.4f}  "
          f"r(freq,trust)={row['corr_freq_trust_pearson']:+.3f}  "
          f"ff={row['ff_trust']:.4f}  ff_actw={row['ff_trust_actweighted']:.4f}")
    return row


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_table(rows: list[dict], dataset: str, out_path: str) -> None:
    def _f(v, fmt="{:.3f}"):
        return fmt.format(v) if v is not None and np.isfinite(v) else "--"

    t0w = rows[0]["ff_trust_actweighted"] if rows and rows[0]["rate"] == 0 else None
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Dead-unit confound check on " + dataset.upper() +
        r": per label-flip rate $p$, the share of hidden units active on "
        r"fewer than 1\% (dead) and 5\% (low) of the test inputs, the mean "
        r"unit trust (projected probability of the unit's incoming weight "
        r"opinions) split between dead and active units, the "
        r"frequency--trust correlation across units, and the audit trust "
        r"recomputed with activity-weighted opinion feedforward (each "
        r"unit's contribution weighted by its own activation frequency, so "
        r"dead units are removed from the aggregate by construction). "
        r"$\Delta_{\mathrm{rel}}$: relative drop vs.\ the $p{=}0$ model.}",
        rf"\label{{tab:deadunits-{dataset}}}",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$p$ & Acc.\ (\%) & Dead (\%) & Low (\%) & "
        r"Trust$_{\mathrm{dead}}$ & Trust$_{\mathrm{active}}$ & "
        r"$r$(freq, trust) & FF trust (act.-w.) & "
        r"$\Delta_{\mathrm{rel}}$ act.-w.\ (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        drel = ((t0w - r["ff_trust_actweighted"]) / t0w * 100
                if t0w and r["rate"] > 0 else 0.0)
        lines.append(
            f"{r['rate']:g} & {_f(r['test_acc']*100, '{:.2f}')} & "
            f"{_f(r['dead_frac']*100, '{:.1f}')} & "
            f"{_f(r['lowact_frac']*100, '{:.1f}')} & "
            f"{_f(r['trust_dead'])} & {_f(r['trust_active'])} & "
            f"{_f(r['corr_freq_trust_pearson'])} & "
            f"{_f(r['ff_trust_actweighted'])} & "
            f"{_f(drel, '{:.1f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Saved {out_path}")


def plot_rows(rows: list[dict], out_dir: str, dataset: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150})
    rates = [r["rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(rates, [r["ff_trust"] for r in rows], marker="o", ms=5, lw=2,
            color="#0072B2", label="FF trust (standard audit)")
    ax.plot(rates, [r["ff_trust_actweighted"] for r in rows], marker="s",
            ms=5, lw=2, color="#D55E00", ls="--",
            label="FF trust (activity-weighted)")
    ax.plot(rates, [r["trust_active"] for r in rows], marker="^", ms=5, lw=2,
            color="#009E73", ls="-.", label="Mean trust, active units")
    ax.plot(rates, [r["dead_frac"] for r in rows], marker="v", ms=5, lw=2,
            color="#555555", ls=":", label="Dead-unit fraction")
    ax.set_xlabel("Train-time label-flip rate $p$")
    ax.set_ylabel("Trust / fraction")
    ax.set_xticks(rates)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title(f"{dataset.upper()} — audit trust vs. unit deadness")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"deadunit_audit.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)

    # frequency-vs-trust scatter for the extreme rates
    lo = rows[0]
    hi = rows[-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(lo["unit_freqs"], lo["unit_trusts"], s=8, alpha=0.4,
               color="#0072B2", label=f"$p={lo['rate']:g}$")
    ax.scatter(hi["unit_freqs"], hi["unit_trusts"], s=8, alpha=0.4,
               color="#D55E00", label=f"$p={hi['rate']:g}$")
    ax.axvline(DEAD_THR, color="#555555", ls=":", lw=1)
    ax.set_xlabel("Unit activation frequency")
    ax.set_ylabel("Unit trust (mean incoming projected probability)")
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(fontsize=10)
    ax.set_title(f"{dataset.upper()} — unit activity vs. unit trust")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"freq_vs_trust.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir}/deadunit_audit.pdf and freq_vs_trust.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=["mnist", "fashion", "gtsrb"],
                   default="mnist")
    p.add_argument("--arch", type=int, nargs="+", default=None)
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eps", type=float, default=_DEFAULT_EPS)
    p.add_argument("--fuse-method",
                   choices=["average", "cumulative", "weighted", "compromise",
                            "constraint"], default="average")
    p.add_argument("--subset", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-missing", action="store_true")
    return p.parse_args()


def main():
    from uq_methods import print_device_info
    print_device_info()
    args = parse_args()
    dataset = args.dataset
    if args.arch:
        DATASET_CFG[dataset]["arch"] = tuple(args.arch)
    arch = DATASET_CFG[dataset]["arch"]

    out_dir = f"results/DeadUnits_{dataset}_{_arch_str(arch)}"
    os.makedirs(out_dir, exist_ok=True)

    rows = [r for r in (eval_rate(dataset, float(p), args)
                        for p in sorted(set(args.rates))) if r is not None]
    if not rows:
        print("No rates evaluated (caches missing?)")
        return

    print(f"\n  {'p':>5} {'acc':>8} {'dead%':>7} {'low%':>7} "
          f"{'t_dead':>8} {'t_act':>8} {'r(f,t)':>8} {'ff':>8} {'ff_actw':>8}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['rate']:>5g} {r['test_acc']*100:7.2f}% "
              f"{r['dead_frac']*100:6.1f}% {r['lowact_frac']*100:6.1f}% "
              f"{r['trust_dead']:8.4f} {r['trust_active']:8.4f} "
              f"{r['corr_freq_trust_pearson']:+8.3f} "
              f"{r['ff_trust']:8.4f} {r['ff_trust_actweighted']:8.4f}")

    plot_rows(rows, out_dir, dataset)
    write_table(rows, dataset, os.path.join(out_dir, "table.tex"))
    slim = [{k: v for k, v in r.items()
             if k not in ("unit_freqs", "unit_trusts")} for r in rows]
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": dataset, "arch": _arch_str(arch),
                   "eps": args.eps, "epochs": args.epochs,
                   "subset": args.subset, "dead_thr": DEAD_THR,
                   "low_thr": LOW_THR, "rows": slim}, fh, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "unit_data.npz"),
        **{f"rate{r['rate']:g}_{k}": np.asarray(r[k])
           for r in rows for k in ("unit_freqs", "unit_trusts")})
    print(f"  Saved {out_dir}/summary.json and unit_data.npz")
    print("\n=== Dead-unit audit complete ===\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    main()
