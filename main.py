#!/usr/bin/env python3
"""
PaTAS — Parallel Trust Assessment System
=========================================
Unified experiment runner for the PhD dissertation.

This is the single entry point for reproducing all experiments described in
the dissertation.  Each key maps to a specific section:

    5g        Chapters 6 + 7  — 5G Energy-Consumption Classification (main use case)
    bc        §7.8.1 Exp 1    — Breast Cancer Classification          (Table 7.2)
    mnist     §7.8.1 Exp 2    — MNIST Digit Classification            (Table 7.3)
    poisoned  §7.8.1 Exp 3    — Poisoned MNIST                        (Tables 7.4-7.5)
    gtsrb     §6.5            — Balancing Bias on GTSRB               (Reference [5])
    cifar10h  §6.6            — Labeling Bias on CIFAR-10H            (Reference [6])

Requirements
------------
    pip install numpy pandas scikit-learn

Usage
-----
    python main.py 5g                         # main 5G use case (synthetic data)
    python main.py 5g --data-dir ./data       # main 5G use case (real CSVs)
    python main.py bc                         # §7.8.1 Exp 1
    python main.py mnist                      # §7.8.1 Exp 2
    python main.py poisoned                   # §7.8.1 Exp 3
    python main.py gtsrb                      # §6.5
    python main.py cifar10h                   # §6.6
    python main.py all                        # all experiments in sequence
    python main.py all --skip poisoned gtsrb  # skip slow experiments
    python main.py --list                     # list experiments and exit
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import List, Optional


# ---------------------------------------------------------------------------
# Experiment registry
# Each entry: key -> (module, entry-function, one-line description)
# ---------------------------------------------------------------------------
_EXPERIMENTS = {
    "5g": (
        "train",
        "train_and_evaluate",
        "5G Energy-Consumption Classification (Chapters 6 + 7)",
    ),
    "bc": (
        "breast_cancer",
        "run_breast_cancer",
        "Breast Cancer — §7.8.1 Exp 1, Table 7.2",
    ),
    "mnist": (
        "mnist",
        "run_mnist",
        "MNIST Digit Classification — §7.8.1 Exp 2, Table 7.3",
    ),
    "poisoned": (
        "poisoned_mnist",
        "run_poisoned_mnist",
        "Poisoned MNIST — §7.8.1 Exp 3, Tables 7.4-7.5",
    ),
    "gtsrb": (
        "gtsrb",
        "run_gtsrb",
        "Balancing Bias on GTSRB — §6.5, Reference [5]",
    ),
    "cifar10h": (
        "cifar10h",
        "run_cifar10h",
        "Labeling Bias on CIFAR-10H — §6.6, Reference [6]",
    ),
}

_ALL_KEYS: List[str] = list(_EXPERIMENTS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _list_experiments() -> None:
    """Print available experiment keys with their descriptions."""
    print("\nAvailable experiments:\n")
    print(f"  {'Key':<12}  Description")
    print("  " + "-" * 68)
    for key, (_, _, desc) in _EXPERIMENTS.items():
        print(f"  {key:<12}  {desc}")
    print()


def _run_one(key: str, args: argparse.Namespace) -> None:
    """Import and run a single experiment, forwarding relevant CLI options."""
    mod_name, fn_name, desc = _EXPERIMENTS[key]

    print(f"\n{'=' * 78}")
    print(f"  {desc}")
    print(f"{'=' * 78}\n")
    t0 = time.time()

    mod = importlib.import_module(mod_name)
    fn  = getattr(mod, fn_name)

    if key == "5g":
        # The 5G experiment optionally reads real CSVs; fall back to synthetic.
        from data_loader import make_synthetic_5g

        if args.data_dir is not None:
            data_dir = args.data_dir
        else:
            print("No CSV directory given — generating a synthetic 5G dataset.")
            data_dir = make_synthetic_5g(
                n_bs=40, n_hours=120, cells_per_bs=2, seed=args.seed
            )
            print(f"Synthetic CSVs written to: {data_dir}\n")

        fn(data_dir, seed=args.seed)
    else:
        fn(seed=args.seed)

    print(f"\n  [{key}] finished in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

_EPILOG = """\
Experiment keys and dissertation sections
-----------------------------------------
  5g        Chapters 6 + 7  — End-to-end 5G energy-classification pipeline
  bc        §7.8.1 Exp 1    — Table 7.2  (30-16-2 MLP, 9 trust combos, 2 ε)
  mnist     §7.8.1 Exp 2    — Table 7.3  (4 hidden sizes, uncertain X/Y)
  poisoned  §7.8.1 Exp 3    — Tables 7.4-7.5  (patch + label-flip poisoning)
  gtsrb     §6.5            — Balancing Bias  (CUQ + BPQ, Reference [5])
  cifar10h  §6.6            — Labeling Bias   (per-entry opinions, Reference [6])

The 5G use case (key: 5g) reads real CSVs when --data-dir is provided,
or falls back to a built-in synthetic dataset generator otherwise.

All other experiments rely on sklearn built-in datasets — no internet
connection or external download is required.
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "PaTAS — Parallel Trust Assessment System\n"
            "Unified runner for all dissertation experiments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "experiment",
        nargs="?",
        choices=_ALL_KEYS + ["all"],
        metavar="EXPERIMENT",
        help=(
            "Experiment to run: "
            + " | ".join(_ALL_KEYS)
            + " | all.  Use --list to see descriptions."
        ),
    )
    p.add_argument(
        "--data-dir",
        metavar="PATH",
        default=None,
        help=(
            "Path to a directory containing the three 5G CSV files "
            "(BSinfo.csv, CLstat.csv, ECstat.csv).  "
            "Applies only to the '5g' experiment.  "
            "When omitted, a synthetic dataset is generated automatically."
        ),
    )
    p.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="KEY",
        help="Keys to skip when running 'all', e.g. --skip poisoned gtsrb.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="Global random seed passed to every experiment (default: 0).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the list of available experiments with descriptions and exit.",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # --list: print experiment catalogue and exit
    if args.list:
        _list_experiments()
        return 0

    # No experiment specified: show help
    if args.experiment is None:
        parser.print_help()
        print(
            "\nTip: pass an experiment key, e.g.  python main.py 5g\n"
            "     or run  python main.py --list  to see all options.\n"
        )
        return 1

    # Build the list of experiments to run
    keys = _ALL_KEYS if args.experiment == "all" else [args.experiment]
    keys = [k for k in keys if k not in args.skip]

    if not keys:
        print("No experiments to run after applying --skip.")
        return 1

    overall_t0 = time.time()
    for key in keys:
        _run_one(key, args)

    if len(keys) > 1:
        print(
            f"\nAll {len(keys)} experiment(s) completed in "
            f"{time.time() - overall_t0:.1f}s.\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
