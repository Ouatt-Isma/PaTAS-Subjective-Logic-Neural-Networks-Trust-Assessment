"""
Cancer dataset PTAS integration test — full scenario sweep.

Run standalone (no pytest needed):
    python tests/test_cancer.py

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_cancer.py -m integration -s

Standalone examples:
    python tests/test_cancer.py                         # all 9 combos × 2 epsilons
    python tests/test_cancer.py --epochs 5             # quick smoke-test
    python tests/test_cancer.py --xtrust trust --ytrust trust --epsilon 0.1
    python tests/test_cancer.py --no-ptas              # baseline NN, no PTAS
    python tests/test_cancer.py --mode server          # only start PTAS side
    python tests/test_cancer.py --mode client          # only start NN side

Default sweep output (9 rows, trust mass per epsilon, one accuracy column each):

  x_trust    y_trust    TM(ε=0.1)  TM(ε=0.01)  Train Acc   Test Acc
  trust      trust        0.8787     0.XXXX      98.68%      96.49%
  ...

Trust Mass definition
---------------------
Trust Mass = trust component of PTAS.aggregation(apply_feedforward(fully-trusted-input))
This is the scalar [t, d, u][0] that ptas_evaluation prints as
"Apply Feed Forward on fully Trusted Input → Aggregated Value".
It represents how much the trained PTAS-network trusts its own output
when the input is fully trusted.

NN accuracy is epsilon-independent (epsilon only drives PTAS weight updates,
never back-propagated into the NN), so Train/Test accuracy appears once per
(x_trust, y_trust) pair rather than once per epsilon.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing
from itertools import product
from typing import Any

# ── Path bootstrap (works with or without pip install) ───────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # triggers patas_module/__init__.py path bootstrap
from main import (  # main.py lives inside patas_module/
    TestCaseConfig,
    get_lr_cancer,
    start_ptas,
    start_client,
    build_trust_generator,
)

try:
    import pytest
except ImportError:  # pytest not installed — mark decorators become no-ops
    class _FakeMark:  # type: ignore[no-redef]
        @staticmethod
        def integration(fn):
            return fn
    class _FakePytest:  # type: ignore[no-redef]
        mark = _FakeMark()
    pytest = _FakePytest()  # type: ignore[assignment]

# ── Scenario constants ────────────────────────────────────────────────────────
TRUST_SPECS: list[str] = ["trust", "vacuous", "distrust"]
EPSILONS: list[float] = [0.1, 0.01]
_DEFAULT_PORT = 5030

# Two extra scenarios beyond the standard 3×3 grid.
# Each entry is a dict of keyword arguments forwarded to make_cancer_cfg.
# "label" is used only for display / caching; "desc" explains the scenario.
EXTRA_SCENARIOS: list[dict[str, Any]] = [
    {
        "label":     "partial_uncertain",
        "x_trust":   "0.25,0.25,0.5",
        "y_trust":   "0.25,0.25,0.5",
        "x_dataset": "noise",       # same degradation as vacuous/vacuous
        "y_dataset": "noise",
        "desc":      "(0.25,0.25,0.5) assessment on noisy features+labels",
    },
    {
        "label":     "mild_degradation",
        "x_trust":   "0.25,0.0,0.75",
        "y_trust":   "trust",
        "x_dataset": "noise_mild",  # features perturbed with noise_prob=0.15
        "y_dataset": "clean",
        "desc":      "(0.25,0,0.75) assessment on mildly degraded features (p=0.15)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_cancer_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = 0.1,
    epochs: int = 15,
    port: int = _DEFAULT_PORT,
    no_round: int | None = None,
    x_dataset: str | None = None,
    y_dataset: str | None = None,
) -> TestCaseConfig:
    """
    Build a TestCaseConfig for the cancer dataset.

    ``x_dataset`` / ``y_dataset`` override the automatic dataset-variant
    selection that normally derives from ``x_trust`` / ``y_trust``.
    Use them when the trust *assessment* and the dataset *degradation* diverge,
    e.g. custom-triplet trust on a noise or noise_mild dataset.
    """
    return TestCaseConfig(
        dataset="cancer",
        input_dim=30,
        output_dim=2,
        hidden_dim=16,
        epochs=epochs,
        batch_size=64,
        learning_rate=get_lr_cancer,
        epsilon_low=epsilon_low,
        x_trust=x_trust,
        y_trust=y_trust,
        port=port,
        mnist_patch_size=None,
        mnist_poisoned_soph=False,
        no_round=no_round,
        x_dataset=x_dataset,
        y_dataset=y_dataset,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Trust-mass helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_trust_mass(ptas_obj: Any, input_dim: int) -> float:
    """
    Final Trust Mass = trust component of the aggregated feedforward output
    when passing a fully-trusted input through the trained PTAS network.

    Concretely: PTAS.aggregation(ptas.apply_feedforward(trusted_input))[0]

    This is identical to what ptas_evaluation prints as:
        "Apply Feed Forward on fully Trusted Input"
        "Aggregated Value:  [t, d, u]"   ← we return t.

    A fully-trusted input (t=1, d=0, u=0 for every feature) through the
    trained omega_thetas tells us: "given completely trustworthy data, how
    much does the PTAS believe in its own conclusion?"
    """
    from concrete.ArrayTO import ArrayTO
    from concrete.TrustOpinion import TrustOpinion
    from NN.PTAStemplate import PTAS as PTASClass

    try:
        trusted_input = ArrayTO(TrustOpinion.fill((1, input_dim), method="trust"))
        a = ptas_obj.apply_feedforward(trusted_input)
        agg = PTASClass.aggregation(a)   # shape (3,): [t, d, u]
        return float(agg[0])             # trust component
    except Exception:
        # Fallback: weighted mean trust of weight tensors
        total_t, total_n = 0.0, 0
        for layer in ptas_obj.omega_thetas:
            v = layer.value
            if isinstance(v, np.ndarray) and v.ndim == 3:
                total_t += float(v[..., 0].sum())
                total_n += v[..., 0].size
        return total_t / total_n if total_n > 0 else float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics file reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_metrics(metrics_path: str) -> dict[str, float]:
    """Parse a metrics.txt file written by primaryNN.train(plot=True)."""
    result: dict[str, float] = {"train_acc": float("nan"), "test_acc": float("nan")}
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("Train:"):
                    result["train_acc"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("Test:"):
                    result["test_acc"] = float(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker functions
# ─────────────────────────────────────────────────────────────────────────────

def _ptas_worker(cfg: TestCaseConfig, result_queue: "multiprocessing.Queue[dict]",
                 ready_event=None) -> None:
    """
    PTAS server subprocess.
    Runs start_ptas(cfg) — which writes evaluation files and returns the
    trained PTAS object — then computes the final trust mass as the trust
    component of the aggregated feedforward output on trusted input.
    ``ready_event`` is set by run_chunk() once the socket is bound/listening.
    """
    try:
        ptas = start_ptas(cfg, ready_event=ready_event)
        trust_mass = compute_trust_mass(ptas, cfg.input_dim) if ptas is not None else float("nan")
        result_queue.put({"trust_mass": trust_mass})
    except Exception as exc:
        result_queue.put({"trust_mass": float("nan"), "error": str(exc)})


def _client_worker(cfg: TestCaseConfig, result_queue: "multiprocessing.Queue[dict]") -> None:
    """
    NN client subprocess.
    Runs start_client(cfg), then reads the metrics.txt written by training.
    """
    try:
        start_client(cfg, not_ptas=False)
        datapath = (
            f"results/NN_Train_{cfg.dataset}_{cfg.hidden_dim}"
            f"_{cfg.x_trust}_{cfg.y_trust}_PathSize_None"
        )
        metrics = _read_metrics(os.path.join(datapath, "metrics.txt"))
        result_queue.put(metrics)
    except Exception as exc:
        result_queue.put({"train_acc": float("nan"), "test_acc": float("nan"), "error": str(exc)})

# ─────────────────────────────────────────────────────────────────────────────
# Single-scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(cfg: TestCaseConfig) -> dict[str, Any]:
    """
    Launch PTAS server + NN client in separate processes, wait for both to
    finish, and return a result dict with trust_mass, train_acc, test_acc.

    Uses a multiprocessing.Event so the client only starts after PTAS has
    bound its socket — eliminates the cold-start race on Windows.
    """
    ptas_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    ptas_proc = multiprocessing.Process(target=_ptas_worker, args=(cfg, ptas_q, ready_event))
    ptas_proc.start()

    ready_event.wait(timeout=60)

    client_proc = multiprocessing.Process(target=_client_worker, args=(cfg, client_q))
    client_proc.start()

    # Read queues BEFORE joining to avoid Windows pipe-buffer deadlock
    try:
        ptas_result = ptas_q.get(timeout=300)
    except Exception:
        ptas_result = {}
    try:
        client_result = client_q.get(timeout=300)
    except Exception:
        client_result = {}

    client_proc.join(timeout=60)
    ptas_proc.join(timeout=60)

    return {
        "x_trust":    cfg.x_trust,
        "y_trust":    cfg.y_trust,
        "epsilon":    cfg.epsilon_low,
        "trust_mass": ptas_result.get("trust_mass", float("nan")),
        "train_acc":  client_result.get("train_acc", float("nan")),
        "test_acc":   client_result.get("test_acc", float("nan")),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep — deduplicating NN runs
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    cfg: TestCaseConfig,
    nn_cache: "dict[tuple, dict[str, float]]",
    *,
    label: str,
) -> "tuple[dict[str, Any], dict[str, float]]":
    """
    Run a single (cfg, epsilon) scenario, re-using cached NN accuracy when
    available (NN accuracy is epsilon-independent).

    Returns ``(result_dict, updated_nn_cache_entry)``.
    """
    # Cache key: everything that affects dataset/accuracy (not epsilon/port)
    nn_key = (cfg.x_trust, cfg.y_trust, cfg.x_dataset, cfg.y_dataset)

    print(f"\n  ► {label}")

    if nn_key in nn_cache:
        # Only re-run PTAS; reuse cached NN accuracy
        ptas_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
        ready_event = multiprocessing.Event()
        ptas_proc = multiprocessing.Process(target=_ptas_worker, args=(cfg, ptas_q, ready_event))
        ptas_proc.start()

        ready_event.wait(timeout=60)

        client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
        client_proc = multiprocessing.Process(target=_client_worker, args=(cfg, client_q))
        client_proc.start()

        # Read queue BEFORE joining (Windows pipe deadlock prevention)
        try:
            ptas_result = ptas_q.get(timeout=300)
        except Exception:
            ptas_result = {}
        client_result = nn_cache[nn_key]   # use cache; discard client_q

        client_proc.join(timeout=60)
        ptas_proc.join(timeout=60)
    else:
        # Full run (PTAS + NN)
        result_full  = run_scenario(cfg)
        ptas_result  = {"trust_mass": result_full["trust_mass"]}
        client_result = {
            "train_acc": result_full["train_acc"],
            "test_acc":  result_full["test_acc"],
        }
        nn_cache[nn_key] = client_result

    result: dict[str, Any] = {
        "x_trust":    cfg.x_trust,
        "y_trust":    cfg.y_trust,
        "x_dataset":  cfg.x_dataset,
        "y_dataset":  cfg.y_dataset,
        "epsilon":    cfg.epsilon_low,
        "trust_mass": ptas_result.get("trust_mass", float("nan")),
        "train_acc":  client_result.get("train_acc", float("nan")),
        "test_acc":   client_result.get("test_acc", float("nan")),
        "group":      "extra",   # overridden for standard grid rows below
    }
    tm, tr, te = result["trust_mass"], result["train_acc"], result["test_acc"]
    print(f"    trust_mass={tm:.4f}  train={tr*100:.2f}%  test={te*100:.2f}%")
    return result, client_result


def run_all_scenarios(
    epochs: int = 15,
    port: int = _DEFAULT_PORT,
    epsilons: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Run the full scenario sweep and return a flat list of result dicts.

    Grid scenarios (9 combos × 2 epsilons)
    ────────────────────────────────────────
    All (x_trust × y_trust) combinations from TRUST_SPECS for each epsilon.
    NN accuracy is cached after the first epsilon so the NN is not retrained
    for subsequent epsilons with the same (x_trust, y_trust) pair.

    Extra scenarios (2 × 2 epsilons)
    ──────────────────────────────────
    1. partial_uncertain  – (0.25, 0.25, 0.5) assessment, noise dataset
    2. mild_degradation   – (0.25, 0, 0.75) assessment, noise_mild dataset
    """
    if epsilons is None:
        epsilons = EPSILONS

    results: list[dict[str, Any]] = []
    nn_cache: dict[tuple, dict[str, float]] = {}   # keyed by (xtrust, ytrust, xds, yds)

    combos = list(product(TRUST_SPECS, TRUST_SPECS))

    # ── Standard 3×3 grid ────────────────────────────────────────────────────
    for eps_idx, eps in enumerate(epsilons):
        print(f"\n{'='*64}")
        print(f"  Standard grid  |  ε = {eps}  ({eps_idx+1}/{len(epsilons)})")
        print(f"{'='*64}")

        for xtrust, ytrust in combos:
            cfg = make_cancer_cfg(xtrust, ytrust, epsilon_low=eps,
                                  epochs=epochs, port=port)
            lbl = f"x={xtrust:<8}  y={ytrust:<8}  ε={eps}"
            result, _ = _run_one(cfg, nn_cache, label=lbl)
            result["group"] = "grid"
            results.append(result)
            time.sleep(2)

    # ── Extra scenarios ───────────────────────────────────────────────────────
    for eps_idx, eps in enumerate(epsilons):
        print(f"\n{'='*64}")
        print(f"  Extra scenarios  |  ε = {eps}  ({eps_idx+1}/{len(epsilons)})")
        print(f"{'='*64}")

        for sc in EXTRA_SCENARIOS:
            cfg = make_cancer_cfg(
                x_trust   = sc["x_trust"],
                y_trust   = sc["y_trust"],
                x_dataset = sc["x_dataset"],
                y_dataset = sc["y_dataset"],
                epsilon_low = eps,
                epochs    = epochs,
                port      = port,
            )
            lbl = f"{sc['label']:<22}  ε={eps}  — {sc['desc']}"
            result, _ = _run_one(cfg, nn_cache, label=lbl)
            result["group"] = "extra"
            results.append(result)
            time.sleep(2)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Table printer — compact side-by-side epsilon comparison
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict[str, Any]]) -> None:
    """
    Print a compact summary table with two Trust-Mass columns (one per ε)
    and a single accuracy column pair (NN accuracy is ε-independent).

    Standard grid rows and extra-scenario rows are separated by a blank line.

    Example layout:

        x_trust           y_trust    TM(ε=0.1)  TM(ε=0.01)  Train Acc   Test Acc
        trust             trust         0.8787      0.XXXX     98.68%     96.49%
        ...
        ── extra ──
        0.25,0.25,0.5     0.25,0.25,0.5   0.XXXX  0.XXXX     63.52%     71.93%
        0.25,0.0,0.75     trust            0.XXXX  0.XXXX     XX.XX%     XX.XX%
    """
    from collections import defaultdict

    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "   N/A "

    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else "  N/A  "

    # Collect unique epsilons in order of appearance
    epsilons_seen: list[float] = []
    for r in results:
        if r["epsilon"] not in epsilons_seen:
            epsilons_seen.append(r["epsilon"])

    # Build per-(xtrust, ytrust) rows indexed by epsilon; preserve insertion order
    rows: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r["x_trust"], r["y_trust"])
        if key not in rows:
            rows[key] = {"eps_data": {}, "group": r.get("group", "grid")}
        rows[key]["eps_data"][r["epsilon"]] = r

    # Column widths — x_trust/y_trust wide enough for triplet strings
    tm_headers = [f"TM(ε={e})" for e in epsilons_seen]
    headers = ["x_trust", "y_trust"] + tm_headers + ["Train Acc", "Test Acc"]
    col_w   = [16, 16] + [12] * len(tm_headers) + [10, 10]

    parts   = [f"{{:<{w}}}" for w in col_w[:2]] + [f"{{:>{w}}}" for w in col_w[2:]]
    row_fmt = "  ".join(parts)
    sep     = "-" * (sum(col_w) + 2 * (len(col_w) - 1))
    thin    = "·" * len(sep)

    print()
    print("=" * len(sep))
    print("  CANCER PTAS — Full Scenario Results")
    print("  Trust Mass = aggregated output trust on fully-trusted input")
    print("  (= 'Apply Feed Forward on trusted input → Aggregated Value[0]')")
    print("=" * len(sep))
    print()
    print("  Note: Train/Test accuracy is ε-independent (epsilon only drives")
    print("        PTAS weight updates, never the NN gradient step).")
    print()
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    prev_group = None
    for (xtrust, ytrust), info in rows.items():
        grp      = info["group"]
        eps_data = info["eps_data"]

        # Print thin separator + label between grid and extra rows
        if prev_group == "grid" and grp == "extra":
            print(thin)
            print(f"  ── extra scenarios ──")
            print(thin)
        prev_group = grp

        tm_vals  = [_f4(eps_data.get(e, {}).get("trust_mass", float("nan")))
                    for e in epsilons_seen]
        any_r    = next(iter(eps_data.values()))
        row_vals = ([xtrust, ytrust]
                    + tm_vals
                    + [_pct(any_r["train_acc"]), _pct(any_r["test_acc"])])
        print(row_fmt.format(*row_vals))

    print(sep)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests (single-scenario, for CI)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_cancer_trust_trust():
    """Cancer: fully trusted X and Y (ε=0.1)."""
    cfg = make_cancer_cfg(x_trust="trust", y_trust="trust", epsilon_low=0.1, epochs=2, port=5020)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert result["trust_mass"] > 0.5, f"Low trust mass: {result['trust_mass']}"


@pytest.mark.integration
def test_cancer_distrust_trust():
    """Cancer: distrusted X, trusted Y (ε=0.1)."""
    cfg = make_cancer_cfg(x_trust="distrust", y_trust="trust", epsilon_low=0.1, epochs=2, port=5021)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_cancer_vacuous_vacuous():
    """Cancer: vacuous X and Y (ε=0.1)."""
    cfg = make_cancer_cfg(x_trust="vacuous", y_trust="vacuous", epsilon_low=0.1, epochs=2, port=5022)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_cancer_partial_uncertain():
    """Cancer: (0.25,0.25,0.5) assessment on noise dataset for both X and Y."""
    cfg = make_cancer_cfg(
        x_trust="0.25,0.25,0.5", y_trust="0.25,0.25,0.5",
        x_dataset="noise", y_dataset="noise",
        epsilon_low=0.1, epochs=2, port=5024,
    )
    result = run_scenario(cfg)
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass out of range: {result['trust_mass']}"


@pytest.mark.integration
def test_cancer_mild_degradation():
    """Cancer: (0.25,0,0.75) assessment on mildly-degraded features (noise_prob=0.15)."""
    cfg = make_cancer_cfg(
        x_trust="0.25,0.0,0.75", y_trust="trust",
        x_dataset="noise_mild", y_dataset="clean",
        epsilon_low=0.1, epochs=2, port=5025,
    )
    result = run_scenario(cfg)
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass out of range: {result['trust_mass']}"


@pytest.mark.integration
def test_cancer_baseline_no_ptas():
    """Cancer: baseline NN without PTAS."""
    cfg = make_cancer_cfg(x_trust="trust", y_trust="trust", epochs=2, port=5023)

    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()

    def _no_ptas_client(cfg_inner, q):
        try:
            start_client(cfg_inner, not_ptas=True)
            dp = (
                f"results/NN_Train_{cfg_inner.dataset}_{cfg_inner.hidden_dim}"
                f"_{cfg_inner.x_trust}_{cfg_inner.y_trust}_PathSize_None"
            )
            q.put(_read_metrics(os.path.join(dp, "metrics.txt")))
        except Exception as exc:
            q.put({"train_acc": float("nan"), "error": str(exc)})

    p = multiprocessing.Process(target=_no_ptas_client, args=(cfg, client_q))
    p.start()
    p.join()
    result = client_q.get(timeout=10)
    assert result.get("train_acc", 0) > 0.5, f"Low train acc: {result}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cancer PTAS test runner.\n"
            "Default (no --xtrust/--ytrust/--epsilon): runs full 9×2 sweep and prints a table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["both", "server", "client"], default="both",
        help="both=PTAS+NN (default), server=PTAS only, client=NN only",
    )
    p.add_argument("--xtrust", default=None,
                   help="X trust for single run: trust | distrust | vacuous")
    p.add_argument("--ytrust", default=None,
                   help="Y trust for single run: trust | distrust | vacuous")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument(
        "--epsilon", "--epsilon-low", dest="epsilon", type=float, default=None,
        help="Epsilon for single run (default: sweep [0.1, 0.01])",
    )
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    p.add_argument("--no-round", type=int, default=None,
                   help="Stop PTAS after N batches (quick test)")
    p.add_argument("--no-ptas", action="store_true", help="Baseline NN without PTAS")
    return p.parse_args()


def _run_simple(cfg: TestCaseConfig, not_ptas: bool = False) -> None:
    """Two-process runner for single-scenario CLI use (no result capture)."""
    if not_ptas:
        start_client(cfg, not_ptas=True)
        return
    ptas_proc = multiprocessing.Process(target=start_ptas, args=(cfg,))
    ptas_proc.start()
    time.sleep(1)
    client_proc = multiprocessing.Process(target=start_client, args=(cfg, False))
    client_proc.start()
    client_proc.join()
    ptas_proc.join()


def main() -> None:
    args = parse_args()

    single_run = (
        args.xtrust is not None
        or args.ytrust is not None
        or args.epsilon is not None
        or args.no_ptas
        or args.mode != "both"
    )

    # ── Full sweep ────────────────────────────────────────────────────────────
    if not single_run:
        print("\nNo single-scenario flags → running full 9×2 scenario sweep.\n")
        results = run_all_scenarios(epochs=args.epochs, port=args.port)
        print_results_table(results)
        return

    # ── Single scenario run ───────────────────────────────────────────────────
    xtrust  = args.xtrust  or "trust"
    ytrust  = args.ytrust  or "trust"
    epsilon = args.epsilon if args.epsilon is not None else 0.1

    cfg = make_cancer_cfg(
        x_trust=xtrust, y_trust=ytrust, epsilon_low=epsilon,
        epochs=args.epochs, port=args.port, no_round=args.no_round,
    )

    print(f"\n{'='*64}")
    print(f"  CANCER TEST  |  mode={args.mode}  |  x={xtrust}  y={ytrust}")
    print(f"  epochs={args.epochs}  epsilon={epsilon}  port={args.port}")
    print(f"{'='*64}\n")

    if args.mode == "server":
        start_ptas(cfg)
    elif args.mode == "client":
        start_client(cfg, not_ptas=args.no_ptas)
    else:
        result = run_scenario(cfg)
        print(f"\n{'='*64}")
        print("  Single-scenario results")
        print(f"{'='*64}")
        print(f"  x_trust    : {result['x_trust']}")
        print(f"  y_trust    : {result['y_trust']}")
        print(f"  epsilon    : {result['epsilon']}")
        print(f"  Trust Mass : {result['trust_mass']:.4f}  "
              f"(aggregated output trust on fully-trusted input)")
        print(f"  Train Acc  : {result['train_acc']*100:.2f}%")
        print(f"  Test Acc   : {result['test_acc']*100:.2f}%")
        print(f"{'='*64}\n")

    print("\n=== Cancer test complete ===\n")


if __name__ == "__main__":
    main()
