"""
MNIST dataset PTAS integration test — architecture sweep.

Run standalone (no pytest needed):
    python tests/test_mnist.py

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_mnist.py -m integration -s

Standalone examples:
    python tests/test_mnist.py                        # all 6 scenarios → table
    python tests/test_mnist.py --epochs 3             # quick smoke-test
    python tests/test_mnist.py --xtrust trust --ytrust trust --hidden-neurons 16
    python tests/test_mnist.py --hidden-neurons 32 --hidden-neurons-2 32  # two layers
    python tests/test_mnist.py --no-ptas              # baseline NN, no PTAS
    python tests/test_mnist.py --mode server          # PTAS side only
    python tests/test_mnist.py --mode client          # NN side only

Default sweep — 6 scenarios:
    vacuous/vacuous  single hidden layer  :  16, 32, 64, 128 neurons
    vacuous/vacuous  two hidden layers    :  16-16
    trust/trust      single hidden layer  :  16 neurons

Default sweep output (6 rows):
  Architecture  x_trust     y_trust     Trust Mass  Train Acc   Test Acc
  16            vacuous     vacuous        0.XXXX    97.XX%      96.XX%
  32            vacuous     vacuous        0.XXXX    97.XX%      96.XX%
  ...
  16-16         vacuous     vacuous        0.XXXX    XX.XX%      XX.XX%
  16            trust       trust          0.XXXX    XX.XX%      XX.XX%

Trust Mass definition
---------------------
Trust Mass = trust component of PTAS.aggregation(apply_feedforward(fully-trusted-input))
This is the scalar [t, d, u][0] that ptas_evaluation prints as
"Apply Feed Forward on fully Trusted Input → Aggregated Value".
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing
from typing import Any

# ── Path bootstrap (works with or without pip install) ────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import patas_module  # triggers patas_module/__init__.py path bootstrap
from main import (  # main.py lives inside patas_module/
    TestCaseConfig,
    get_lr_mnist,
    start_ptas,
    start_client,
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

# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS: list[dict[str, Any]] = [
    # vacuous/vacuous — single hidden layers (width sweep)
    {"hidden_dims": (16,),    "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (32,),    "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (64,),    "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (128,),   "x_trust": "vacuous", "y_trust": "vacuous"},
    # vacuous/vacuous — two hidden layers
    {"hidden_dims": (16, 16), "x_trust": "vacuous", "y_trust": "vacuous"},
    # trust/trust — 16 neurons
    {"hidden_dims": (16,),    "x_trust": "trust",   "y_trust": "trust"},
]

_BASE_PORT     = 5041   # ports 5041–5046 (one per scenario; 5040 is reserved/occupied)
_DEFAULT_EPS   = 0.05
_DEFAULT_EPOCHS = 20

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_mnist_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = _DEFAULT_EPS,
    epochs: int = _DEFAULT_EPOCHS,
    hidden_dims: tuple[int, ...] = (128,),
    port: int = _BASE_PORT,
    no_round: int | None = None,
    noise_level: float | None = None,
) -> TestCaseConfig:
    """
    Build a TestCaseConfig for the MNIST dataset.

    ``hidden_dims`` specifies the hidden-layer architecture, e.g.:
        (16,)     → one hidden layer with 16 neurons
        (16, 16)  → two hidden layers with 16 neurons each

    ``hidden_dim`` (the legacy scalar field) is automatically set to
    ``hidden_dims[0]`` so that old PTAS code paths stay compatible.

    ``noise_level`` (float in [0, 1] or None) overrides the default noise
    probability used by load_X / load_y in the "noise" (vacuous) data variant.
    When None the default (0.30) applies.
    """
    return TestCaseConfig(
        dataset="mnist",
        input_dim=28 * 28,
        output_dim=10,
        hidden_dim=hidden_dims[0],   # kept for backwards-compat
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=128,
        learning_rate=get_lr_mnist,
        epsilon_low=epsilon_low,
        x_trust=x_trust,
        y_trust=y_trust,
        port=port,
        mnist_patch_size=None,
        mnist_poisoned_soph=False,
        no_round=no_round,
        noise_level=noise_level,
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
    trained PTAS object — then computes the final trust mass.
    ``ready_event`` is set by run_chunk() as soon as the socket is bound and
    listening, so the client process can start connecting immediately.
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
    The datapath mirrors what start_client builds (uses arch_str, not hidden_dim).
    """
    try:
        start_client(cfg, not_ptas=False)
        hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
        arch_str = "_".join(str(h) for h in hidden_list)
        datapath = (
            f"results/NN_Train_{cfg.dataset}_{arch_str}"
            f"_{cfg.x_trust}_{cfg.y_trust}_PathSize_None"
        )
        metrics = _read_metrics(os.path.join(datapath, "metrics.txt"))
        result_queue.put(metrics)
    except Exception as exc:
        result_queue.put({"train_acc": float("nan"), "test_acc": float("nan"), "error": str(exc)})


def _client_only_worker(cfg: TestCaseConfig, result_queue: "multiprocessing.Queue[dict]") -> None:
    """
    Standalone NN subprocess — no PTAS server involved.
    Calls start_client(cfg, not_ptas=True) so no socket connection is attempted,
    then reads the metrics.txt produced by training.
    """
    try:
        start_client(cfg, not_ptas=True)
        hidden_list = list(cfg.hidden_dims) if cfg.hidden_dims else [cfg.hidden_dim]
        arch_str = "_".join(str(h) for h in hidden_list)
        datapath = (
            f"results/NN_Train_{cfg.dataset}_{arch_str}"
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
    finish, and return a result dict with hidden_dims, trust_mass, train_acc,
    test_acc.

    Uses a multiprocessing.Event to synchronise: the client only starts after
    PTAS has bound and is listening (run_chunk sets the event), so there is no
    fixed sleep and no cold-start race on Windows.
    """
    ptas_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    ptas_proc = multiprocessing.Process(target=_ptas_worker, args=(cfg, ptas_q, ready_event))
    ptas_proc.start()

    # Wait until PTAS has bound its socket (set by run_chunk after s.listen)
    ready_event.wait(timeout=60)

    client_proc = multiprocessing.Process(target=_client_worker, args=(cfg, client_q))
    client_proc.start()

    # Read queues BEFORE joining to avoid Windows pipe-buffer deadlock
    # (a subprocess putting large data blocks until the parent drains the pipe)
    # Timeout: MNIST 20-epoch runs take ~2 hours; use 7200 s as safe upper bound.
    _QUEUE_TIMEOUT = 7200
    try:
        ptas_result = ptas_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception:
        ptas_result = {}
    try:
        client_result = client_q.get(timeout=_QUEUE_TIMEOUT)
    except Exception:
        client_result = {}

    client_proc.join(timeout=60)
    ptas_proc.join(timeout=60)

    return {
        "hidden_dims": cfg.hidden_dims,
        "x_trust":    cfg.x_trust,
        "y_trust":    cfg.y_trust,
        "trust_mass": ptas_result.get("trust_mass", float("nan")),
        "train_acc":  client_result.get("train_acc", float("nan")),
        "test_acc":   client_result.get("test_acc", float("nan")),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios(
    epochs: int = _DEFAULT_EPOCHS,
    epsilon_low: float = _DEFAULT_EPS,
    base_port: int = _BASE_PORT,
) -> list[dict[str, Any]]:
    """
    Run all 6 MNIST scenarios sequentially and return a flat list of result dicts.

    Scenarios:
        vacuous/vacuous — single hidden layers: 16, 32, 64, 128
        vacuous/vacuous — two hidden layers:    16-16
        trust/trust     — single hidden layer:  16
    """
    results: list[dict[str, Any]] = []

    for i, sc in enumerate(SCENARIOS):
        cfg = make_mnist_cfg(
            x_trust     = sc["x_trust"],
            y_trust     = sc["y_trust"],
            hidden_dims = sc["hidden_dims"],
            epsilon_low = epsilon_low,
            epochs      = epochs,
            port        = base_port + i,
        )
        arch_label = "-".join(str(h) for h in sc["hidden_dims"])
        lbl = f"arch={arch_label:<8}  x={sc['x_trust']:<8}  y={sc['y_trust']}"
        print(f"\n  ► {lbl}")

        result = run_scenario(cfg)
        tm = result["trust_mass"]
        tr = result["train_acc"]
        te = result["test_acc"]
        print(f"    trust_mass={tm:.4f}  train={tr*100:.2f}%  test={te*100:.2f}%")
        results.append(result)
        time.sleep(2)

    return results

def run_scenario_no_ptas(cfg: TestCaseConfig) -> dict[str, Any]:
    """
    Run only the standalone NN (no PTAS server) in a subprocess and return a
    result dict with hidden_dims, trust_mass (NaN), train_acc, test_acc.
    Used by the noise sweep where PTAS is not needed.
    """
    client_q: "multiprocessing.Queue[dict]" = multiprocessing.Queue()

    client_proc = multiprocessing.Process(
        target=_client_only_worker, args=(cfg, client_q)
    )
    client_proc.start()

    try:
        client_result = client_q.get(timeout=300)
    except Exception:
        client_result = {}

    client_proc.join(timeout=60)

    return {
        "hidden_dims": cfg.hidden_dims,
        "x_trust":    cfg.x_trust,
        "y_trust":    cfg.y_trust,
        "trust_mass": float("nan"),   # no PTAS in this mode
        "train_acc":  client_result.get("train_acc", float("nan")),
        "test_acc":   client_result.get("test_acc", float("nan")),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Noise sweep (vacuous, feature + label noise 0.1 → 1.0)
# ─────────────────────────────────────────────────────────────────────────────

#: Noise levels probed by run_noise_sweep_vacuous (feature noise_prob = label flip_rate).
NOISE_LEVELS: list[float] = [round(x * 0.1, 1) for x in range(1, 11)]


def run_noise_sweep_vacuous(
    epochs: int = _DEFAULT_EPOCHS,
    hidden_dims: tuple[int, ...] = (128,),
    epsilon_low: float = _DEFAULT_EPS,
    noise_levels: list[float] | None = None,
    base_port: int = _BASE_PORT,
) -> list[dict[str, Any]]:
    """
    Run the vacuous/vacuous MNIST scenario for each noise level in *noise_levels*
    (default: 0.1, 0.2, …, 1.0) and return a list of result dicts.

    Each dict has keys:
        noise_level, hidden_dims, x_trust, y_trust,
        trust_mass, train_acc, test_acc

    Architecture is fixed to ``hidden_dims`` (default: single layer, 128 neurons).
    Both feature noise (``noise_prob`` in ``load_X``) and label noise
    (``noise_prob`` in ``load_y``) are set to the same ``noise_level`` value.

    Scenarios are run sequentially; each reuses the same port (safe because
    processes fully complete before the next starts).
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    arch_label = "-".join(str(h) for h in hidden_dims)
    print(f"\n  Noise sweep  |  arch={arch_label}  epochs={epochs}  ε={epsilon_low}")
    print(f"  Noise levels: {noise_levels}\n")

    results: list[dict[str, Any]] = []
    for i, nl in enumerate(noise_levels):
        cfg = make_mnist_cfg(
            x_trust="vacuous",
            y_trust="vacuous",
            hidden_dims=hidden_dims,
            epochs=epochs,
            epsilon_low=epsilon_low,
            port=base_port + i,
            noise_level=nl,
        )
        print(f"  ► noise={nl:.1f}  arch={arch_label}  x=vacuous  y=vacuous")
        result = run_scenario_no_ptas(cfg)
        result["noise_level"] = nl
        tm = result["trust_mass"]
        tr = result["train_acc"]
        te = result["test_acc"]
        print(f"    trust_mass={tm:.4f}  train={tr*100:.2f}%  test={te*100:.2f}%")
        results.append(result)
        time.sleep(2)

    return results


def plot_noise_sweep(
    results: list[dict[str, Any]],
    output_path: str = "results/noise_sweep_vacuous.pdf",
) -> None:
    """
    Plot test (and train) accuracy vs noise level for the vacuous noise sweep
    and save to *output_path*.

    X-axis : noise level (feature noise_prob = label flip_rate)
    Y-axis : accuracy [%]
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (safe for subprocesses)
    import matplotlib.pyplot as plt

    noise_levels = [r["noise_level"] for r in results]
    test_accs    = [r["test_acc"]  * 100 for r in results]
    train_accs   = [r["train_acc"] * 100 for r in results]

    # ── infer metadata from first result ──────────────────────────────────────
    hidden_dims = results[0].get("hidden_dims", (128,)) if results else (128,)
    arch_label  = "-".join(str(h) for h in hidden_dims) if hidden_dims else "?"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(noise_levels, test_accs,  "b-o",  label="Test Accuracy",
            linewidth=2, markersize=7)
    ax.plot(noise_levels, train_accs, "r--s", label="Train Accuracy",
            linewidth=2, markersize=7)

    ax.set_xlabel("Noise Level  (feature noise prob. = label flip rate)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"MNIST Accuracy vs. Noise Level\n"
        f"(Vacuous trust · {arch_label} neurons · 20 epochs)",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([0.05, 1.05])
    ax.set_ylim([0, 105])
    ax.set_xticks(noise_levels)
    ax.set_xticklabels([f"{v:.1f}" for v in noise_levels])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Noise-sweep plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict[str, Any]]) -> None:
    """
    Print a compact summary table:

      Architecture  x_trust     y_trust     Trust Mass  Train Acc   Test Acc
      16            vacuous     vacuous        0.XXXX    97.XX%      96.XX%
      ...
    """
    def _pct(v: float) -> str:
        return f"{v*100:6.2f}%" if v == v else "   N/A "

    def _f4(v: float) -> str:
        return f"{v:.4f}" if v == v else "  N/A  "

    def _arch(dims: tuple[int, ...] | None) -> str:
        if dims is None:
            return "?"
        return "-".join(str(h) for h in dims)

    headers = ["Architecture", "x_trust",  "y_trust",  "Trust Mass", "Train Acc", "Test Acc"]
    col_w   = [14,              10,          10,          12,           10,          10]
    parts   = [f"{{:<{w}}}" for w in col_w[:3]] + [f"{{:>{w}}}" for w in col_w[3:]]
    row_fmt = "  ".join(parts)
    sep     = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print("=" * len(sep))
    print("  MNIST PTAS — Architecture Sweep Results")
    print("  Trust Mass = aggregated output trust on fully-trusted input")
    print("  (= 'Apply Feed Forward on trusted input → Aggregated Value[0]')")
    print("=" * len(sep))
    print()
    print(sep)
    print(row_fmt.format(*headers))
    print(sep)

    for r in results:
        row_vals = [
            _arch(r.get("hidden_dims")),
            r.get("x_trust", "?"),
            r.get("y_trust", "?"),
            _f4(r.get("trust_mass", float("nan"))),
            _pct(r.get("train_acc",  float("nan"))),
            _pct(r.get("test_acc",   float("nan"))),
        ]
        print(row_fmt.format(*row_vals))

    print(sep)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests (single-scenario, for CI)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_mnist_vacuous_16():
    """MNIST: vacuous/vacuous, single hidden layer 16 neurons."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(16,), epochs=2, port=5041)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"


@pytest.mark.integration
def test_mnist_vacuous_32():
    """MNIST: vacuous/vacuous, single hidden layer 32 neurons."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(32,), epochs=2, port=5042)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_mnist_vacuous_64():
    """MNIST: vacuous/vacuous, single hidden layer 64 neurons."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(64,), epochs=2, port=5043)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_mnist_vacuous_128():
    """MNIST: vacuous/vacuous, single hidden layer 128 neurons."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(128,), epochs=2, port=5044)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_mnist_vacuous_16_16():
    """MNIST: vacuous/vacuous, two hidden layers [16, 16]."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(16, 16), epochs=2, port=5045)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"


@pytest.mark.integration
def test_mnist_trust_trust_16():
    """MNIST: trust/trust, single hidden layer 16 neurons."""
    cfg = make_mnist_cfg(x_trust="trust", y_trust="trust",
                         hidden_dims=(16,), epochs=2, port=5046)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.5, f"Low train acc: {result['train_acc']}"
    assert result["trust_mass"] > 0.5, f"Low trust mass: {result['trust_mass']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MNIST PTAS architecture sweep.\n"
            "Default (no flags): runs all 6 scenarios and prints a summary table.\n\n"
            "Examples:\n"
            "  python tests/test_mnist.py                          # full sweep\n"
            "  python tests/test_mnist.py --xtrust trust --ytrust trust --hidden-neurons 16\n"
            "  python tests/test_mnist.py --hidden-neurons 32 --hidden-neurons-2 32\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["both", "server", "client"], default="both",
                   help="both=PTAS+NN (default), server=PTAS only, client=NN only")
    p.add_argument("--xtrust", default=None,
                   help="X trust for single run: trust | distrust | vacuous | t,d,u")
    p.add_argument("--ytrust", default=None,
                   help="Y trust for single run: trust | distrust | vacuous | t,d,u")
    p.add_argument("--hidden-neurons", type=int, default=None,
                   help="First hidden layer size (single-run mode; default 128)")
    p.add_argument("--hidden-neurons-2", type=int, default=None,
                   help="Second hidden layer size — activates two-hidden-layer mode")
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS,
                   help=f"Training epochs (default: {_DEFAULT_EPOCHS})")
    p.add_argument("--epsilon-low", type=float, default=_DEFAULT_EPS,
                   help=f"PTAS epsilon (default: {_DEFAULT_EPS})")
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--no-round", type=int, default=None,
                   help="Stop PTAS after N batches (quick test)")
    p.add_argument("--no-ptas", action="store_true",
                   help="Baseline NN without PTAS (client only)")
    p.add_argument("--noise-sweep", action="store_true",
                   help=(
                       "Sweep noise level 0.1–1.0 for vacuous/vacuous MNIST "
                       "(hidden=128, epochs=20 unless overridden). "
                       "Applies the same noise_prob to both features and labels "
                       "and saves a test-accuracy-vs-noise-level plot."
                   ))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Noise sweep ───────────────────────────────────────────────────────────
    if getattr(args, "noise_sweep", False):
        h1     = args.hidden_neurons if args.hidden_neurons is not None else 128
        h2     = getattr(args, "hidden_neurons_2", None)
        hidden_dims: tuple[int, ...] = (h1, h2) if h2 is not None else (h1,)
        arch_str = "-".join(str(h) for h in hidden_dims)

        print(f"\n{'='*64}")
        print(f"  MNIST NOISE SWEEP  |  vacuous/vacuous")
        print(f"  arch={arch_str}  epochs={args.epochs}  ε={args.epsilon_low}")
        print(f"  noise ∈ {NOISE_LEVELS}  (feature noise_prob = label flip_rate)")
        print(f"{'='*64}\n")

        sweep_results = run_noise_sweep_vacuous(
            epochs      = args.epochs,
            hidden_dims = hidden_dims,
            epsilon_low = args.epsilon_low,
            base_port   = args.port,
        )

        outpath = (
            f"results/noise_sweep_mnist_vacuous_{arch_str}"
            f"_epochs{args.epochs}.pdf"
        )
        plot_noise_sweep(sweep_results, output_path=outpath)

        # Also print a compact table
        print()
        print(f"{'Noise':>8}  {'Train Acc':>10}  {'Test Acc':>10}")
        print("-" * 33)
        for r in sweep_results:
            print(f"  {r['noise_level']:.1f}     "
                  f"{r['train_acc']*100:8.2f}%   "
                  f"{r['test_acc']*100:8.2f}%")
        print()
        print("=== Noise sweep complete ===\n")
        return

    single_run = (
        args.xtrust is not None
        or args.ytrust is not None
        or args.hidden_neurons is not None
        or args.no_ptas
        or args.mode != "both"
    )

    # ── Full sweep ────────────────────────────────────────────────────────────
    if not single_run:
        print("\nNo single-scenario flags → running full 6-scenario architecture sweep.\n")
        results = run_all_scenarios(
            epochs      = args.epochs,
            epsilon_low = args.epsilon_low,
            base_port   = args.port,
        )
        print_results_table(results)
        return

    # ── Single scenario run ───────────────────────────────────────────────────
    xtrust = args.xtrust or "vacuous"
    ytrust = args.ytrust or "vacuous"
    h1     = args.hidden_neurons if args.hidden_neurons is not None else 128
    h2     = args.hidden_neurons_2
    hidden_dims: tuple[int, ...] = (h1, h2) if h2 is not None else (h1,)
    arch_label = "-".join(str(h) for h in hidden_dims)

    cfg = make_mnist_cfg(
        x_trust     = xtrust,
        y_trust     = ytrust,
        hidden_dims = hidden_dims,
        epsilon_low = args.epsilon_low,
        epochs      = args.epochs,
        port        = args.port,
        no_round    = args.no_round,
    )

    print(f"\n{'='*64}")
    print(f"  MNIST TEST  |  mode={args.mode}  |  x={xtrust}  y={ytrust}")
    print(f"  arch={arch_label}  epochs={args.epochs}  ε={args.epsilon_low}  port={args.port}")
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
        print(f"  Architecture : {arch_label}")
        print(f"  x_trust      : {result['x_trust']}")
        print(f"  y_trust      : {result['y_trust']}")
        print(f"  Trust Mass   : {result['trust_mass']:.4f}  "
              f"(aggregated output trust on fully-trusted input)")
        print(f"  Train Acc    : {result['train_acc']*100:.2f}%")
        print(f"  Test Acc     : {result['test_acc']*100:.2f}%")
        print(f"{'='*64}\n")

    print("\n=== MNIST test complete ===\n")


if __name__ == "__main__":
    main()
