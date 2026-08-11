"""
GTSRB (German Traffic Sign) PTAS integration test.

Dataset: 43 classes, grayscale 32×32 (input_dim = 1024), downloaded once via
torchvision and cached to data/gtsrb_gray32.npz.

Run standalone:
    python tests/test_gtsrb.py                       # full sweep → table
    python tests/test_gtsrb.py --epochs 3            # quick smoke-test
    python tests/test_gtsrb.py --xtrust trust --ytrust trust --hidden-neurons 128
    python tests/test_gtsrb.py --hidden-neurons 128 --hidden-neurons-2 128

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_gtsrb.py -m integration -s
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
_tests_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_v2_dir, _patas_dir, _tests_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import patas_module  # noqa: F401 — triggers path bootstrap
from main import TestCaseConfig, get_lr_gtsrb, start_ptas, start_client
from test_mnist import run_scenario, print_results_table  # reusable runners

try:
    import pytest
except ImportError:
    class _FakeMark:  # type: ignore[no-redef]
        @staticmethod
        def integration(fn): return fn
    class _FakePytest:  # type: ignore[no-redef]
        mark = _FakeMark()
    pytest = _FakePytest()  # type: ignore[assignment]

# ── Scenario definitions ──────────────────────────────────────────────────────

SCENARIOS: list[dict[str, Any]] = [
    {"hidden_dims": (128,),      "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (256,),      "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (128, 128),  "x_trust": "vacuous", "y_trust": "vacuous"},
    {"hidden_dims": (128,),      "x_trust": "trust",   "y_trust": "trust"},
]

_BASE_PORT      = 5091
_DEFAULT_EPS    = 0.05
_DEFAULT_EPOCHS = 20
_INPUT_DIM      = 32 * 32
_OUTPUT_DIM     = 43

# ─────────────────────────────────────────────────────────────────────────────
# Config factory
# ─────────────────────────────────────────────────────────────────────────────

def make_gtsrb_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = _DEFAULT_EPS,
    epochs: int = _DEFAULT_EPOCHS,
    hidden_dims: tuple[int, ...] = (128,),
    port: int = _BASE_PORT,
    no_round: int | None = None,
    noise_level: float | None = None,
) -> TestCaseConfig:
    """Build a TestCaseConfig for the GTSRB dataset (43 classes, 32×32 gray)."""
    return TestCaseConfig(
        dataset="gtsrb",
        input_dim=_INPUT_DIM,
        output_dim=_OUTPUT_DIM,
        hidden_dim=hidden_dims[0],   # legacy scalar, kept in sync
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=128,
        learning_rate=get_lr_gtsrb,
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
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_all_scenarios(epochs: int = _DEFAULT_EPOCHS,
                      epsilon_low: float = _DEFAULT_EPS,
                      base_port: int = _BASE_PORT) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, sc in enumerate(SCENARIOS):
        cfg = make_gtsrb_cfg(
            x_trust=sc["x_trust"], y_trust=sc["y_trust"],
            hidden_dims=sc["hidden_dims"],
            epsilon_low=epsilon_low, epochs=epochs, port=base_port + i,
        )
        arch = "-".join(str(h) for h in sc["hidden_dims"])
        print(f"\n  ► arch={arch:<10}  x={sc['x_trust']:<8}  y={sc['y_trust']}")
        result = run_scenario(cfg)
        print(f"    trust_mass={result['trust_mass']:.4f}  "
              f"train={result['train_acc']*100:.2f}%  "
              f"test={result['test_acc']*100:.2f}%")
        results.append(result)
        time.sleep(2)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_gtsrb_trust_128():
    """GTSRB: trust/trust, single hidden layer 128 neurons."""
    cfg = make_gtsrb_cfg(x_trust="trust", y_trust="trust",
                         hidden_dims=(128,), epochs=2, port=5095)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.3, f"Low train acc: {result['train_acc']}"
    assert 0.0 <= result["trust_mass"] <= 1.0, f"Trust mass OOB: {result['trust_mass']}"


@pytest.mark.integration
def test_gtsrb_vacuous_128_128():
    """GTSRB: vacuous/vacuous, two hidden layers [128, 128]."""
    cfg = make_gtsrb_cfg(x_trust="vacuous", y_trust="vacuous",
                         hidden_dims=(128, 128), epochs=2, port=5096)
    result = run_scenario(cfg)
    assert result["train_acc"] > 0.2, f"Low train acc: {result['train_acc']}"

# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GTSRB PTAS architecture sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["both", "server", "client"], default="both")
    p.add_argument("--xtrust", default=None,
                   help="X trust for single run: trust | distrust | vacuous | t,d,u")
    p.add_argument("--ytrust", default=None,
                   help="Y trust for single run: trust | distrust | vacuous | t,d,u")
    p.add_argument("--hidden-neurons", type=int, default=None,
                   help="First hidden layer size (single-run mode; default 128)")
    p.add_argument("--hidden-neurons-2", type=int, default=None,
                   help="Second hidden layer size — activates two-hidden-layer mode")
    p.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--epsilon-low", type=float, default=_DEFAULT_EPS)
    p.add_argument("--port", type=int, default=_BASE_PORT)
    p.add_argument("--no-round", type=int, default=None)
    p.add_argument("--no-ptas", action="store_true",
                   help="Baseline NN without PTAS (client only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    single_run = (
        args.xtrust is not None
        or args.ytrust is not None
        or args.hidden_neurons is not None
        or args.no_ptas
        or args.mode != "both"
    )

    if not single_run:
        print("\nNo single-scenario flags → running full GTSRB architecture sweep.\n")
        results = run_all_scenarios(
            epochs=args.epochs, epsilon_low=args.epsilon_low, base_port=args.port,
        )
        print_results_table(results)
        return

    xtrust = args.xtrust or "vacuous"
    ytrust = args.ytrust or "vacuous"
    h1 = args.hidden_neurons if args.hidden_neurons is not None else 128
    h2 = args.hidden_neurons_2
    hidden_dims: tuple[int, ...] = (h1, h2) if h2 is not None else (h1,)
    arch_label = "-".join(str(h) for h in hidden_dims)

    cfg = make_gtsrb_cfg(
        x_trust=xtrust, y_trust=ytrust, hidden_dims=hidden_dims,
        epsilon_low=args.epsilon_low, epochs=args.epochs,
        port=args.port, no_round=args.no_round,
    )

    print(f"\n{'='*64}")
    print(f"  GTSRB TEST  |  mode={args.mode}  |  x={xtrust}  y={ytrust}")
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
        print(f"  Trust Mass   : {result['trust_mass']:.4f}")
        print(f"  Train Acc    : {result['train_acc']*100:.2f}%")
        print(f"  Test Acc     : {result['test_acc']*100:.2f}%")
        print(f"{'='*64}\n")

    print("\n=== GTSRB test complete ===\n")


if __name__ == "__main__":
    # Force 'spawn' — 'fork' (Linux default) breaks once CUDA is initialized
    # in this process, which happens as soon as a torch model is built.
    multiprocessing.set_start_method("spawn", force=True)
    main()
