"""
Cancer dataset PTAS integration test.

Run standalone (no pytest needed):
    python tests/test_cancer.py

Run with pytest (skipped by default; use -m integration):
    pytest tests/test_cancer.py -m integration -s

Argparse examples:
    python tests/test_cancer.py --xtrust trust  --ytrust trust
    python tests/test_cancer.py --xtrust distrust --ytrust trust
    python tests/test_cancer.py --xtrust vacuous --ytrust vacuous
    python tests/test_cancer.py --no-ptas           # baseline NN, no PTAS
    python tests/test_cancer.py --mode server       # only start PTAS side
    python tests/test_cancer.py --mode client       # only start NN side
"""

import sys
import os
import time
import argparse
import multiprocessing

# ── Path bootstrap (works with or without pip install) ───────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_cancer_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = 0.1,
    epochs: int = 3,
    port: int = 5020,
    no_round: int | None = None,
) -> TestCaseConfig:
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
    )


def run_both(cfg: TestCaseConfig, not_ptas: bool = False) -> None:
    """Launch PTAS server + NN client in separate processes."""
    if not_ptas:
        start_client(cfg, not_ptas=True)
        return

    ptas_proc = multiprocessing.Process(target=start_ptas, args=(cfg,))
    ptas_proc.start()
    time.sleep(1)  # let the socket bind

    client_proc = multiprocessing.Process(target=start_client, args=(cfg, False))
    client_proc.start()

    client_proc.join()
    ptas_proc.join()


# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_cancer_trust_trust():
    """Cancer: fully trusted X and Y."""
    cfg = make_cancer_cfg(x_trust="trust", y_trust="trust", epochs=2, port=5020)
    run_both(cfg)


@pytest.mark.integration
def test_cancer_distrust_trust():
    """Cancer: distrusted X, trusted Y."""
    cfg = make_cancer_cfg(x_trust="distrust", y_trust="trust", epochs=2, port=5021)
    run_both(cfg)


@pytest.mark.integration
def test_cancer_vacuous_vacuous():
    """Cancer: vacuous (uncertain) X and Y."""
    cfg = make_cancer_cfg(x_trust="vacuous", y_trust="vacuous", epochs=2, port=5022)
    run_both(cfg)


@pytest.mark.integration
def test_cancer_baseline_no_ptas():
    """Cancer: baseline NN without PTAS."""
    cfg = make_cancer_cfg(epochs=2, port=5023)
    run_both(cfg, not_ptas=True)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cancer PTAS test runner")
    p.add_argument("--mode", choices=["both", "server", "client"], default="both")
    p.add_argument("--xtrust", default="trust",
                   help="X trust: trust | distrust | vacuous | random | t,d,u")
    p.add_argument("--ytrust", default="trust",
                   help="Y trust: trust | distrust | vacuous | random | t,d,u")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--epsilon-low", type=float, default=0.1)
    p.add_argument("--port", type=int, default=5020)
    p.add_argument("--no-round", type=int, default=None,
                   help="Stop PTAS after N batches (quick test; client exits early)")
    p.add_argument("--no-ptas", action="store_true", help="Baseline NN without PTAS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_cancer_cfg(
        x_trust=args.xtrust,
        y_trust=args.ytrust,
        epsilon_low=args.epsilon_low,
        epochs=args.epochs,
        port=args.port,
        no_round=args.no_round,
    )

    print(f"\n{'='*60}")
    print(f"  CANCER TEST  |  mode={args.mode}  |  x={args.xtrust}  y={args.ytrust}")
    print(f"  epochs={args.epochs}  epsilon_low={args.epsilon_low}  port={args.port}")
    print(f"{'='*60}\n")

    if args.mode == "server":
        start_ptas(cfg)
    elif args.mode == "client":
        start_client(cfg, not_ptas=args.no_ptas)
    else:
        run_both(cfg, not_ptas=args.no_ptas)

    print("\n=== Cancer test complete ===\n")


if __name__ == "__main__":
    main()
