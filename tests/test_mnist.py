"""
MNIST dataset PTAS integration test.

Run standalone (no pytest needed):
    python tests/test_mnist.py

Run with pytest:
    pytest tests/test_mnist.py -m integration -s

Argparse examples:
    python tests/test_mnist.py --xtrust trust  --ytrust trust
    python tests/test_mnist.py --xtrust vacuous --ytrust trust
    python tests/test_mnist.py --xtrust distrust --ytrust distrust
    python tests/test_mnist.py --no-ptas              # baseline NN
    python tests/test_mnist.py --mode server          # PTAS only
    python tests/test_mnist.py --mode client          # NN only
"""

import sys
import os
import time
import argparse
import multiprocessing

# ── Path bootstrap ───────────────────────────────────────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_patas_dir = os.path.join(_v2_dir, "patas_module")
for _p in (_v2_dir, _patas_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import patas_module  # triggers patas_module/__init__.py path bootstrap
from main import (
    TestCaseConfig,
    get_lr_mnist,
    start_ptas,
    start_client,
)

try:
    import pytest
except ImportError:
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

def make_mnist_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = 0.05,
    epochs: int = 5,
    hidden_dim: int = 128,
    port: int = 5030,
    no_round: int | None = None,
) -> TestCaseConfig:
    return TestCaseConfig(
        dataset="mnist",
        input_dim=28 * 28,
        output_dim=10,
        hidden_dim=hidden_dim,
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
    )


def run_both(cfg: TestCaseConfig, not_ptas: bool = False) -> None:
    """Launch PTAS server + NN client in separate processes."""
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


# ─────────────────────────────────────────────────────────────────────────────
# pytest integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_mnist_trust_trust():
    """MNIST: fully trusted X and Y, 2 epochs."""
    cfg = make_mnist_cfg(x_trust="trust", y_trust="trust", epochs=2, port=5030)
    run_both(cfg)


@pytest.mark.integration
def test_mnist_vacuous_trust():
    """MNIST: vacuous (uncertain) X, trusted Y, 2 epochs."""
    cfg = make_mnist_cfg(x_trust="vacuous", y_trust="trust", epochs=2, port=5031)
    run_both(cfg)


@pytest.mark.integration
def test_mnist_distrust_distrust():
    """MNIST: distrusted X and Y, 2 epochs."""
    cfg = make_mnist_cfg(x_trust="distrust", y_trust="distrust", epochs=2, port=5032)
    run_both(cfg)


@pytest.mark.integration
def test_mnist_baseline_no_ptas():
    """MNIST: baseline NN without PTAS, 2 epochs."""
    cfg = make_mnist_cfg(epochs=2, port=5033)
    run_both(cfg, not_ptas=True)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST PTAS test runner")
    p.add_argument("--mode", choices=["both", "server", "client"], default="both")
    p.add_argument("--xtrust", default="trust",
                   help="X trust: trust | distrust | vacuous | random | t,d,u")
    p.add_argument("--ytrust", default="trust",
                   help="Y trust: trust | distrust | vacuous | random | t,d,u")
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of training epochs (default: 5)")
    p.add_argument("--hidden-neurons", type=int, default=128,
                   help="Hidden layer size (default: 128)")
    p.add_argument("--epsilon-low", type=float, default=0.05)
    p.add_argument("--port", type=int, default=5030)
    p.add_argument("--no-round", type=int, default=None,
                   help="Stop PTAS after N batches (quick test; client exits early)")
    p.add_argument("--no-ptas", action="store_true", help="Baseline NN without PTAS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_mnist_cfg(
        x_trust=args.xtrust,
        y_trust=args.ytrust,
        epsilon_low=args.epsilon_low,
        epochs=args.epochs,
        hidden_dim=args.hidden_neurons,
        port=args.port,
        no_round=args.no_round,
    )

    print(f"\n{'='*60}")
    print(f"  MNIST TEST  |  mode={args.mode}  |  x={args.xtrust}  y={args.ytrust}")
    print(f"  epochs={args.epochs}  hidden={args.hidden_neurons}  "
          f"epsilon_low={args.epsilon_low}  port={args.port}")
    print(f"{'='*60}\n")

    if args.mode == "server":
        start_ptas(cfg)
    elif args.mode == "client":
        start_client(cfg, not_ptas=args.no_ptas)
    else:
        run_both(cfg, not_ptas=args.no_ptas)

    print("\n=== MNIST test complete ===\n")


if __name__ == "__main__":
    main()
