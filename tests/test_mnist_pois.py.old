"""
MNIST *poisoned* dataset PATAS integration test.

The poisoning attack flips labels 6 ↔ 9 in one third of the training data
and adds a white trigger patch to the top-left corner.

Run standalone:
    python tests/test_mnist_pois.py

With custom patch size:
    python tests/test_mnist_pois.py --patch-size 8

Naive trust (both X and Y fully trusted, no poison awareness):
    python tests/test_mnist_pois.py --no-soph

Sophisticated poison-aware trust generator (default):
    python tests/test_mnist_pois.py --soph --patch-size 4

Baseline NN without PTAS:
    python tests/test_mnist_pois.py --no-ptas

Run with pytest:
    pytest tests/test_mnist_pois.py -m integration -s
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

def make_mnist_pois_cfg(
    x_trust: str = "trust",
    y_trust: str = "trust",
    epsilon_low: float = 0.05,
    epochs: int = 5,
    hidden_dim: int = 128,
    patch_size: int = 4,
    poisoned_soph: bool = True,
    port: int = 5040,
    no_round: int | None = None,
) -> TestCaseConfig:
    """
    Build a TestCaseConfig for the poisoned MNIST scenario.

    Parameters
    ----------
    poisoned_soph:
        True  → use the poison-aware trust generator (Tgenpoisoned_soph).
                 x_trust / y_trust are ignored in this mode.
        False → use plain trust generators specified by x_trust / y_trust
                 while still training on poisoned data.
    """
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
        mnist_patch_size=patch_size,
        mnist_poisoned_soph=poisoned_soph,
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
def test_mnist_pois_soph_patch4():
    """MNIST poisoned: poison-aware trust generator, patch_size=4, 2 epochs."""
    cfg = make_mnist_pois_cfg(
        poisoned_soph=True, patch_size=4, epochs=2, port=5040
    )
    run_both(cfg)


@pytest.mark.integration
def test_mnist_pois_naive_trust():
    """MNIST poisoned: naive fully-trusted assessor (no poison awareness), 2 epochs."""
    cfg = make_mnist_pois_cfg(
        x_trust="trust", y_trust="trust",
        poisoned_soph=False, patch_size=4, epochs=2, port=5041
    )
    run_both(cfg)


@pytest.mark.integration
def test_mnist_pois_vacuous_trust():
    """MNIST poisoned: vacuous X assessor, trusted Y, 2 epochs."""
    cfg = make_mnist_pois_cfg(
        x_trust="vacuous", y_trust="trust",
        poisoned_soph=False, patch_size=4, epochs=2, port=5042
    )
    run_both(cfg)


@pytest.mark.integration
def test_mnist_pois_baseline_no_ptas():
    """MNIST poisoned: baseline NN without PTAS, 2 epochs."""
    cfg = make_mnist_pois_cfg(
        poisoned_soph=False, patch_size=4, epochs=2, port=5043
    )
    run_both(cfg, not_ptas=True)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST poisoned PTAS test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["both", "server", "client"], default="both")
    p.add_argument("--xtrust", default="trust",
                   help="X trust spec (ignored when --soph is active)")
    p.add_argument("--ytrust", default="trust",
                   help="Y trust spec (ignored when --soph is active)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden-neurons", type=int, default=128)
    p.add_argument("--patch-size", type=int, default=4,
                   help="Trigger patch size (top-left square, pixels per side)")
    p.add_argument("--soph", dest="soph", action="store_true", default=True,
                   help="Use poison-aware trust generator (default: ON)")
    p.add_argument("--no-soph", dest="soph", action="store_false",
                   help="Disable poison-aware trust generator (naive trust instead)")
    p.add_argument("--epsilon-low", type=float, default=0.05)
    p.add_argument("--port", type=int, default=5040)
    p.add_argument("--no-round", type=int, default=None,
                   help="Stop PTAS after N batches (quick test)")
    p.add_argument("--no-ptas", action="store_true", help="Baseline NN without PTAS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_mnist_pois_cfg(
        x_trust=args.xtrust,
        y_trust=args.ytrust,
        epsilon_low=args.epsilon_low,
        epochs=args.epochs,
        hidden_dim=args.hidden_neurons,
        patch_size=args.patch_size,
        poisoned_soph=args.soph,
        port=args.port,
        no_round=args.no_round,
    )

    soph_label = "poison-aware (soph)" if args.soph else "naive"
    print(f"\n{'='*60}")
    print(f"  MNIST POISONED TEST  |  mode={args.mode}  |  trust_gen={soph_label}")
    print(f"  patch_size={args.patch_size}  x={args.xtrust}  y={args.ytrust}")
    print(f"  epochs={args.epochs}  hidden={args.hidden_neurons}  port={args.port}")
    print(f"{'='*60}\n")

    if args.mode == "server":
        start_ptas(cfg)
    elif args.mode == "client":
        start_client(cfg, not_ptas=args.no_ptas)
    else:
        run_both(cfg, not_ptas=args.no_ptas)

    print("\n=== MNIST poisoned test complete ===\n")


if __name__ == "__main__":
    main()
