"""
pytest configuration for the PATAS test suite.

Integration tests (full end-to-end training) are marked with @pytest.mark.integration
and are skipped by default.  Run them explicitly with:

    pytest tests/ -m integration
    # or to run everything:
    pytest tests/ -m "integration or not integration"
"""

import sys
import os

# ── Make patas_module importable without pip install ──────────────────────────
_v2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # v2/
if _v2_dir not in sys.path:
    sys.path.insert(0, _v2_dir)

# Trigger patas_module/__init__.py so its sys.path bootstrap runs.
import patas_module  # noqa: F401 E402

# ── Fixtures (only registered when pytest is installed) ───────────────────────
try:
    import pytest

    @pytest.fixture(scope="session")
    def patas_dir():
        """Path to the patas_module/ directory."""
        return os.path.join(_v2_dir, "patas_module")

    @pytest.fixture(scope="session")
    def cancer_cfg():
        """Minimal cancer TestCaseConfig (2 epochs, port 5010)."""
        sys.path.insert(0, os.path.join(_v2_dir, "patas_module"))
        from main import TestCaseConfig, get_lr_cancer  # noqa: E402
        return TestCaseConfig(
            dataset="cancer",
            input_dim=30,
            output_dim=2,
            hidden_dim=16,
            epochs=2,
            batch_size=64,
            learning_rate=get_lr_cancer,
            epsilon_low=0.1,
            x_trust="trust",
            y_trust="trust",
            port=5010,
            mnist_patch_size=None,
            mnist_poisoned_soph=False,
            no_round=None,
        )

except ImportError:
    pass  # pytest not installed; conftest still bootstraps the path correctly
