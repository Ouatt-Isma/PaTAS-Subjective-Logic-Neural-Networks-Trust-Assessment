"""Evaluation helpers for the §7.8 experiments."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from primary_nn import PrimaryNN
from patas import (TrustNodesNetwork, trust_feedforward,
                   aggregate_output_trust, gen_ipta, build_input_trust)
from subjective_logic import Opinion


def evaluate_canonical_profiles(tnn: TrustNodesNetwork, n_in: int,
                                decision: Optional[int] = None
                                ) -> Dict[str, Opinion]:
    """Aggregated output opinion under {trusted, vacuous, distrusted} input."""
    out = {}
    for profile in ("trusted", "vacuous", "distrusted"):
        T_x = build_input_trust(n_in, kind=profile)
        T_out = trust_feedforward(T_x, tnn)
        out[profile] = aggregate_output_trust(T_out, decision=decision)
    return out


def per_class_trust(tnn: TrustNodesNetwork, n_in: int) -> List[Opinion]:
    """Per-class output trust under a fully-trusted input."""
    T_x = build_input_trust(n_in, kind="trusted")
    return trust_feedforward(T_x, tnn)


def ipta_for_sample(tnn: TrustNodesNetwork, nn: PrimaryNN,
                    X_query: np.ndarray, *, sample_idx: int = 0,
                    feature_trust: Optional[List[Opinion]] = None,
                    decision: Optional[int] = None) -> Opinion:
    if X_query.ndim == 1:
        X_query = X_query[None, :]
    fwd = nn.forward(X_query)
    if feature_trust is None:
        feature_trust = build_input_trust(nn.n_in, kind="trusted")
    return gen_ipta(feature_trust, tnn, fwd, sample_idx=sample_idx,
                    decision=decision)
