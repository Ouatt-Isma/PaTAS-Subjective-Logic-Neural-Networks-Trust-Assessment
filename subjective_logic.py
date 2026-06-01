"""
Backward-compatibility shim for scripts that pre-date the integration of
Subjective Logic into patas_module.

The **canonical** implementation lives at ``patas_module/subjective_logic.py``.
New code should import from there:

    from patas_module.subjective_logic import Opinion, fuse_many, ...
    # or
    from patas_module import Opinion, fuse_many, ...
"""

from patas_module.subjective_logic import (  # noqa: F401
    EPS,
    Opinion,
    vacuous, trusted, distrusted,
    bpq, ewq, cuq,
    discount,
    cumulative_fusion, averaging_fusion,
    weighted_belief_fusion, consensus_compromise_fusion,
    fuse_many,
    revise, multiply, conservative_div, deduce,
    opinions_to_array,
    _prod_except, _fuse_cbf_multi, _fuse_abf_multi,
    _EPS_VEC, _normalize_vec,
    averaging_fusion_vec, cumulative_fusion_vec,
    weighted_belief_fusion_vec, ccf_fusion_vec, constraint_fusion_vec,
    multiply_vec, discount_vec, bpq_vec, deduce_vec,
)
