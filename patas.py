"""
PaTAS-TP : Parallel Trust Assessment System -- Trust Propagation.

Mirrors a PrimaryNN at the level of trust opinions.  Implements:

    * Trust Nodes Network          (Def. 7.2)
    * Trust Function (per-neuron)  (Def. 7.4)
    * Trust Feedforward            (Alg. 5)
    * Parameter-Trust Update       (Alg. 6, Chapter 7)
    * Output-Trust Aggregation
    * Inference-Path Trust Assessment (IPTA / GenIPTA, §7.3)

Supports PrimaryNN architectures with one or more hidden layers.
The system runs **in parallel** with the NN -- it never modifies the
underlying weights or predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from patas_module.subjective_logic import (
    Opinion, vacuous, trusted, distrusted,
    discount, averaging_fusion, fuse_many,
    revise, multiply, conservative_div, deduce, bpq,
)
from primary_nn import PrimaryNN, ForwardCache, BackwardCache


# ====================================================================== #
#  Trust Nodes Network  (Def. 7.2)                                       #
# ====================================================================== #

@dataclass
class TrustNodesNetwork:
    """
    Mirrors PrimaryNN's parameters as binomial trust opinions.

    Attributes
    ----------
    T_Ws : list of object arrays, one per layer transition.
           T_Ws[k] has shape (sizes[k], sizes[k+1]), where
           sizes = [n_in, *hidden, n_out].
           T_Ws[k][j, i] is the trust opinion on the weight connecting
           pre-layer-k unit j to post-layer-k unit i.
    T_bs : list of object arrays, one per layer transition.
           T_bs[k] has shape (sizes[k+1],).
           T_bs[k][i] is the trust opinion on the bias of unit i
           in layer k+1.

    All opinions are initialised to vacuous, as prescribed in §7.3:
    "The parameters of the Trust Nodes ... are initialized as vacuous
    opinions and refined during training by the Parameter-Trust Update
    module."
    """
    T_Ws: List[np.ndarray]   # T_Ws[k] shape (sizes[k], sizes[k+1])
    T_bs: List[np.ndarray]   # T_bs[k] shape (sizes[k+1],)

    @classmethod
    def from_nn(cls, nn: PrimaryNN) -> "TrustNodesNetwork":
        """Build a TrustNodesNetwork mirroring a PrimaryNN (all vacuous)."""
        def vac_grid(shape: tuple) -> np.ndarray:
            arr = np.empty(shape, dtype=object)
            for idx in np.ndindex(*shape):
                arr[idx] = vacuous()
            return arr

        sizes = [nn.n_in] + list(nn.hidden) + [nn.n_out]
        T_Ws  = [vac_grid((sizes[k], sizes[k + 1])) for k in range(len(nn.Ws))]
        T_bs  = [vac_grid((sizes[k + 1],))          for k in range(len(nn.Ws))]
        return cls(T_Ws=T_Ws, T_bs=T_bs)


# ====================================================================== #
#  Trust Function -- single linear layer  (Def. 7.4)                    #
# ====================================================================== #

def trust_linear_layer(T_in: List[Opinion],
                       T_W: np.ndarray,
                       T_b: np.ndarray,
                       active_mask: Optional[np.ndarray] = None
                       ) -> List[Opinion]:
    """
    Trust counterpart of a linear layer:
        z_i = Σ_j W_ij * x_j + b_i
        T_z_i = ⊕_j ( T_W[j,i] ⊗ T_in[j] )        (Def. 7.4)

    Fusion uses *averaging* belief fusion (cf. §7.8.1 implementation note).

    Def. 7.4 of the dissertation defines the Trust Function over the
    weighted-input sum only; the bias trust is updated separately by the
    Parameter-Trust Update and is not fused into the per-neuron output opinion.

    Parameters
    ----------
    T_in       : trust opinions for the layer's inputs (length = n_prev).
    T_W        : object array of shape (n_prev, n_curr) holding parameter
                 trust opinions.
    T_b        : object array of shape (n_curr,) holding bias trust opinions.
    active_mask: optional boolean array of shape (n_curr,).  When given,
                 neurons whose mask entry is False receive a vacuous opinion
                 (used by IPTA to restrict propagation to fired neurons).
    """
    n_in  = len(T_in)
    n_out = T_W.shape[1]
    T_out: List[Opinion] = []

    for i in range(n_out):
        if active_mask is not None and not active_mask[i]:
            T_out.append(vacuous())
            continue
        # discount each input opinion by the corresponding parameter trust,
        # then fuse all contributions with averaging belief fusion
        contributions = [discount(T_W[j, i], T_in[j]) for j in range(n_in)]
        T_out.append(fuse_many(contributions, how="averaging"))

    return T_out


# ====================================================================== #
#  Trust Feedforward  (Alg. 5)                                           #
# ====================================================================== #

def trust_feedforward(T_x: List[Opinion],
                      tnn: TrustNodesNetwork,
                      fwd: Optional[ForwardCache] = None,
                      path_index: Optional[int] = None,
                      ) -> List[Opinion]:
    """
    Propagate input-feature trust opinions through the Trust Nodes Network.

    Parameters
    ----------
    T_x        : trust opinions for the n_in input features.
    tnn        : the Trust Nodes Network.
    fwd        : forward cache from PrimaryNN; required when ``path_index``
                 is set (IPTA mode).
    path_index : sample index within ``fwd``; when given, hidden-layer trust
                 is restricted to the neurons that fired for that sample,
                 as required by GenIPTA (§7.3).

    Returns
    -------
    List of n_out output-class trust opinions.
    """
    T_h      = T_x
    n_layers = len(tnn.T_Ws)

    # Hidden layers (all layers except the last)
    for k in range(n_layers - 1):
        if path_index is not None and fwd is not None:
            mask = fwd.activated[k][path_index]   # (n_hid_k,) bool
        else:
            mask = None
        T_h = trust_linear_layer(T_h, tnn.T_Ws[k], tnn.T_bs[k],
                                 active_mask=mask)
        # ReLU treated as identity for trust (cf. Def. 7.4 note)

    # Output layer (no ReLU mask)
    T_out = trust_linear_layer(T_h, tnn.T_Ws[-1], tnn.T_bs[-1])
    return T_out


# ====================================================================== #
#  Output-Trust Aggregation                                              #
# ====================================================================== #

def aggregate_output_trust(T_out: List[Opinion],
                           decision: Optional[int] = None) -> Opinion:
    """
    Combine per-class trust opinions into one overall trust score.

    Per §7.3 (last paragraph of "Output-Trust Aggregation"):
    if ``decision`` is given, return only the trust opinion for that class;
    otherwise fuse all classes with averaging belief fusion.
    """
    if decision is not None:
        return T_out[decision]
    return fuse_many(T_out, how="averaging")


# ====================================================================== #
#  GenIPTA -- Inference-Path Trust Assessment  (§7.3)                    #
# ====================================================================== #

def gen_ipta(T_x: List[Opinion],
             tnn: TrustNodesNetwork,
             fwd: ForwardCache,
             sample_idx: int = 0,
             decision: Optional[int] = None) -> Opinion:
    """
    Build a sample-specific trust assessment restricted to the neurons
    that fired during inference (§7.3 GenIPTA).

    Parameters
    ----------
    T_x        : feature trust opinions.
    tnn        : Trust Nodes Network.
    fwd        : forward cache containing the per-layer activation masks.
    sample_idx : index of the sample within ``fwd``.
    decision   : predicted class; if given, return only that class's opinion.
    """
    T_out = trust_feedforward(T_x, tnn, fwd=fwd, path_index=sample_idx)
    return aggregate_output_trust(T_out, decision=decision)


# ====================================================================== #
#  Parameter-Trust Update  (Alg. 6)                                      #
# ====================================================================== #

def _node_trust(grad_row: np.ndarray, eps: float, W: float = 2.0) -> Opinion:
    """
    NODETRUST helper (Alg. 6, step 2).

    Splits the gradient magnitudes of a neuron's incoming weights into:
        r = count of |g| <  eps  (small gradient → positive evidence of stability)
        s = count of |g| >= eps  (large gradient → negative evidence)
    and returns BPQ(r, s).
    """
    r = float(np.sum(np.abs(grad_row) <  eps))
    s = float(np.sum(np.abs(grad_row) >= eps))
    return bpq(r, s, W=W)


def parameter_trust_update(tnn: TrustNodesNetwork,
                           grads: BackwardCache,
                           T_x_batch: List[List[Opinion]],
                           T_y_batch: List[Opinion],
                           T_lr: Opinion,
                           eps: float = 1e-2,
                           y_labels: Optional[np.ndarray] = None) -> None:
    """
    In-place refinement of all parameter trust opinions in ``tnn``.

    Implements the six steps of Alg. 6, generalised to networks with an
    arbitrary number of hidden layers:

        1. Fuse all label opinions in the batch → T_y_global.
        2-3. For each neuron, compute T_{n|y} from gradient magnitudes
             (NODETRUST).
        4. Deduce overall neuron trust via inferential deduction.
        5. REVISE every incoming edge with the deduced neuron trust.
        6. UPDATE with auxiliary factors (lr trust, input-feature trust,
           label trust) via trust discounting:
               T_θ ← discount(T_lr ⊙ (T_x ⊘ T_y_batch), T_θ)

    Layer-specific details
    ----------------------
    * **Output layer** (last): per-class label trust T_y is used when
      ``y_labels`` is provided, so a poisoning attack on a specific class
      affects only the edges feeding that class's output neuron (cf. §7.8.3).
    * **Hidden layers** (all others): batch-wide T_y_global is used.
    * **Input→hidden layer** (k=0): T_x_proxy for edge j is taken from
      T_x_mean[j] (direct per-feature trust).
    * **Deeper layers** (k>0): T_x_proxy is the overall mean input trust
      (T_x_mean fused across all features), since the edge index j no
      longer corresponds directly to an input feature.

    Note on ⊙ in Step 6
    --------------------
    Alg. 6 prescribes T_θ ← T_θ ⊙ (T_x ⊘ T_y_batch).  The ⊙ operator
    in §7.6 admits two readings: Jøsang's binomial AND (which compounds
    distrust unboundedly across batches) or *trust discounting* (which
    leaves T_θ unchanged when the auxiliary factor is fully trusted).
    We adopt discounting — it reproduces the t = 0.87 result in Table 7.2
    and is consistent with the conservative semantics of §7.6.

    Parameters
    ----------
    tnn        : Trust Nodes Network to update (in place).
    grads      : BackwardCache from the current mini-batch.
    T_x_batch  : per-sample, per-feature trust opinions;
                 shape [batch_size][n_features].
    T_y_batch  : per-sample label trust opinions; length = batch_size.
    T_lr       : trust opinion on the chosen learning rate.
    eps        : gradient threshold ε (tie to lr, cf. §7.9).
    y_labels   : integer class labels for the batch; when provided, enables
                 class-aware label trust on the output layer.
    """
    # Step 1 -- batch-wide aggregated label trust
    T_y_global = fuse_many(T_y_batch, how="averaging")

    # Mean per-feature input trust across the batch (one Opinion per feature)
    n_features = len(T_x_batch[0])
    T_x_mean = [
        fuse_many([row[j] for row in T_x_batch], how="averaging")
        for j in range(n_features)
    ]

    # Overall input trust: fuse all feature opinions into a single scalar
    # used as T_x_proxy for edges in non-input layers (k > 0)
    T_x_global = fuse_many(T_x_mean, how="averaging")

    # Per-class label trust for the output layer
    n_layers = len(tnn.T_Ws)
    n_out    = tnn.T_Ws[-1].shape[1]
    if y_labels is not None:
        T_y_per_class: List[Opinion] = []
        for c in range(n_out):
            mask = (y_labels == c)
            if mask.any():
                T_y_per_class.append(fuse_many(
                    [T_y_batch[i] for i in np.where(mask)[0]],
                    how="averaging"))
            else:
                T_y_per_class.append(T_y_global)
    else:
        T_y_per_class = [T_y_global] * n_out

    # ---- Iterate over all layer transitions (output layer first) ----
    for k in range(n_layers - 1, -1, -1):
        is_output_layer = (k == n_layers - 1)
        n_curr = tnn.T_Ws[k].shape[1]   # neurons in the destination layer
        n_prev = tnn.T_Ws[k].shape[0]   # neurons / features in the source layer

        for i in range(n_curr):          # for each destination neuron
            # Label trust for this neuron (per-class only on output layer)
            T_y_i = T_y_per_class[i] if is_output_layer else T_y_global

            # Steps 2-3: NODETRUST -- BPQ on gradient magnitudes
            grad_row       = grads.dWs[k][:, i]
            T_n_given_y    = _node_trust(grad_row, eps)
            T_n_given_noty = vacuous()

            # Step 4: inferential deduction → overall node trust
            T_node = deduce(T_y_i, T_n_given_y, T_n_given_noty)

            for j in range(n_prev):      # for each incoming edge
                # Step 5: revise parameter trust with deduced node trust
                tnn.T_Ws[k][j, i] = revise(tnn.T_Ws[k][j, i], T_node)

                # Step 6: auxiliary modulation via trust discounting
                # For the input→first-hidden layer (k=0), use the direct
                # per-feature trust T_x_mean[j]; for deeper layers use the
                # overall mean input trust (j no longer maps to a feature).
                T_x_proxy = T_x_mean[j] if k == 0 else T_x_global
                aux = multiply(T_lr, conservative_div(T_x_proxy, T_y_i))
                tnn.T_Ws[k][j, i] = discount(aux, tnn.T_Ws[k][j, i])

            # Bias trust (no edge index j)
            tnn.T_bs[k][i] = revise(tnn.T_bs[k][i], T_node)


# ====================================================================== #
#  Convenience -- canonical input-feature trust vectors                  #
# ====================================================================== #

def build_input_trust(n_features: int, kind: str = "vacuous") -> List[Opinion]:
    """Quick constructor for canonical feature-trust profiles."""
    if kind == "trusted":    return [trusted()    for _ in range(n_features)]
    if kind == "distrusted": return [distrusted() for _ in range(n_features)]
    if kind == "vacuous":    return [vacuous()    for _ in range(n_features)]
    raise ValueError(f"Unknown kind {kind!r}; expected 'trusted', 'vacuous', or 'distrusted'.")


def build_label_trust(y_batch: np.ndarray, kind: str = "trusted") -> List[Opinion]:
    """Build trust opinions for each label in a batch."""
    if kind == "trusted":    return [trusted()    for _ in y_batch]
    if kind == "distrusted": return [distrusted() for _ in y_batch]
    if kind == "vacuous":    return [vacuous()    for _ in y_batch]
    raise ValueError(f"Unknown kind {kind!r}; expected 'trusted', 'vacuous', or 'distrusted'.")


# ====================================================================== #
#  Diagnostics                                                            #
# ====================================================================== #

def tnn_stats(tnn: TrustNodesNetwork) -> dict:
    """Return mean belief / disbelief / uncertainty across all TNN opinions.

    Aggregates every weight and bias opinion across all layer transitions.
    Useful as a per-epoch progress signal: b grows from 0 toward 1 as the
    PTU accumulates evidence, u falls correspondingly.

    Returns
    -------
    dict with keys ``b``, ``d``, ``u`` (each a float).
    """
    bs: List[float] = []
    ds: List[float] = []
    us: List[float] = []
    for T_W, T_b in zip(tnn.T_Ws, tnn.T_bs):
        for op in T_W.flat:
            bs.append(op.b); ds.append(op.d); us.append(op.u)
        for op in T_b.flat:
            bs.append(op.b); ds.append(op.d); us.append(op.u)
    return dict(b=float(np.mean(bs)),
                d=float(np.mean(ds)),
                u=float(np.mean(us)))
