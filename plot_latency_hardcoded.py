"""plot_latency_hardcoded.py
Reproduce the latency plots from hardcoded table data — no model files needed.
Linear axes throughout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

_C_NN   = "#555555"
_C_PTAS = "#8e44ad"

# ---------------------------------------------------------------------------
# Hardcoded data
# ---------------------------------------------------------------------------

# Training latency  (5 epochs, 3 trials mean ± std)
TRAIN = {
    #  name          nn_mean  nn_std  ptas_mean  ptas_std  nn_ep   ptas_ep  overhead_ms  ratio
    "clean":    dict(nn_m=284.6, nn_s=66.8, pt_m=348.9, pt_s=45.3,
                     nn_ep=56.91, pt_ep=69.79, overhead_ms=12874, ratio=1.23),
    "feature":  dict(nn_m=269.7, nn_s=2.4,  pt_m=380.4, pt_s=80.8,
                     nn_ep=53.93, pt_ep=76.08, overhead_ms=22144, ratio=1.41),
    "combined": dict(nn_m=240.2, nn_s=19.5, pt_m=275.0, pt_s=40.9,
                     nn_ep=48.05, pt_ep=54.99, overhead_ms=6944,  ratio=1.14),
}

# Inference latency  (median over 200 reps)
#  batch  nn_ms    ptas_ms
INF_ROWS = [
    (1,    0.009,   0.015),
    (8,    0.012,   0.020),
    (32,   0.017,   0.035),
    (64,   0.027,   0.058),
    (128,  0.239,   0.557),
    (256,  0.287,   0.742),
    (512,  0.352,   1.460),
    (1024, 0.589,   3.054),
]

# ---------------------------------------------------------------------------
# Figure: 2 × 2 grid
#   [0,0] Inference lines        [0,1] Training bars
#   [1,0] Inference ratio bars   [1,1] Per-epoch overhead bars
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                         gridspec_kw={"height_ratios": [2, 1]})

# ── Inference latency (top-left) ──────────────────────────────────────────
ax = axes[0, 0]
bs      = [r[0] for r in INF_ROWS]
nn_ms   = [r[1] for r in INF_ROWS]
ptas_ms = [r[2] for r in INF_ROWS]
x_idx   = np.arange(len(bs))

ax.plot(x_idx, nn_ms,   color=_C_NN,   marker="o", lw=2, ms=6, label="NN only")
ax.plot(x_idx, ptas_ms, color=_C_PTAS, marker="s", lw=2, ms=6, label="NN + PaTAS")
ax.set_xticks(x_idx)
ax.set_xticklabels([str(b) for b in bs], fontsize=8)
ax.set_xlabel("Batch size")
ax.set_ylabel("Latency (ms, median)")
ax.set_title("Inference latency vs batch size", fontsize=10)
ax.legend()
ax.set_xlim(-0.5, len(bs) - 0.5)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", alpha=0.4)

# ── Inference overhead ratio (bottom-left) ────────────────────────────────
ax = axes[1, 0]
ratio = [p / max(n, 1e-9) for n, p in zip(nn_ms, ptas_ms)]
bars = ax.bar(x_idx, ratio, color=_C_PTAS, zorder=3, width=0.6)
ax.axhline(1.0, color="black", lw=0.8, linestyle="--")
ax.set_xticks(x_idx)
ax.set_xticklabels([str(b) for b in bs], fontsize=8)
ax.set_xlabel("Batch size")
ax.set_ylabel("Overhead ratio\n(PTAS / NN)")
ax.set_xlim(-0.5, len(bs) - 0.5)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
for bar, r in zip(bars, ratio):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{r:.1f}×", ha="center", va="bottom",
            fontsize=7, color=_C_PTAS)

# ── Training latency bars (top-right) ─────────────────────────────────────
ax = axes[0, 1]
cond_names = list(TRAIN.keys())
x = np.arange(len(cond_names))
w = 0.32

for off, m_key, s_key, color, lbl in [
    (-w / 2, "nn_m",  "nn_s",  _C_NN,   "NN only"),
    ( w / 2, "pt_m",  "pt_s",  _C_PTAS, "NN + PaTAS"),
]:
    means = [TRAIN[c][m_key] for c in cond_names]
    stds  = [TRAIN[c][s_key] for c in cond_names]
    bars  = ax.bar(x + off, means, w, color=color, label=lbl,
                   yerr=stds, capsize=4,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"},
                   zorder=3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{m:.0f}s", ha="center", va="bottom",
                fontsize=7, color=color)

ax.set_xticks(x)
ax.set_xticklabels(cond_names, fontsize=9)
ax.set_ylabel("Total wall-clock (5 epochs, s)")
ax.set_title("Training latency across noise conditions", fontsize=10)
ax.legend()
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

# Ratio annotations above error bars
for i, c in enumerate(cond_names):
    r = TRAIN[c]["ratio"]
    pt_top = TRAIN[c]["pt_m"] + TRAIN[c]["pt_s"]
    ax.text(x[i] + w / 2, pt_top + 18,
            f"{r:.2f}×", ha="center", va="bottom",
            fontsize=7, color=_C_PTAS, style="italic")

# ── Per-epoch overhead (bottom-right) ─────────────────────────────────────
ax = axes[1, 1]
overhead_ms = [TRAIN[c]["overhead_ms"] for c in cond_names]
bars2 = ax.bar(x, overhead_ms, 0.55, color=_C_PTAS, zorder=3)
for bar, ov in zip(bars2, overhead_ms):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 150,
            f"{ov/1000:.1f}s", ha="center", va="bottom",
            fontsize=8, color=_C_PTAS)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cond_names, fontsize=9)
ax.set_ylabel("PaTAS overhead / epoch (ms)")
ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

# ---------------------------------------------------------------------------
fig.suptitle(
    "PaTAS latency overhead — inference (left) and training (right, 5 epochs, 3 trials)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig("latency_plot.pdf", bbox_inches="tight")
fig.savefig("latency_plot.png", bbox_inches="tight")
plt.close(fig)
print("Saved latency_plot.pdf / .png")
