"""plot_effectiveness_hardcoded.py
Reproduce the patas_effectiveness plot (feature noise + combined noise)
from hardcoded table data -- no model files required.
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

_C = {"b": "#2b7bba", "d": "#c0392b", "u": "#e07b39"}

# -- Hardcoded data -----------------------------------------------------------

fn_records = {
    0.00: (0.8086, 0.5294, 0.0000, 0.0000, 0.1914, 0.4706),
    0.10: (0.4317, 0.1487, 0.0480, 0.0169, 0.5203, 0.8344),
    0.30: (0.1776, 0.0446, 0.0761, 0.0191, 0.7463, 0.9363),
    0.50: (0.0703, 0.0173, 0.0703, 0.0173, 0.8593, 0.9643),
}  # sigma -> (b_c, b_w, d_c, d_w, u_c, u_w)

COMBINED = [(0.10, 0.05), (0.30, 0.15), (0.50, 0.30)]
cb_records = {
    (0.10, 0.05): (0.4224, 0.1954, 0.0469, 0.0219, 0.5307, 0.7827),
    (0.30, 0.15): (0.1670, 0.1350, 0.0715, 0.0573, 0.7615, 0.8077),
    (0.50, 0.30): (0.0651, 0.0631, 0.0651, 0.0631, 0.8698, 0.8748),
}

# -- Panel specs --------------------------------------------------------------

fn_x     = sorted(fn_records.keys())          # [0.00, 0.10, 0.30, 0.50]
cb_x     = list(range(len(COMBINED)))          # [0, 1, 2]
cb_ticks = [f"$\\sigma$={s:.2f}\n$p$={p:.2f}" for s, p in COMBINED]

nan6 = (float("nan"),) * 6

panels = [
    (
        {v: fn_records[v] for v in fn_x},
        fn_x, None,
        r"Feature noise $\sigma_{rel}$",
        r"Feature noise ($\sigma_{rel}$-calibrated $T_x$)",
    ),
    (
        {i: cb_records[k] for i, k in enumerate(COMBINED)},
        cb_x, cb_ticks,
        "Condition",
        "Combined noise ($T_x$ calibrated)",
    ),
]

# -- Figure -------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                         gridspec_kw={"height_ratios": [2, 1]})

for col, (records, x_vals, tick_labels, xlabel, title) in enumerate(panels):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    b_c = [records.get(v, nan6)[0] for v in x_vals]
    b_w = [records.get(v, nan6)[1] for v in x_vals]
    d_c = [records.get(v, nan6)[2] for v in x_vals]
    d_w = [records.get(v, nan6)[3] for v in x_vals]
    u_c = [records.get(v, nan6)[4] for v in x_vals]
    u_w = [records.get(v, nan6)[5] for v in x_vals]

    xs     = np.array(x_vals, dtype=float)
    bw_bar = min(np.diff(xs).min() if len(xs) > 1 else 1.0,
                 0.04 if tick_labels is None else 0.25) * 0.9
    xlim   = (xs[0] - 2.5 * bw_bar, xs[-1] + 2.5 * bw_bar)

    # Top panel: 6 lines
    for vals, color, ls, mk, lbl in [
        (b_c, _C["b"], "-",  "o", "Belief (correct)"),
        (b_w, _C["b"], "--", "s", "Belief (wrong)"),
        (d_c, _C["d"], "-",  "o", "Disbelief (correct)"),
        (d_w, _C["d"], "--", "s", "Disbelief (wrong)"),
        (u_c, _C["u"], "-",  "o", "Uncertainty (correct)"),
        (u_w, _C["u"], "--", "s", "Uncertainty (wrong)"),
    ]:
        ax_top.plot(x_vals, vals, color=color, linestyle=ls,
                    marker=mk, lw=2, ms=6, label=lbl)

    if col == 0:
        ax_top.set_ylabel("Mean opinion mass in NN's predicted class")
    ax_top.set_title(title, fontsize=10)
    ax_top.legend(fontsize=6, ncol=2)

    all_vals = [v for lst in [b_c, b_w, d_c, d_w, u_c, u_w]
                for v in lst if not np.isnan(v)]
    if all_vals:
        ylo, yhi = min(all_vals), max(all_vals)
        margin = max(0.03, (yhi - ylo) * 0.3)
        ax_top.set_ylim(max(0.0, ylo - margin), min(1.0, yhi + margin))

    ax_top.set_xlim(*xlim)
    if tick_labels is not None:
        ax_top.set_xticks(x_vals)
        ax_top.set_xticklabels(tick_labels, fontsize=7)

    # Bottom panel: gap bars
    gaps_b = [bc - bw for bc, bw in zip(b_c, b_w)]
    gaps_d = [dc - dw for dc, dw in zip(d_c, d_w)]
    gaps_u = [uc - uw for uc, uw in zip(u_c, u_w)]

    for off, gaps, color, lbl in zip(
        [-bw_bar, 0.0, bw_bar],
        [gaps_b, gaps_d, gaps_u],
        [_C["b"], _C["d"], _C["u"]],
        ["b", "d", "u"],
    ):
        bars = ax_bot.bar(xs + off, gaps, width=bw_bar,
                          color=color, label=lbl, zorder=3)
        for bar, g in zip(bars, gaps):
            if abs(g) < 1e-9:
                continue
            va  = "bottom" if g >= 0 else "top"
            pad = 0.0005   if g >= 0 else -0.0005
            ax_bot.text(bar.get_x() + bar.get_width() / 2, g + pad,
                        f"{g:+.3f}", ha="center", va=va,
                        fontsize=6, color=color)

    ax_bot.axhline(0, color="black", lw=0.8)
    ax_bot.set_xlabel(xlabel)
    if col == 0:
        ax_bot.set_ylabel(r"$\Delta$ (correct $-$ wrong)")
    ax_bot.set_xlim(*xlim)
    ax_bot.legend(fontsize=6, ncol=3)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    if tick_labels is not None:
        ax_bot.set_xticks(x_vals)
        ax_bot.set_xticklabels(tick_labels, fontsize=7)

fig.suptitle("PaTAS as NN confidence signal: opinion masses in predicted class",
             fontsize=12)
fig.tight_layout()
fig.savefig("patas_effectiveness_test.pdf", bbox_inches="tight")
fig.savefig("patas_effectiveness_test.png", bbox_inches="tight")
plt.close(fig)
print("Saved patas_effectiveness_test.pdf / .png")
