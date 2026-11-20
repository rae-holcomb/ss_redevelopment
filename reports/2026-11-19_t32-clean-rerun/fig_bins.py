"""Success / alias / wrong by period bin for _v7, local and every T32 variant (Stage-2 screen = scorecard per star)."""
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from style import plt
b = pd.read_csv("data/screen_by_bin.csv")
s = pd.read_csv("data/screen_summary.csv")
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d", "all"]
s2 = s.rename(columns={}).assign(period_bin="all")[["variant", "mode", "period_bin", "n", "success", "alias", "wrong"]]
b = pd.concat([b, s2])
b = b[b.variant != "local_screen2"]  # identical to _v7 per star
for mode, lab in [("fy", "full-year"), ("as", "assigned")]:
    d = b[b["mode"] == mode]
    order = s[(s["mode"] == mode) & (s.variant != "local_screen2")].sort_values("success", ascending=False).variant.tolist()
    order = ["_v7"] + [v for v in order if v != "_v7"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.32 * len(order) + 1.6), sharey=True)
    for ax, met, sign in zip(axes, ["success", "alias", "wrong"], [1, -1, -1]):
        M = d.pivot(index="variant", columns="period_bin", values=met).loc[order, BINS]
        D = (M - M.loc["_v7"]) * sign
        lim = max(0.02, np.nanmax(np.abs(D.values)))
        ax.imshow(D.values, cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if i == 0 else "normal")
        ax.set_xticks(range(len(BINS)), BINS)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(f"{met} (blue = better than _v7)")
        ax.grid(False)
        ax.axvline(4.5, color="#0b0b0b", lw=1)
    n = d[d.variant == "_v7"].set_index("period_bin").n
    fig.suptitle(f"{lab}, clean dev_medium: outcome fraction by true-period bin  "
                 f"(n per bin: {', '.join(f'{k} {int(n[k])}' for k in BINS)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"figures/fig_by_bin_{mode}.png")
