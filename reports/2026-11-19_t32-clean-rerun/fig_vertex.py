"""Figure: free vertex residual vs n for the three groups (step 1)."""
import numpy as np, pandas as pd
from style import plt, C, INK2
d = pd.read_csv("data/vertex_residuals.csv")
G = [("gain", "(a) local gains over _v7"), ("none2wrong", "(b) none -> wrong under local"), ("shared", "(c) shared correct")]
fig, ax = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey="row")
rng = np.random.default_rng(0)
for j, (g, title) in enumerate(G):
    s = d[d.group == g]
    nst = s.groupby(["mode", "star_id"]).ngroups
    for i, (col, lab) in enumerate([("resid_line", "free vertex − (t0 + nP)  [units of P]"),
                                    ("off_seed", "free vertex − seed window centre  [window half-widths]")]):
        a = ax[i, j]
        x = s.n + rng.uniform(-0.18, 0.18, len(s))
        acc = s.accepted.values
        a.scatter(x[acc], s[col][acc], s=9, color=C[0], alpha=0.55, lw=0, label="accepted by local (max, in window)")
        a.scatter(x[~acc], s[col][~acc], s=9, color=C[1], alpha=0.55, lw=0, label="rejected (outside window / not a max)")
        med = s.assign(a=s[col].abs()).groupby("n").a.median()
        a.plot(med.index, med.values, color="#0b0b0b", lw=1.6, label="median |value|")
        a.plot(med.index, -med.values, color="#0b0b0b", lw=1.6)
        if i == 1:
            for y in (-1, 1):
                a.axhline(y, color=INK2, ls="--", lw=1)
            a.set_ylim(-2.2, 2.2)
            a.set_xlabel("harmonic n (tooth index)")
        else:
            a.set_ylim(-0.4, 0.4)
            frac = s.groupby("n").accepted.mean()
            for n, f in frac.items():
                a.text(n, 0.36, f"{f:.0%}", ha="center", fontsize=7, color=INK2)
        if j == 0:
            a.set_ylabel(lab, fontsize=8)
        if i == 0:
            a.set_title(f"{title}  ({nst} star-modes)")
ax[0, 0].text(-0.4, 0.30, "% accepted ↑", fontsize=7, color=INK2)
ax[1, 2].legend(loc="lower right", fontsize=7, markerscale=2)
fig.suptitle("Free per-window ACF vertices of local's selected comb (clean dev_medium, full-year + assigned)", fontsize=11)
fig.savefig("figures/fig_vertex_residual_vs_n.png")
