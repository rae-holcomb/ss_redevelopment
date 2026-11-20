"""Gate-failure breakdown of the leader's lost stars vs _v7 (success under _v7, not under the leader)."""
import sys
import pandas as pd
from style import plt, C, INK2
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
LEAD = sys.argv[1] if len(sys.argv) > 1 else "local_seq_trim"
S = {"correct": "succ", "diffrot": "succ", "half": "alias", "double": "alias", "wrong": "wrong", "none": "none"}
GATES = ["n_peaks", "frac_positive_heights", "mean_height_snr", "frac_vertex_in_window", "half_lag_consistency"]
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
rows = []
for m, lab in [("fy", "full-year"), ("as", "assigned")]:
    b = pd.read_csv(ROOT / f"runs/t32_clean/s2/local_screen2_{m}.csv")
    d = pd.read_csv(ROOT / f"runs/t32_clean/s2/{LEAD}_{m}.csv")
    x = d.merge(b[["star_id", "final_outcome"]], on="star_id", suffixes=("", "_v7"))
    L = x[(x.final_outcome_v7.map(S) == "succ") & (x.final_outcome.map(S) != "succ")].copy()
    L["mode"] = lab
    rows.append(L)
L = pd.concat(rows)
L.to_csv("data/leader_lost_stars.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
for k, lab in enumerate(["full-year", "assigned"]):
    s = L[L["mode"] == lab]
    fg = s.best_success_failed_gates.fillna("")
    cnt = [fg.str.contains(g).sum() for g in GATES]
    sole = [(fg == g).sum() for g in GATES]
    y = [i + 0.2 * (1 - 2 * k) for i in range(len(GATES))]
    ax[0].barh(y, cnt, height=0.38, color=C[k], label=f"{lab} (n={len(s)}): fails gate (any)")
    ax[0].barh(y, sole, height=0.38, color="none", edgecolor=INK2, hatch="///", lw=0.8,
               label="… as its only failed gate" if k == 0 else None)
    for yy, c in zip(y, cnt):
        ax[0].text(c + 0.2, yy, str(c), va="center", fontsize=7, color=INK2)
    other = s.final_outcome.map(S).value_counts()
    ax[1].bar([i + 0.2 * (2 * k - 1) for i in range(5)], [(s.period_bin == pb).sum() for pb in BINS], width=0.38,
              color=C[k], label=f"{lab}: {other.get('none', 0)} declined, {other.get('wrong', 0) + other.get('alias', 0)} wrong/alias")
ax[0].set_yticks(range(len(GATES)), GATES)
ax[0].invert_yaxis()
ax[0].set_xlabel("lost stars whose best success-candidate fails the gate")
ax[0].legend(fontsize=7, loc="lower right")
ax[1].set_xticks(range(5), BINS)
ax[1].set_xlabel("true-period bin")
ax[1].set_ylabel("lost stars")
ax[1].legend(fontsize=7)
fig.suptitle(f"{LEAD}: why it loses _v7's successes (clean dev_medium)", fontsize=10)
fig.savefig("figures/fig_leader_lost_gates.png")
