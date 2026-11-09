"""Figures of the 34 'quiet' noise sources with coherent short-period signals (from c07c):
(1) overview: full-year periodogram of each A+B stack, period marked;
(2) detail: light curve, fold at P split by half of the year, and per-half periodograms near P,
    for a representative subset plus reference sources (a periodic source of similar amplitude, a typical quiet one)."""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
sys.path.insert(0, f"{A}/tools/audit")
import galaxy_pass as gp
HERE = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-note-review-2d28d8/reports/2026-11-09_smarts-note-review"
D = "/Volumes/Data_Vault/smarts_data/extracted/galaxy_stack"
S = np.load(f"{D}/stacks.npy", mmap_mode="r")
idx = pd.read_csv(f"{D}/stack_index.csv")
row = {t: i for i, t in enumerate(idx.tic)}
src = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv").set_index("noise_tic")
q = pd.read_csv(f"{HERE}/checks/quiet_coherent.csv").sort_values("P").reset_index(drop=True)
t_all = np.arange(gp.N_CAD) / 48.0

def get(tic):
    """Cleaned A+B stack (corrupt / interpolated cadences dropped), as ppm about the median."""
    g = np.asarray(S[row[tic], 2], float)
    bad = ~np.isfinite(g) | (np.abs(g - 1) > gp.OUTLIER_THRESH)
    flat = np.zeros(len(g), bool); flat[1:-1] = gp.collinear(np.where(bad, 1.0, g)); bad |= flat
    t, y = t_all[~bad], g[~bad]
    return t, (y / np.median(y) - 1) * 1e6

FREQ = np.linspace(1 / 20, 1 / 0.1, 40000)

# ---- (1) overview
n = len(q); nc = 5; nr = int(np.ceil(n / nc))
fig, axs = plt.subplots(nr, nc, figsize=(15, 2.0 * nr), sharex=True)
for ax, r in zip(axs.flat, q.itertuples()):
    t, y = get(r.tic)
    p = LombScargle(t, y).power(FREQ, method="fast")
    ax.plot(1 / FREQ, p, lw=0.5, color="0.3")
    ax.axvline(r.P, color="C3", lw=0.8, alpha=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_title(f"TIC {r.tic}  P={r.P:.3f} d  {r.ls_P1_amp_ppm:.0f} ppm\nS/N halves {r.snr_h1:.0f}, {r.snr_h2:.0f}"
                 + ("  LMC" if r.near_lmc else ""), fontsize=7, loc="left")
    ax.tick_params(labelsize=6)
for ax in axs.flat[n:]: ax.axis("off")
for ax in axs[-1]: ax.set_xlabel("period (d)", fontsize=8)
fig.suptitle("34 sources classed 'quiet' whose short-period LS peak (red dashes) stands > 20x above the local periodogram level in both halves of the year", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(f"{HERE}/figures/quiet_coherent_periodograms.png", dpi=110); plt.close(fig)

# ---- (2) detail
pick = list(q.iloc[np.linspace(0, n - 1, 6).round().astype(int)].tic)
fam = q[np.isclose(q.P, 0.829, atol=0.002)].tic.tolist()[:1]
pick = list(dict.fromkeys(pick + fam))
per = src[src.periodic & (src.P_ref_d < 2) & src.ls_P1_amp_ppm.between(200, 400)]
ref_per = per.index[0]
ref_q = src[~src.periodic & (src.ls_P1_d > 10)].sample(1, random_state=3).index[0]
rows = [(t, float(q.set_index("tic").P[t]), "quiet (coherent)") for t in pick]
rows += [(ref_per, float(src.P_ref_d[ref_per]), f"reference: {src.classification[ref_per]}"),
         (ref_q, float(src.ls_P1_d[ref_q]), "reference: typical quiet (fold at its LS peak)")]
fig, axs = plt.subplots(len(rows), 3, figsize=(14, 2.1 * len(rows)), gridspec_kw=dict(width_ratios=[2.2, 1, 1.2]))
for (tic, P, lab), (a0, a1, a2) in zip(rows, axs):
    t, y = get(tic)
    a0.plot(t, y, ",", color="0.6", rasterized=True)
    nb = 48 // 4   # 6-h bins
    k = len(t) // nb * nb
    a0.plot(t[:k].reshape(-1, nb).mean(1), y[:k].reshape(-1, nb).mean(1), ".", ms=1.5, color="k")
    lo, hi = np.percentile(y, [1, 99]); a0.set_ylim(lo * 1.3, hi * 1.3)
    s = src.loc[tic]
    a0.set_title(f"TIC {tic} — {lab}; P = {P:.4g} d, LS amp {s.ls_P1_amp_ppm:.0f} ppm, power {s.ls_P1_power:.3f}, "
                 f"{int(s.n_files)} files", fontsize=8, loc="left")
    a0.set_ylabel("ppm", fontsize=7)
    edges = np.linspace(0, 1, 41); c = 0.5 * (edges[1:] + edges[:-1])
    for h, m, col in (("days 0-182", t < 182.5, "C0"), ("days 182-365", t >= 182.5, "C3")):
        ph = (t[m] / P) % 1
        prof = [np.median(y[m][(ph >= edges[i]) & (ph < edges[i + 1])]) for i in range(40)]
        a1.plot(np.r_[c, c + 1], np.r_[prof, prof], "-", color=col, lw=1, label=h)
        f0 = 1 / P; ff = np.linspace(f0 * 0.75, f0 * 1.25, 3000)
        a2.plot(1 / ff, LombScargle(t[m], y[m]).power(ff), color=col, lw=0.7)
    a1.set_xlabel("phase (two cycles)", fontsize=7); a1.set_ylabel("median ppm", fontsize=7)
    a2.axvline(P, color="k", lw=0.6, ls=":"); a2.set_xlabel("period (d), ±25% around P", fontsize=7)
    for a in (a0, a1, a2): a.tick_params(labelsize=6)
axs[0, 1].legend(fontsize=6)
axs[0, 2].set_title("LS power, each half of the year", fontsize=8)
fig.tight_layout()
fig.savefig(f"{HERE}/figures/quiet_coherent_detail.png", dpi=110); plt.close(fig)
print("picked", pick, "refs", ref_per, ref_q)
print("period clusters among the 34:", q.P.round(2).value_counts().loc[lambda s: s > 1].to_dict())
