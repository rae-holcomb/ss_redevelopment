"""Is each source's period present in BOTH halves of the year (t < 182.5 d vs >= 182.5 d) of its A+B stack?
The note's A/B test splits member *files*, which share the same noise light curve, so a red-noise peak in g
passes it; a time split is the relevant persistence test. Statistic: per half, local S/N of LS power at the
reference period (power / median power within +-25% in frequency, excluding +-3% of the period) and whether the
half's top peak in 0.1-60 d lies within 5% of P (or P/2, 2P). For quiet sources P = their LS top peak."""
import sys
import numpy as np, pandas as pd
from astropy.timeseries import LombScargle
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
sys.path.insert(0, f"{A}/tools/audit")
import galaxy_pass as gp
D = "/Volumes/Data_Vault/smarts_data/extracted/galaxy_stack"
S = np.load(f"{D}/stacks.npy", mmap_mode="r")
idx = pd.read_csv(f"{D}/stack_index.csv")
src = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv").set_index("noise_tic")
t_all = np.arange(gp.N_CAD) / 48.0
def clean(g):
    bad = ~np.isfinite(g) | (np.abs(g - 1) > gp.OUTLIER_THRESH)
    flat = np.zeros(len(g), bool); flat[1:-1] = gp.collinear(np.where(bad, 1.0, g)); bad |= flat
    return ~bad
def half_stats(t, y, P):
    if len(t) < 500 or P > 60: return np.nan, False
    fmax = 1 / 0.1; f = np.linspace(1 / 60, fmax, int((fmax - 1 / 60) * 182 * 5))
    ls = LombScargle(t, y); p = ls.power(f, method="fast")
    f0 = 1 / P; band = (np.abs(f / f0 - 1) < 0.25) & (np.abs(f / f0 - 1) > 0.03)
    snr = float(ls.power(f0) / np.median(p[band])) if band.sum() > 5 else np.nan
    top = 1 / f[np.argmax(p)]
    hit = any(abs(top / q - 1) < 0.05 for q in (P, P / 2, 2 * P))
    return snr, hit
rec = []
for i, tic in enumerate(idx.tic):
    s = src.loc[tic]
    P = s.P_ref_d if s.periodic else s.ls_P1_d
    g = np.asarray(S[i, 2], float); ok = clean(g)
    r = dict(tic=tic)
    for h, m in (("h1", t_all < 182.5), ("h2", t_all >= 182.5)):
        mm = m & ok
        r[f"snr_{h}"], r[f"top_{h}"] = half_stats(t_all[mm], g[mm] - np.median(g[mm]) if mm.any() else g[mm], P)
    rec.append(r)
r = pd.DataFrame(rec).set_index("tic").join(src[["classification", "periodic", "P_ref_d", "ls_P1_d", "search_detected", "ls_periodic", "ffi_check"]])
r["both_top"] = r.top_h1 & r.top_h2
r["min_snr"] = r[["snr_h1", "snr_h2"]].min(axis=1)
r["long"] = r.P_ref_d.fillna(r.ls_P1_d) >= 5
r.to_csv(sys.argv[1])
qs = r[~r.periodic]
thr = qs.min_snr.quantile(0.95)
print("quiet min-half S/N 95th pct:", round(thr, 2), "99th:", round(qs.min_snr.quantile(.99), 2))
r["coh"] = r.min_snr > thr
for lab, d in (("quiet", qs), ("periodic", r[r.periodic])):
    for L, dd in d.groupby("long"):
        print(f"{lab:9s} P>=5:{L!s:5s} n={len(dd):4d}  top peak in both halves={dd.both_top.mean():.2f}  min-half S/N > quiet95={ (dd.min_snr > thr).mean():.2f}")
p = r[r.periodic]
print(p.groupby("classification").agg(n=("coh", "size"), coh=("coh", "mean"), both_top=("both_top", "mean")).round(2).to_string())
print("pipeline-only (not ls_periodic):", p[~p.ls_periodic].coh.mean().round(2), "n", (~p.ls_periodic).sum(),
      "| ls_periodic:", p[p.ls_periodic].coh.mean().round(2))
print("by ffi_check:", p.groupby("ffi_check").coh.mean().round(2).to_dict())
print("periodic P>=5 that fail time coherence:", int((p.long & ~p.coh).sum()), "of", int(p.long.sum()))
