"""Absolute-threshold version of the time-split test, and the 'quiet' sources with coherent short-period signals."""
import sys, numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
r = pd.read_csv(sys.argv[1])
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv").set_index("noise_tic")
r = r.join(S[["ls_P1_power", "ls_P1_amp_ppm", "near_lmc", "n_gaia_nb_var", "gaia_nb_var_classes", "n_files", "ra_deg", "dec_deg"]], on="tic")
r["P"] = r.P_ref_d.fillna(r.ls_P1_d)
r["bin"] = pd.cut(r.P, [0.1, 0.5, 1.5, 5, 10, 20, 61])
for thr in (10, 20):
    r["coh"] = r.min_snr > thr
    print(f"--- min-half S/N > {thr}")
    print(r.groupby(["bin", "periodic"], observed=True).coh.agg(["size", "mean"]).unstack().round(2).to_string())
q = r[~r.periodic & (r.min_snr > 20) & (r.P < 5)].sort_values("P")
print("quiet coherent (P<5, S/N>20 both halves):", len(q), "files", q.n_files.sum(), "near LMC", q.near_lmc.sum(),
      "with Gaia var nb", (q.n_gaia_nb_var > 0).sum())
print("their LS power range", q.ls_P1_power.min(), q.ls_P1_power.max(), "amp ppm median/range", q.ls_P1_amp_ppm.median(), q.ls_P1_amp_ppm.min(), q.ls_P1_amp_ppm.max())
print(q[["tic", "P", "snr_h1", "snr_h2", "ls_P1_power", "ls_P1_amp_ppm", "near_lmc", "gaia_nb_var_classes"]].head(12).round(3).to_string())
q.to_csv(sys.argv[2], index=False)
# same signal shared with a periodic neighbour?
p = r[r.periodic]
def nbr(row):
    d = np.hypot((p.ra_deg - row.ra_deg) * np.cos(np.radians(row.dec_deg)), p.dec_deg - row.dec_deg) * 60
    m = (d < 10) & ((np.abs(p.P / row.P - 1) < 0.02) | (np.abs(p.P / (2*row.P) - 1) < 0.02) | (np.abs(2*p.P / row.P - 1) < 0.02))
    return int(m.sum())
q = q.assign(n_periodic_nb_same_P=q.apply(nbr, axis=1))
print("quiet-coherent with a periodic source within 10' sharing the period:", (q.n_periodic_nb_same_P > 0).sum())
