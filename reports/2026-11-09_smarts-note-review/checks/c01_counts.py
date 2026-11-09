"""Recompute the note's per-class counts, fractions and ranges from the public tables."""
import sys
import numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
print("rows", len(S), "unique tic", S.noise_tic.is_unique, "cols", len(S.columns))
g = S.groupby("classification").agg(n=("noise_tic", "size"), files=("n_files", "sum"),
    medP=("P_ref_d", "median"), medamp=("ls_P1_amp_ppm", "median"), lmc=("near_lmc", "sum"))
print(g)
p = S[S.periodic]
print("periodic", len(p), p.n_files.sum(), "files total", S.n_files.sum(), "median P_ref", p.P_ref_d.median(),
      "P range", p.P_ref_d.min(), p.P_ref_d.max())
print("amp range periodic (ls_P1_amp_ppm)", p.ls_P1_amp_ppm.min(), p.ls_P1_amp_ppm.max())
print("n_files range", S.n_files.min(), S.n_files.max())
print("search_detected", S.search_detected.sum(), "ls_periodic", S.ls_periodic.sum(), "both", (S.search_detected & S.ls_periodic).sum(),
      "search only", (S.search_detected & ~S.ls_periodic).sum(), "search & not ab_persistent", (S.search_detected & ~S.ab_persistent).sum())
print("periodic == search|ls", (S.periodic == (S.search_detected | S.ls_periodic)).all())
print("P_ref==search_P where detected", np.allclose(p.P_ref_d[p.search_detected], p.search_P_d[p.search_detected]),
      "P_ref==ls_P1 else", np.allclose(p.P_ref_d[~p.search_detected], p.ls_P1_d[~p.search_detected]))
print("P_ref != ls_P1 (>2%) among periodic:", (abs(p.P_ref_d/p.ls_P1_d-1) > .02).sum())
print("ab_persistent all", S.ab_persistent.sum(), "quiet", S[~S.periodic].ab_persistent.mean())
print("ls power median, 90th", S.ls_P1_power.median(), S.ls_P1_power.quantile(.9))
print("quiet ls power median/90/max", S[~S.periodic].ls_P1_power.quantile([.5,.9,.99,1]).values)
print("quiet with power>=0.1 (not persistent)", ((~S.periodic) & (S.ls_P1_power>=.1)).sum())
# LMC
n = S[S.near_lmc]; o = S[~S.near_lmc]
print("near lmc", len(n), "periodic", n.periodic.sum(), n.periodic.mean(), "elsewhere", o.periodic.sum(), len(o), o.periodic.mean(),
      "files near", n.n_files.sum(), "ratio", n.periodic.mean()/o.periodic.mean())
print("median n_gaia_nb near/else", n.n_gaia_nb.median(), o.n_gaia_nb.median())
q = S[~S.periodic]
for lab, d in (("near", S.near_lmc), ("else", ~S.near_lmc)):
    print(lab, "quiet with var nb", (q[d[~S.periodic]].n_gaia_nb_var > 0).mean(), "periodic with var", (p[d[S.periodic]].n_gaia_nb_var>0).mean(),
          "quiet with bright nb", (q[d[~S.periodic]].n_gaia_nb_bright > 0).mean())
print("d_lmc consistent", ((S.d_lmc_deg < 4) == S.near_lmc).all())
# corrupted
c = S[S.has_corrupted_segment]
print("corrupted", len(c), c.n_files.sum(), c.n_files.sum()/1e6, "periodic frac corr/not", c.periodic.mean(), S[~S.has_corrupted_segment].periodic.mean())
# exclusion
ex = S[S.classification.isin(["contamination_known_variable","contamination_candidate_crowded","source_specific_unexplained"])]
print("exclude", len(ex), ex.n_files.sum())
print(S.ffi_check.value_counts().to_dict())
print("ffi_check vs P_ref<5:", pd.crosstab(S.ffi_check, S.P_ref_d < 5))
# five largest
print(p.nlargest(6, "ls_P1_amp_ppm")[["noise_tic","classification","P_ref_d","ls_P1_amp_ppm","gaia_nb_var_classes","near_lmc"]].to_string())
print("classes in known var", p[p.classification=="contamination_known_variable"].gaia_nb_var_classes.str.split(";").explode().value_counts().to_dict())
print("known var with no class string", (p[p.classification=="contamination_known_variable"].gaia_nb_var_classes.isna()).sum())
print("documented-cols check", list(S.columns))
