"""Recompute FFI-table statistics (Wilson, Fisher), stack residual numbers, the dev_medium test, and
chance rates for the Gaia-variable / crowding attributions."""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest
import sys
sys.path.insert(0, "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1/tools/audit")
from smarts_release_products import wilson
def proportion_confint(k, n, method=None):
    return wilson(k, n)
def mh_or(tabs):
    num = sum(t[0,0]*t[1,1]/t.sum() for t in tabs); den = sum(t[0,1]*t[1,0]/t.sum() for t in tabs); return num/den
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
T = f"{A}/reports/2026-10-21_t20-noise-sources-phase2/data"
src = pd.read_csv(f"{T}/sources.csv")
f = pd.read_csv(f"{T}/ffi_compare_all.csv")
ft = f[f.testable]
c = ft[ft.control]; k0, n0 = int(c.confirmed.sum()), len(c)
print("controls", k0, n0, "all controls", len(f[f.control]))
for cls, g in ft[~ft.control].groupby("classification"):
    k, n = int(g.confirmed.sum()), len(g)
    lo, hi = proportion_confint(k, n, method="wilson")
    p1 = fisher_exact([[k, n - k], [k0, n0 - k0]], alternative="greater")[1]
    p2 = fisher_exact([[k, n - k], [k0, n0 - k0]])[1]
    print(f"{cls:34s} {k}/{n} {k/n:.2f} [{lo:.2f},{hi:.2f}] p1={p1:.2g} p2={p2:.2g} bonf5={min(1,5*p1):.2g}")
lo, hi = proportion_confint(k0, n0, method="wilson"); print("null CI", lo, hi)
kv = ft[(ft.classification == "contamination_known_variable") & ~ft.control]
un = ft[(ft.classification == "source_specific_unexplained") & ~ft.control]
print("known vs unexplained fisher p", fisher_exact([[kv.confirmed.sum(), len(kv)-kv.confirmed.sum()], [un.confirmed.sum(), len(un)-un.confirmed.sum()]])[1])
print("match types among confirmed:", ft[ft.confirmed].groupby(["control", "match"]).size().to_dict())
print("confirmed with match==P only, by class:", ft[ft.match == "P"].groupby(["control","classification"]).size().to_dict())
print("controls: own P_ref tested? P_ref_own present:", "P_ref_own" in f.columns)
print("control testable P_ref dist", np.round(np.sort(c.P_ref.values), 2))
print("authors by class", ft.groupby(["classification", "authors"]).size().unstack(fill_value=0).to_string())
print("ffi_n_sectors_matching among confirmed:", ft[ft.confirmed].n_sectors_matching.value_counts().to_dict())
# controls: how many near LMC / which quiet
q = src.set_index("tic").loc[c.tic]
print("controls near LMC", q.near_lmc.sum(), "of", len(q), "; periodic sources testable near LMC:",
      src.set_index("tic").loc[ft[~ft.control].tic].near_lmc.mean())
# ---- residual numbers
r = pd.read_csv("/Volumes/Data_Vault/smarts_data/extracted/galaxy_stack/residuals.csv")
print("residuals roles", r.role.value_counts().to_dict())
for role, g in r.groupby(r.role == "check"):
    print("check" if role else "stack", "res_rms median", g.res_rms_ppm.median(), "res_p2p median", g.res_p2p_ppm.median())
# typical p2p of a smarts file: galaxy pass stat
print("median member p2p (pass stats) per source median:", src.p2p_ppm.median())
# ---- dev_medium
d = pd.read_csv(f"{T}/devmedium_vs_source.csv").merge(src[["galaxy", "classification", "periodic", "pipe_P", "pool"]], on="galaxy")
print("devmedium n", len(d), "pools", d.pool.value_counts().to_dict(), "outcomes", d.final_outcome.value_counts(dropna=False).to_dict())
w = d.final_outcome == "wrong"
print("wrong", w.sum(), "strict match among wrong", (w & d.strict_match).sum())
for lab, m in (("periodic", d.periodic), ("quiet", ~d.periodic)):
    g = d[m]; oc = g.final_outcome.fillna("none").value_counts()
    print(lab, len(g), "wrong", (g.final_outcome == "wrong").sum(), "success(correct+diffrot)", g.final_outcome.isin(["correct", "diffrot"]).mean(),
          "dist", (oc / len(g)).round(3).to_dict())
print("strict matches among wrong by class", d[w & d.strict_match].classification.value_counts().to_dict())
print("strict matches among ALL outcomes", d[d.strict_match].final_outcome.value_counts().to_dict())
print("true period of wrong strict matches: range", d[w & d.strict_match].true_period.describe()[["min", "50%", "max"]].to_dict())
print("strict-match wrong: P_noise range", d[w & d.strict_match].pipe_P.describe()[["min","50%","max"]].to_dict())
# ---- chance attribution
p = src[src.periodic]
order_ok = ~p.classification.isin(["systematic_common_mode", "misclassified_star"])
pp = p[order_ok]
qq = src[~src.periodic]
exp_var = sum(len(pp[pp.near_lmc == L]) * (qq[qq.near_lmc == L].n_nb_var > 0).mean() for L in (True, False))
print("eligible periodic", len(pp), "observed with var nb", (pp.n_nb_var > 0).sum(), "expected by chance", exp_var)
# periodic rate given var-nb, stratified by LMC
tabs = []
for L in (True, False):
    s = src[src.near_lmc == L]
    t = pd.crosstab(s.n_nb_var > 0, s.periodic).reindex(index=[True, False], columns=[True, False]).values
    tabs.append(t)
    print("LMC" if L else "off-LMC", "var nb: periodic rate", t[0,0]/t[0].sum(), "no var:", t[1,0]/t[1].sum(), "OR p", fisher_exact(t)[1])
print("MH pooled OR", mh_or(tabs))
# crowded: bright neighbour, off LMC
s = src[~src.near_lmc]
t = pd.crosstab(s.n_nb_brighter > 0, s.periodic).reindex(index=[True, False], columns=[True, False]).values
print("off-LMC bright nb: periodic", t[0,0]/t[0].sum(), "no bright nb", t[1,0]/t[1].sum(), "fisher p", fisher_exact(t)[1])
# among crowded class, how many qualify only via bright nb, and expected
cr = p[p.classification == "contamination_candidate_crowded"]
print("crowded via LMC", cr.near_lmc.sum(), "via bright nb only", (~cr.near_lmc).sum())
# neighbour classes that can make a few-day signal
kv = p[p.classification == "contamination_known_variable"]
short = ["ECL", "CEP", "RR", "DSCT", "RS", "SOLAR_LIKE", "ACV", "S"]
def cls_ok(s):
    if pd.isna(s): return False
    return any(any(k in tok for k in short) for tok in s.split(";"))
kv_ok = kv.nb_var_classes.map(cls_ok)
print("known_var with a short-period-capable neighbour class", kv_ok.sum(), "of", len(kv), "; only LPV/BE/none:", (~kv_ok).sum())
print("  classes of the 'other' group:", kv[~kv_ok].nb_var_classes.value_counts(dropna=False).to_dict())
print("  their P_ref median", kv[~kv_ok].P_ref.median(), "near LMC", kv[~kv_ok].near_lmc.sum())
print("tess_timescale (13.7/27.4/1 d within 3%) among periodic by class", p[p.tess_timescale].classification.value_counts().to_dict())
print("sys common mode members:", p[p.classification == "systematic_common_mode"][["tic", "ra", "dec", "P_ref", "cm_retained"]].round(3).to_string())
