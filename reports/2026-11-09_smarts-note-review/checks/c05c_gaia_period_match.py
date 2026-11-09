"""Does a variable neighbour's own Gaia DR3 period match the noise source's P_ref (within 3% at P, P/2 or 2P)?
Null: P_ref shuffled among the same sources (500 permutations). Inputs: c05b outputs in <dir>."""
import sys, glob, numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
d = sys.argv[1]
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
gx = pd.read_csv("/Volumes/Data_Vault/smarts_data/extracted/tic_lookup/gaia_xmatch.csv")
conv = {"vari_eclipsing_binary": ("frequency", True), "vari_cepheid": ("pf", False), "vari_rrlyrae": ("pf", False),
        "vari_short_timescale": ("frequency", True), "vari_ms_oscillator": ("frequency1", True),
        "vari_rotation_modulation": ("best_rotation_period", False), "vari_long_period_variable": ("frequency", True)}
rows = []
for t, (col, isf) in conv.items():
    x = pd.read_csv(f"{d}/esa_{t}.csv"); x.columns = [c.lower() for c in x.columns]; x = x.dropna(subset=[col])
    rows.append(pd.DataFrame(dict(Source=x.source_id, P_nb=(1 / x[col]) if isf else x[col], table=t)))
nb = pd.concat(rows)
nb = nb.merge(gx[["Source", "tic", "sep_arcsec", "Gmag"]], on="Source")
p = S[S.periodic].set_index("noise_tic")
nb = nb[nb.tic.isin(p.index)]
def hits(Pref):
    """Per TIC: any neighbour period within 3% of Pref, Pref/2 or 2 Pref."""
    pr = nb.tic.map(Pref)
    r = nb.P_nb / pr
    ok = (np.abs(r - 1) < .03) | (np.abs(r - .5) < .015) | (np.abs(r - 2) < .06)
    return nb.tic[ok].unique()
has = nb.tic.unique()
print("periodic sources with >=1 neighbour Gaia period:", len(has), "| by class:", p.loc[has].classification.value_counts().to_dict())
h = hits(p.P_ref_d)
print("period match:", len(h), "| by class:", p.loc[h].classification.value_counts().to_dict())
print("matched tables:", nb[nb.tic.isin(h)].assign(r=lambda z: z.P_nb / z.tic.map(p.P_ref_d)).query(
      "abs(r-1)<.03 or abs(r-.5)<.015 or abs(r-2)<.06").table.value_counts().to_dict())
rng = np.random.default_rng(0)
sub = p.loc[has]
null = [len(hits(pd.Series(rng.permutation(sub.P_ref_d.values), index=sub.index))) for _ in range(500)]
print("null (P_ref shuffled among these sources): mean", np.mean(null), "95th", np.percentile(null, 95), "max", max(null))
kv = p[p.classification == "contamination_known_variable"]
print("known_variable: with Gaia neighbour period", int(kv.index.isin(has).sum()), "of", len(kv), "; matched", int(kv.index.isin(h).sum()))
print("matched known_var FFI:", kv[kv.index.isin(h)].ffi_check.value_counts().to_dict())
print("unmatched-but-has-period known_var P_ref median:", kv[kv.index.isin(has) & ~kv.index.isin(h)].P_ref_d.median())
