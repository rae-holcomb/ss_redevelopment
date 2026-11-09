"""Fallback for c05: Gaia DR3 variability periods for the VARIABLE neighbours of periodic noise sources, from the
ESA Gaia archive (one ADQL query per table on source_id). Run: codeastro python -I c05b_gaia_esa_fetch.py <outdir>"""
import sys, pandas as pd
from astroquery.gaia import Gaia
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
out = sys.argv[1]
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
gx = pd.read_csv("/Volumes/Data_Vault/smarts_data/extracted/tic_lookup/gaia_xmatch.csv")
v = gx[(gx.VarFlag == "VARIABLE") & gx.tic.isin(S.noise_tic[S.periodic]) & (gx.sep_arcsec >= 3)]
ids = ",".join(map(str, sorted(set(v.Source))))
print(len(set(v.Source)), "ids", flush=True)
tabs = {"vari_eclipsing_binary": "frequency", "vari_cepheid": "pf", "vari_rrlyrae": "pf",
        "vari_short_timescale": "frequency", "vari_ms_oscillator": "frequency1", "vari_rotation_modulation": "best_rotation_period",
        "vari_long_period_variable": "frequency"}
for t, col in tabs.items():
    try:
        j = Gaia.launch_job(f"SELECT source_id, {col} FROM gaiadr3.{t} WHERE source_id IN ({ids})")
        d = j.get_results().to_pandas(); d.to_csv(f"{out}/esa_{t}.csv", index=False); print(t, len(d), flush=True)
    except Exception as e:
        print(t, "ERR", str(e)[:200], flush=True)
