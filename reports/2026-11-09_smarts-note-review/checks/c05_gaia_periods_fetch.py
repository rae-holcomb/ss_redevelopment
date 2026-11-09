"""Fetch Gaia DR3 variability periods (VizieR I/358 per-class tables) within 63" of each periodic noise source,
by batched cone search. Run: /opt/anaconda3/envs/codeastro/bin/python -I c05_gaia_periods_fetch.py <outdir>"""
import sys, pandas as pd, astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
out = sys.argv[1]
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
p = S[S.periodic].reset_index(drop=True)
c = SkyCoord(p.ra_deg.values * u.deg, p.dec_deg.values * u.deg)
Vizier.VIZIER_SERVER = "vizier.cds.unistra.fr"
V = Vizier(columns=["**"], row_limit=-1, timeout=120); V.VIZIER_SERVER = "vizier.cds.unistra.fr"
for tab in ["veb", "vcep", "vrrlyr", "vst", "vmsosc", "vrm", "vlpv"]:
    rows = []
    for i0 in range(0, len(p), 50):
        try:
            res = V.query_region(c[i0:i0 + 50], radius=63 * u.arcsec, catalog=f"I/358/{tab}")
        except Exception as e:
            print(tab, i0, "ERR", str(e)[:120], flush=True); continue
        if res is not None and len(res):
            d = res[0].to_pandas(); d["noise_tic"] = p.noise_tic.values[i0 + d["_q"].values - 1]; rows.append(d)
    d = pd.concat(rows) if rows else pd.DataFrame()
    d.to_csv(f"{out}/gaia_{tab}.csv", index=False)
    print(tab, len(d), [x for x in d.columns if any(k in x.lower() for k in ("per", "freq", "pf", "p1"))][:8], flush=True)
