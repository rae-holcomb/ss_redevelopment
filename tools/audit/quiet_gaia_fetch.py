"""Fetch Gaia DR3 variability periods for the VARIABLE neighbours (3-63") of quiet-class noise sources.

Mirrors the 2026-11-09 note review's ``checks/c05b_gaia_esa_fetch.py`` (which covered only
periodic sources): one ADQL query per DR3 variability table on ``source_id``, against the public
ESA Gaia archive. Needs astroquery (not in ss_openblas; run with the ``codeastro`` env, ``python -I``).

Usage::

    python -I tools/audit/quiet_gaia_fetch.py <sources.csv> <gaia_xmatch.csv> <outdir>
"""
import sys

import pandas as pd
from astroquery.gaia import Gaia

TABLES = {"vari_eclipsing_binary": "frequency", "vari_cepheid": "pf", "vari_rrlyrae": "pf",
          "vari_short_timescale": "frequency", "vari_ms_oscillator": "frequency1",
          "vari_rotation_modulation": "best_rotation_period", "vari_long_period_variable": "frequency"}


def main():
    """Query each table for the quiet sources' variable neighbours and write ``esa_<table>.csv``."""
    src, gxf, out = sys.argv[1:4]
    s = pd.read_csv(src)
    gx = pd.read_csv(gxf)
    v = gx[(gx.VarFlag == "VARIABLE") & gx.tic.isin(s.tic[s.classification == "quiet"]) & (gx.sep_arcsec >= 3)]
    ids = ",".join(map(str, sorted(set(v.Source))))
    print(len(set(v.Source)), "ids", flush=True)
    for t, col in TABLES.items():
        j = Gaia.launch_job_async(f"SELECT source_id, {col} FROM gaiadr3.{t} WHERE source_id IN ({ids})")
        d = j.get_results().to_pandas()
        d.to_csv(f"{out}/esa_{t}.csv", index=False)
        print(t, len(d), flush=True)


if __name__ == "__main__":
    main()
