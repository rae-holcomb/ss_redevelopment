"""Count red-clump-like stars bright enough for TESS FFIs in Gaia DR3 (ESA Gaia archive TAP), P27.

Selection follows Claytor & Tayar (2025): -0.5 < M_G < 1.5 (no extinction correction), plus a colour cut
0.9 < BP-RP < 1.6 to drop hot horizontal-branch / blue stars, parallax_over_error > 10, G < Gmax.
Writes reports/2026-11-12_validation-set-scoping/data/gaia_rc_counts.csv.
"""
import io
import os
import sys
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://gea.esac.esa.int/tap-server/tap/sync"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "2026-11-12_validation-set-scoping",
                   "data", "gaia_rc_counts.csv")


def count(where: str) -> int:
    """Return COUNT(*) of gaiadr3.gaia_source rows matching ``where`` (synchronous ADQL)."""
    q = f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE {where}"
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=600) as r:
        return int(pd.read_csv(io.BytesIO(r.read()))["n"][0])


if __name__ == "__main__":
    rc = ("parallax_over_error > 10 AND bp_rp BETWEEN 0.9 AND 1.6 AND "
          "phot_g_mean_mag + 5*LOG10(parallax) - 10 BETWEEN -0.5 AND 1.5")
    rows = []
    for gmax in (11, 12, 13):
        rows.append(dict(region="all-sky", gmax=gmax, n=count(f"{rc} AND phot_g_mean_mag < {gmax}")))
        rows.append(dict(region="SCVZ (ecl_lat<-78)", gmax=gmax,
                         n=count(f"{rc} AND phot_g_mean_mag < {gmax} AND ecl_lat < -78")))
        print(rows[-2], rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows))
    sys.exit(0)
