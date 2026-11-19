"""Count Gaia DR3 variability-table rows (contaminant classes for P27) via the ESA Gaia archive TAP.

Writes reports/2026-11-12_validation-set-scoping/data/gaia_vari_counts.csv.
"""
import io
import os
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://gea.esac.esa.int/tap-server/tap/sync"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "2026-11-12_validation-set-scoping",
                   "data", "gaia_vari_counts.csv")
TABLES = ["vari_summary", "vari_classifier_result", "vari_rotation_modulation", "vari_eclipsing_binary",
          "vari_ms_oscillator", "vari_rrlyrae", "vari_cepheid", "vari_long_period_variable",
          "vari_short_timescale"]


def adql(q: str) -> pd.DataFrame:
    """Run a synchronous ADQL query on the Gaia archive and return a DataFrame."""
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=600) as r:
        return pd.read_csv(io.BytesIO(r.read()))


if __name__ == "__main__":
    rows = [dict(table=t, n=int(adql(f"SELECT COUNT(*) AS n FROM gaiadr3.{t}")["n"][0])) for t in TABLES]
    cls = adql("SELECT best_class_name, COUNT(*) AS n FROM gaiadr3.vari_classifier_result GROUP BY best_class_name")
    cls = cls.rename(columns={"best_class_name": "table"}).assign(table=lambda d: "class:" + d["table"])
    out = pd.concat([pd.DataFrame(rows), cls], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(out.to_string())
