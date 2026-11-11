"""Look up TIC v8.2 rows for a list of TIC IDs via the MAST Catalogs API (P27).

Usage: python tools/validation_set/tic_lookup.py <ids.txt> <out.csv>
"""
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import vo  # noqa: E402

COLS = ("ID,ra,dec,Tmag,Teff,logg,MH,rad,mass,lum,GAIA,plx,e_plx,GAIAmag,gaiabp,gaiarp,"
        "lumclass,objType,d,ebv,contratio,numcont,disposition,wdflag,KIC,HIP,Jmag,Kmag")


def tic_rows(ids, chunk: int = 500) -> pd.DataFrame:
    """Return TIC rows (columns ``COLS``) for an iterable of TIC IDs, querying MAST in chunks."""
    ids = [str(int(i)) for i in ids]
    frames = []
    for i in range(0, len(ids), chunk):
        js = vo.mast_invoke({"service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                             "params": {"columns": COLS,
                                        "filters": [{"paramName": "ID", "values": ids[i:i + chunk]}]},
                             "pagesize": 5000, "page": 1})
        frames.append(pd.DataFrame(js["data"]))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    ids = [line.strip() for line in open(src) if line.strip()]
    df = tic_rows(ids)
    df.to_csv(out, index=False)
    print(len(ids), "ids ->", len(df), "rows ->", out)
