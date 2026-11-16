"""Extract the small tables the figures need from the harness pickles (runs/p43_clean/harness/, gitignored).

Writes to data/:
  cands_{mode}.csv      every fitted candidate of local_tol / local_tol_trim / local_tol_cs_trim (all stars)
  lost_{mode}.csv       one row per star _v7 got right and local_tol does not: the closest success candidate
                        under local_tol (fewest failed gates, then lowest redchi), its failed gates, how many
                        windows outlier rejection dropped, and the same for the lsq fit of the candidate _v7
                        selected; plus every variant's outcome for that star.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p43lib as L  # noqa: E402

S = {"correct", "diffrot"}
KEEP = ("local_tol", "local_tol_trim", "local_tol_cs_trim")


def main():
    for mode in ("fullyear", "assigned"):
        recs = pickle.load(open(L.ROOT / f"runs/p43_clean/harness/all_{mode}.pkl", "rb"))
        frames = []
        for v in KEEP:
            df = L.candidate_frame(recs, v).drop(columns=["teeth"])
            df.insert(0, "variant", v)
            frames.append(df)
        pd.concat(frames).to_csv(HERE / f"data/cands_{mode}.csv.gz", index=False)

        changed = {r["star_id"]: r for r in pickle.load(open(L.ROOT / f"runs/p43_clean/harness/changed_{mode}.pkl", "rb"))}
        rows = []
        for r in recs:
            vv = r["variants"]
            if not (vv["local_screen2"]["final_outcome"] in S and vv["local_tol"]["final_outcome"] not in S):
                continue
            ok = [c for c in vv["local_tol"]["candidates"] if c["outcome"] in S]
            best = min(ok, key=lambda c: (len(c["failed"].split(";")) if c["failed"] else 0, c["redchi"])) if ok else None
            lsq = [c for c in changed[r["star_id"]]["variants"]["lsq"]["candidates"] if c["selected"]]
            row = dict(star_id=r["star_id"], true_period=r["true_period"], period_bin=r["period_bin"],
                       tol_outcome=vv["local_tol"]["final_outcome"],
                       best_P=best["P"] if best else np.nan, best_failed=best["failed"] if best else "no_success_candidate",
                       best_n_dropped=best["n_dropped"] if best else np.nan,
                       best_n_used=best["n_peaks_used"] if best else np.nan,
                       best_snr=best.get("v_mean_height_snr", np.nan) if best else np.nan,
                       best_fvt=best.get("v_frac_vertex_tol", np.nan) if best else np.nan,
                       lsq_n_dropped=lsq[0]["n_dropped"] if lsq else np.nan,
                       lsq_snr=lsq[0].get("v_mean_height_snr", np.nan) if lsq else np.nan)
            for v in vv:
                row[f"out_{v}"] = vv[v]["final_outcome"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(HERE / f"data/lost_{mode}.csv", index=False)
        print(mode, len(rows), "lost stars")


if __name__ == "__main__":
    main()
