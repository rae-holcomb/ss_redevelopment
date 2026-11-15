"""Step-0 diagnosis, compact: every success candidate of local_tol's lost stars, one line each."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p43lib as P  # noqa: E402
import pandas as pd  # noqa: E402

lost = pd.read_csv(P.ROOT / "reports/2026-11-13_t32-local-drift/data/changed_stars_local_tol.csv", dtype={"star_id": str})
lost = lost[lost.change == "lost"]
rows = []
for _, row in lost.iterrows():
    st, pooled = P.star_and_guesses(row.star_id, row["mode"])
    res = P.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant="local_tol")
    fin = P.classify(res.best_fit.P, st) if res.success else "none"
    for r in res.candidates:
        if r.fit is None or P.classify(r.fit.P, st) not in ("correct", "diffrot"):
            continue
        g = r.gate_results
        tt = [t for t in P.tooth_table(r, P.C.COMB_VARIANTS["local_tol"]) if not t["lag0"]]
        rows.append(dict(star=row.star_id[-6:], mode=row["mode"][:2], Pt=round(st["true_p"], 2), fin=fin,
                         P=round(r.fit.P, 2), failed=",".join(x.replace("frac_", "f_").replace("half_lag_consistency", "HL")
                                                         .replace("mean_height_snr", "snr") for x in r.failed_gates),
                         nwin=r.fit.n_peaks_used, trim=r.fit.n_peaks_trimmed,
                         fvt=round(g["frac_vertex_tol"]["value"], 2),
                         fvt_pos=round(sum(t["within"] for t in tt if t["h"] > 0) / max(1, sum(t["h"] > 0 for t in tt)), 2),
                         fph=round(g["frac_positive_heights"]["value"], 2), snr=round(g["mean_height_snr"]["value"], 2),
                         hl=round(g["half_lag_consistency"]["value"], 3),
                         hs="".join("+" if t["h"] > 0 else "-" for t in tt),
                         vs="".join("v" if t["within"] else "." for t in tt)))
pd.set_option("display.width", 250)
df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv(Path(__file__).resolve().parent / "data" / "diag_lost_candidates.csv", index=False)
