"""Quick screen: outcome of every P43 variant on local_tol's T32 lost stars (+ T32 gained stars, to watch for regressions)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p43lib as P  # noqa: E402
import pandas as pd  # noqa: E402

VS = sys.argv[1].split(",")
ch = pd.read_csv(P.ROOT / "reports/2026-11-13_t32-local-drift/data/changed_stars_local_tol.csv", dtype={"star_id": str})
ch = ch[ch.change.isin(["lost", "gained"])]
rows = []
for _, row in ch.iterrows():
    st, pooled = P.star_and_guesses(row.star_id, row["mode"])
    rec = dict(star=row.star_id[-6:], mode=row["mode"][:2], t32=row.change, Pt=round(st["true_p"], 2))
    for v in VS:
        res = P.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
        rec[v] = P.classify(res.best_fit.P, st) if res.success else "none"
    rows.append(rec)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string())
for v in VS:
    print(v, df.groupby("t32")[v].apply(lambda s: s.isin(["correct", "diffrot"]).sum()).to_dict(),
          "wrong:", int((df[v] == "wrong").sum()))
