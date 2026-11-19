"""Step-0 diagnosis: for local_tol's lost stars (T32 list), print the success-candidate's gates and teeth."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p43lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

lost = pd.read_csv(P.ROOT / "reports/2026-11-13_t32-local-drift/data/changed_stars_local_tol.csv", dtype={"star_id": str})
lost = lost[lost.change == "lost"]
sel = sys.argv[1:] or None
spec = P.C.COMB_VARIANTS["local_tol"]
for _, row in lost.iterrows():
    if sel and row.star_id[-6:] not in sel:
        continue
    st, pooled = P.star_and_guesses(row.star_id, row["mode"])
    res = P.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant="local_tol")
    print(f"\n#### {row.star_id[-6:]} {row['mode']} Ptrue={st['true_p']:.3f} final={res.best_fit.P if res.success else None}")
    for r in res.candidates:
        if r.fit is None:
            continue
        oc = P.classify(r.fit.P, st)
        if oc not in ("correct", "diffrot"):
            continue
        g = r.gate_results
        print(f"  cand P0={r.period:.3f} P={r.fit.P:.3f} {oc} failed={r.failed_gates} trimmed={r.fit.n_peaks_trimmed}"
              f" fvt={g['frac_vertex_tol']['value']:.2f} fph={g['frac_positive_heights']['value']:.2f}"
              f" snr={g['mean_height_snr']['value']:.2f} hl_min={g['half_lag_consistency']['value']:.3f}"
              f" hl_c={g['half_lag_consistency']['contrast']:.3f}")
        for t in P.tooth_table(r, spec):
            print(f"     n={t['n']} m={t['m']:.3f} v={t['v']:.3f} h={t['h']:+.3f} hfree={t['h_free']:+.3f}"
                  f" tol={t['tol']:.3f} {'L0' if t['lag0'] else ''} {'ok' if t['within'] else '--'}")
        break
