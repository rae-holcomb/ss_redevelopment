"""T32: candidate-level vertex statistics under local / local_track / local_quad for a manifest (threshold study).

Usage: python cand_stats.py <manifest> <mode> <out.csv>
Writes one row per fitted candidate with its outcome, whether it passes every
gate except the vertex gate ("other_ok"), and the T32 statistics.
"""
import sys, pickle, warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t32lib as L
import numpy as np, pandas as pd

SPECS = {"sqrt": dict(tol_growth="sqrt", tol_base=0.05), "lin": dict(tol_growth="linear", tol_base=0.05),
         "width": dict(tol_growth="width", tol_k=0.5)}


def one(task, cache):
    """Candidate rows for one star under three solver variants."""
    warnings.simplefilter("ignore")
    st = L.load_star(task)
    pooled = pickle.loads((Path(cache) / f"{st['star_id']}.pkl").read_bytes())
    rows = []
    for var in ("local", "local_track", "local_quad"):
        r = L.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=var)
        for c in r.candidates:
            fg = [g for g in c.failed_gates if g != "frac_vertex_in_window"]
            row = dict(star_id=st["star_id"], variant=var, P=c.fit.P, true_period=st["true_p"],
                       outcome=L.ev.classify_period(c.fit.P, st["true_p"], L.ev.REL_TOL, st["prange"]),
                       other_ok=(len(fg) == 0), fviw=c.diagnostics["frac_vertex_in_window"],
                       redchi=c.fit.redchi, n_teeth=len(c.fit.per_peak))
            for k, sp in SPECS.items():
                vs = L.C._vertex_stats(c.fit, dict(sp, min_free=0.0))
                row[f"tol_{k}"] = vs["frac_vertex_tol"]
            vs = L.C._vertex_stats(c.fit, dict(min_free=0.0))
            row.update(scatter=vs["vertex_scatter"], frac_free=vs["frac_free_vertices"], drift_a_frac=vs["drift_a_frac"])
            rows.append(row)
    return rows


if __name__ == "__main__":
    man, mode, out = sys.argv[1:4]
    tasks = L.make_tasks(mode, man)
    cache = L.ROOT / "runs/t32/stage1_cache" / (Path(man).stem + "_" + mode)
    with ProcessPoolExecutor(3, initializer=L.determinism.init_worker) as ex:
        rows = [r for rr in ex.map(partial(one, cache=str(cache)), tasks, chunksize=4) for r in rr]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows))
