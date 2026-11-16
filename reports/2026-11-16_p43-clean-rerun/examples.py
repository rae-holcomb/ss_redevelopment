"""Compute the example-star data for the README figures (light: one refit per star and variant).

For each example star and mode: the light curve, the Stage-2 ACF, and for each listed variant the candidate
it shows (the selected one, or, if the star was declined / answered something else, the success candidate
with the fewest failed gates) with its per-tooth table (model centre, tolerance, free vertex, fitted parabola)
and, for trimmed teeth, their windows. Saved to data/examples.pkl; analysis.py draws the figures from it.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p43lib as L  # noqa: E402

C, ev = L.C, L.ev
S = {"correct", "diffrot"}

# (mode, star suffix, group): recovered by the recommended variant, still lost, newly wrong
EXAMPLES = [
    ("fullyear", "008023", "recovered"), ("fullyear", "015438", "recovered"), ("fullyear", "103059", "recovered"),
    ("fullyear", "188364", "recovered"), ("assigned", "001519", "recovered"), ("fullyear", "039633", "recovered"),
    ("fullyear", "130951", "lost"), ("fullyear", "273135", "lost"), ("fullyear", "344069", "lost"),
    ("fullyear", "125098", "lost"),
    ("fullyear", "136454", "newwrong"), ("fullyear", "009011", "newwrong"), ("fullyear", "912897", "newwrong"),
    ("assigned", "316783", "newwrong"),
]
VARIANTS = ("local_tol", "local_tol_trim")


def _pick(res, true_p, prange, seed_P0=None):
    """The candidate to show: the selected one if the fit succeeded, else the closest success candidate."""
    cands = res.candidates
    if seed_P0 is not None:
        same = [c for c in cands if np.isclose(c.period, seed_P0)]
        if same:
            return same[0]
    if res.success:
        return cands[0]
    ok = [c for c in cands if ev.classify_period(c.fit.P, true_p, ev.REL_TOL, prange) in S]
    if not ok:
        return cands[0] if cands else None
    return min(ok, key=lambda c: (len(c.failed_gates), c.fit.redchi))


def _cand_record(st, cand, spec, outcome_star):
    """Plot-ready dict for one candidate."""
    fit = cand.fit
    teeth = L.tooth_table(cand, spec)
    for t in teeth:
        w = [w for w in fit.windows if w.n == t["n"]][0]
        t["lag_lo"], t["lag_hi"] = float(w.lag_lo), float(w.lag_hi)
        t["center"] = float(fit.per_peak[t["n"]]["center"])
    used = {w.n for w in fit.windows}
    return dict(P0=float(cand.period), P=float(fit.P), passed=bool(cand.passed_gates),
                failed=list(cand.failed_gates), outcome=ev.classify_period(fit.P, st["true_p"], ev.REL_TOL, st["prange"]),
                star_outcome=outcome_star, teeth=teeth, n_trimmed=int(fit.n_peaks_trimmed),
                n_dropped=int(fit.n_peaks_dropped), used=sorted(used),
                gates={k: (v.get("value"), v.get("passed")) for k, v in (cand.gate_results or {}).items()
                       if v.get("enabled", True)})


def main():
    warnings.simplefilter("ignore")
    out = []
    tasks = {m: {t["star_id"]: t for t in L.make_tasks(m)} for m in ("fullyear", "assigned")}
    for mode, suf, group in EXAMPLES:
        sid = [k for k in tasks[mode] if k.endswith(suf)][0]
        st = L.load_star(tasks[mode][sid])
        pooled = L.cached_guesses(st, L.CACHE_ROOT / f"dev_medium_clean_{mode}")
        rec = dict(mode=mode, star_id=sid, group=group, true_p=float(st["true_p"]), prange=st["prange"],
                   t=st["t"].astype(np.float32), f=st["f"].astype(np.float32),
                   lags=st["lags"].astype(np.float32), acf=st["acf"].astype(np.float32), variants={})
        res = {v: C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
               for v in VARIANTS}
        # show the same seed candidate in both panels: the one the "interesting" variant shows
        lead = "local_tol_trim" if group in ("recovered", "newwrong") else "local_tol"
        outc = {v: ev.classify_period(res[v].best_fit.P if res[v].success else None, st["true_p"], ev.REL_TOL,
                                      st["prange"]) for v in VARIANTS}
        c_lead = _pick(res[lead], st["true_p"], st["prange"])
        for v in VARIANTS:
            c = c_lead if v == lead else _pick(res[v], st["true_p"], st["prange"], seed_P0=c_lead.period)
            rec["variants"][v] = _cand_record(st, c, C.COMB_VARIANTS[v], outc[v])
        out.append(rec)
        print(mode, suf, group, {v: outc[v] for v in VARIANTS})
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data/examples.pkl").write_bytes(pickle.dumps(out))


if __name__ == "__main__":
    main()
