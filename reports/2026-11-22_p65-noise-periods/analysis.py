"""P65 analysis: secondary coherent signals in the noise stacks, and the 22 periodic_low_amplitude dev stars.

Inputs (all local; no drive reads):
- data/stack_peaks.csv, data/query_snr.csv: from tools/audit/noise_secondary_peaks.py (see README "Reproduce");
- data/source_flags.csv: per-source T20 flags (classification, P_ref, tsplit_min_snr), cut from the drive's
  noise_source_flags.csv.gz on 2026-11-22;
- reports/2026-11-21_ranker-v1/data/loss_categories.csv: v1's per-star category at the p = 0.86 pick;
- <main>/runs/ranker_v1/nohalf/dev_top1_*.csv and thresholds.json (ranker v1 per-star answers);
- <main>/runs/v9_dm{fy,as}/per_star.csv (hand _v9 per-star answers on dev_medium_clean).

Writes data/summary.json, data/rates.csv, data/wrong_explained.csv, data/secondary_sources.csv,
data/null_snr.npy and prints the tables quoted in the README.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MAIN = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment"
V1 = os.path.join(MAIN, "runs/ranker_v1/nohalf")
HAND = {"fullyear": os.path.join(MAIN, "runs/v9_dmfy/per_star.csv"),
        "assigned": os.path.join(MAIN, "runs/v9_dmas/per_star.csv")}
SUC, ALI = {"correct", "diffrot"}, {"half", "double"}
SNR_COH = 10.0     # min-half local S/N for a "coherent" peak (chance level, see README: ~2e-3 per line search)
SNR_STRICT = 20.0  # the periodic_low_amplitude rule's threshold
TOL = 0.02         # period match tolerance for sharp lines (P, P/2 or 2P)
KEPT = ("quiet", "systematic_common_mode")   # classes kept in training
N_NULL = 50
SNR_WRONG = 5.0    # min-half S/N at one given period (no look-elsewhere): shuffled-source null rate ~0.3%


def same_period(p, q, tol=TOL):
    """p within tol of q, q/2 or 2q (relative); as in loss_categories.py."""
    r = np.asarray(p, float) / np.asarray(q, float)
    return np.isfinite(r) & ((abs(r - 1) < tol) | (abs(r - 0.5) < tol / 2) | (abs(r - 2) < 2 * tol))


def load_peaks():
    """Stack peaks joined to the source flags; ``secondary`` = not the P_ref line or its 1/2, 2x."""
    fl = pd.read_csv(os.path.join(DATA, "source_flags.csv")).set_index("tic")
    pk = pd.read_csv(os.path.join(DATA, "stack_peaks.csv")).join(fl, on="tic")
    pk["is_pref"] = same_period(pk.P, pk.P_ref)
    pk["coherent"] = pk.min_half_snr >= SNR_COH
    return pk, fl


def rates(thr: dict) -> pd.DataFrame:
    """Hand _v9 and v1 success / wrong / alias rates on dev_medium_clean, with and without the 22."""
    lc = pd.read_csv(os.path.join(HERE, "../2026-11-21_ranker-v1/data/loss_categories.csv"))
    pla = set(lc[(lc.set == "dev_medium") & (lc.noise_classification == "periodic_low_amplitude")].star_id)
    rows = []
    for mode in ("fullyear", "assigned"):
        h = pd.read_csv(HAND[mode], usecols=["star_id", "final_outcome"])
        v = pd.read_csv(os.path.join(V1, f"dev_top1_dev_medium_{mode}.csv"))
        v = v[v.is_rotator]
        ans = v.score_v1 >= v.top_bin.map(thr).astype(float)
        v["outcome"] = np.where(ans, v.top_outcome, "none")
        for who, d, col in (("hand_v9", h, "final_outcome"), ("ranker_v1", v, "outcome")):
            for sub, m in (("all 874", np.ones(len(d), bool)), ("without the 22", ~d.star_id.isin(pla)),
                           ("only the 22", d.star_id.isin(pla).values)):
                o = d.loc[m, col]
                rows.append(dict(mode=mode, method=who, subset=sub, n=int(m.sum()),
                                 success=o.isin(SUC).mean(), wrong=(o == "wrong").mean(), alias=o.isin(ALI).mean(),
                                 n_success=int(o.isin(SUC).sum()), n_wrong=int((o == "wrong").sum()),
                                 n_alias=int(o.isin(ALI).sum())))
    return pd.DataFrame(rows), pla


def noise_scores(pk: pd.DataFrame) -> tuple[pd.Series, callable]:
    """S/N of a noise stack at a period: max of the query at P, P/2, 2P and any listed stack peak within 2%."""
    q = pd.read_csv(os.path.join(DATA, "query_snr.csv"))
    q_snr = q.groupby("key").min_half_snr.max()
    q_tic = q.groupby("key").tic.first()
    by_tic = {t: g[["P", "min_half_snr"]] for t, g in pk.groupby("tic")}

    def score(key, tic, P):
        s = q_snr.get(key, np.nan)
        g = by_tic.get(tic)
        if g is not None:
            m = g[same_period(P, g.P.values)]
            if len(m):
                s = np.nanmax([s, m.min_half_snr.max()])
        return s
    return q_tic, score


def explain_wrong(lc: pd.DataFrame, pk: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Noise-stack S/N at each star's top-1 period; and at the same period in 50 random stacks (the null)."""
    q_tic, score = noise_scores(pk)
    lc = lc.copy()
    lc["key"] = lc.set + "|" + lc["mode"] + "|" + lc.star_id
    lc["noise_snr_at_top"] = [score(k, t, P) for k, t, P in zip(lc.key, lc.noise_tic, lc.top_period)]
    null = []
    for k, P in zip(lc.key[lc.category == "answered_wrong"], lc.top_period[lc.category == "answered_wrong"]):
        null.append([score(f"null{i}|{k}", q_tic[f"null{i}|{k}"], P) for i in range(N_NULL)])
    return lc, np.array(null)


def build_query(out: str, stack_index: str) -> None:
    """The --query CSV for noise_secondary_peaks.py: every dev star-table's top-1 in its own stack, plus 50
    random other stacks per answered-wrong row (seed 0), keyed ``null{k}|<set>|<mode>|<star_id>``."""
    d = pd.read_csv(os.path.join(HERE, "../2026-11-21_ranker-v1/data/loss_categories.csv"))
    d["key"] = d.set + "|" + d["mode"] + "|" + d.star_id
    q = d[["key", "noise_tic", "top_period"]].rename(columns={"noise_tic": "tic", "top_period": "P"}).dropna()
    tics = pd.read_csv(stack_index).tic
    rng = np.random.default_rng(0)
    nulls = [dict(key=f"null{k}|" + r.key, tic=t, P=r.top_period)
             for _, r in d[d.category == "answered_wrong"].iterrows()
             for k, t in enumerate(rng.choice(tics[tics != r.noise_tic].values, N_NULL, replace=False))]
    pd.concat([q, pd.DataFrame(nulls)]).to_csv(out, index=False)


def main():
    """Entry point (``--build-query OUT STACK_INDEX`` writes the query CSV instead)."""
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "--build-query":
        return build_query(sys.argv[2], sys.argv[3])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    thr = json.load(open(os.path.join(V1, "thresholds.json")))["rules"]["head_perbin_p0.86"]
    summ: dict = {}

    # ---- (b) rates with and without the 22 periodic_low_amplitude stars
    rt, pla = rates(thr)
    rt.to_csv(os.path.join(DATA, "rates.csv"), index=False)
    print(rt.round(4).to_string())
    summ["n_periodic_low_amplitude_dev_medium"] = len(pla)

    # ---- (a) population: coherent secondary lines in the stacks
    pk, fl = load_peaks()
    lines = pk[pk.kind == "line"]
    sec = pk[pk.coherent & ~pk.is_pref].sort_values("min_half_snr", ascending=False).drop_duplicates("tic")
    sec = sec.rename(columns={"P": "P_secondary", "min_half_snr": "snr_secondary"})
    best_line = lines.groupby("tic").min_half_snr.max()
    pop = []
    for cls, g in fl.groupby("classification"):
        s = sec[sec.classification == cls]
        pop.append(dict(classification=cls, n_sources=len(g),
                        secondary_ge10=int((s.snr_secondary >= SNR_COH).sum()),
                        secondary_ge20=int((s.snr_secondary >= SNR_STRICT).sum()),
                        best_line_median_snr=float(best_line.reindex(g.index).median())))
    pop = pd.DataFrame(pop)
    print(pop.to_string())
    summ["population"] = pop.to_dict("records")
    sec_kept = sec[sec.classification.isin(KEPT)]
    sec_kept[["tic", "classification", "P_ref", "tsplit_min_snr", "P_secondary", "snr_secondary", "snr_h1",
              "snr_h2", "snr_full", "kind"]].to_csv(os.path.join(DATA, "secondary_sources.csv"), index=False)
    # Is the secondary line shared between sources? (cluster at 0.5% in log P)
    lp = np.log(sec_kept.P_secondary.values)
    shared = np.array([(np.abs(lp - x) < 0.005).sum() - 1 for x in lp])
    summ["kept_sources_with_secondary_ge10"] = int(len(sec_kept))
    summ["kept_sources_with_secondary_ge20"] = int((sec_kept.snr_secondary >= SNR_STRICT).sum())
    summ["kept_secondary_sharing_period_with_another_source"] = int((shared > 0).sum())
    summ["kept_secondary_P_quantiles"] = sec_kept.P_secondary.quantile([0.1, 0.5, 0.9]).round(3).tolist()

    # ---- wrong answers vs the noise stack's S/N at the reported period
    lc = pd.read_csv(os.path.join(HERE, "../2026-11-21_ranker-v1/data/loss_categories.csv"))
    ex, null = explain_wrong(lc, pk)
    np.save(os.path.join(DATA, "null_snr.npy"), null)
    w = ex[ex.category == "answered_wrong"].copy()
    is_pref = same_period(w.top_period, w.noise_P_ref, 0.05)
    coh = w.noise_snr_at_top >= SNR_WRONG
    w["why"] = np.select([coh & is_pref, coh, is_pref],
                         ["coherent noise signal at P_ref", "coherent secondary noise signal",
                          "P_ref, not coherent (red-noise peak)"], "no noise-period match")
    ex.loc[w.index, "why"] = w.why
    ex.to_csv(os.path.join(DATA, "wrong_explained.csv"), index=False)
    tab = pd.crosstab(w.why, w.noise_classification, margins=True)
    print(tab)
    summ["wrong_n"] = len(w)
    summ["wrong_why"] = w.why.value_counts().to_dict()
    summ["wrong_why_by_class"] = {k: v for k, v in tab.to_dict().items()}
    for th in (5, 10, 20):
        summ[f"wrong_snr_ge{th}_observed"] = int((w.noise_snr_at_top >= th).sum())
        summ[f"wrong_snr_ge{th}_null_expected"] = float((null >= th).mean() * len(w))
        summ[f"wrong_snr_ge{th}_null_max_in_draws"] = int((null >= th).sum(axis=0).max())
    print({k: v for k, v in summ.items() if k.startswith("wrong_snr")})
    # other categories (does the same lure capture successes or declined stars?)
    ex["noise_coh"] = ex.noise_snr_at_top >= SNR_WRONG
    summ["noise_coh_rate_by_category"] = ex.groupby("category").noise_coh.mean().round(4).to_dict()
    print(ex.groupby("category").noise_coh.agg(["sum", "size", "mean"]))
    # exposure: dev stars whose noise source carries a coherent secondary signal (kept classes)
    has_sec = set(sec_kept.tic)
    ex["src_has_secondary"] = ex.noise_tic.isin(has_sec)
    ex["answered"] = ex.category.str.startswith("answered")
    expo = ex[ex.noise_classification.isin(KEPT)].groupby(["src_has_secondary"]).agg(
        n=("star_id", "size"), answered=("answered", "sum"),
        wrong=("category", lambda c: (c == "answered_wrong").sum()),
        success=("category", lambda c: (c == "answered_success").sum()))
    expo["wrong_per_answered"] = expo.wrong / expo.answered
    print(expo)
    summ["exposure_kept_classes"] = expo.reset_index().to_dict("records")

    # ---- the repeat offenders
    off = [10001742265, 10001638867, 10001514155, 10002408980, 10001478100, 10001742277, 10001658496]
    o = w[w.noise_tic.isin(off)][["set", "mode", "star_id", "true_period", "top_period", "noise_tic",
                                  "noise_classification", "noise_P_ref", "noise_snr_at_top", "why"]]
    print(o.sort_values(["noise_tic", "star_id"]).to_string())
    summ["offender_peaks"] = {str(t): pk[(pk.tic == t) & (pk.min_half_snr >= 5)][
        ["kind", "rank", "P", "min_half_snr", "snr_full"]].round(3).to_dict("records") for t in off}
    print(json.dumps(summ["offender_peaks"], indent=0)[:3000])
    json.dump(summ, open(os.path.join(DATA, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
