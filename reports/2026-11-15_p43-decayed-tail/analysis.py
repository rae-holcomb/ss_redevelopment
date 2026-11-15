"""P43 analysis: tables and figures from the scorecards, the harness pickles and the profile CSVs.

Usage: python reports/2026-11-15_p43-decayed-tail/analysis.py <recommended_variant>
Reads
  benchmarks/baselines/dev_medium_<mode>_v7_tables/per_star.csv   (_v7 = local_screen2)
  runs/p43/sc/<variant>_{dmfy,dmas}/per_star.csv                  (P43 scorecards; local_tol = reference)
  runs/p43/harness/dev_medium_<mode>.pkl                          (candidate-level stats, every variant)
  runs/p43/prof/prof_<variant>_r<k>.csv                           (serial timing, profile_stars.txt)
Writes data/*.csv and figures/*.png.
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p43lib as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = P.ROOT
L = P.L
FIG, DATA = HERE / "figures", HERE / "data"
MODES = (("fullyear", "dmfy"), ("assigned", "dmas"))
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
SUCC = ("correct", "diffrot")
REFS = ["local_screen2", "local_tol"]
VARIANTS = ["local_tol_pos", "local_tol_tail", "local_tol_norej", "local_tol_norej_tail", "local_tol_norej_pos",
            "local_tol_tail_rh", "local_tol_norej_rh"]
COLORS = {"local_screen2": "#2166ac", "local_tol": "#8073ac", "local_tol_pos": "#e08214",
          "local_tol_tail": "#7fbc41", "local_tol_norej": "#b2182b", "local_tol_norej_tail": "#1a1a1a",
          "local_tol_norej_pos": "#c51b7d", "local_tol_tail_rh": "#35978f", "local_tol_norej_rh": "#f4a582"}
LABEL = {"local_screen2": "_v7 (local_screen2)"}
PROD_STAGES = ["load", "acf_long", "acf_short", "guess_pairwise_histogram", "guess_lombscargle", "guess_acf_fft", "fit"]


def run_dir(v: str, tag: str) -> Path:
    """Scorecard output folder for variant v and mode tag."""
    if v == "local_screen2":
        mode = "fullyear" if tag == "dmfy" else "assigned"
        return ROOT / f"benchmarks/baselines/dev_medium_{mode}_v7_tables"
    return ROOT / "runs/p43/sc" / f"{v}_{tag}"


def load_per_star() -> dict:
    """{(variant, mode): per_star DataFrame} for every finished scorecard."""
    out = {}
    for v in REFS + VARIANTS:
        for mode, tag in MODES:
            p = run_dir(v, tag) / "per_star.csv"
            if p.exists():
                out[(v, mode)] = pd.read_csv(p, dtype={"star_id": str})
    return out


def rates(ps: pd.DataFrame) -> dict:
    """success / wrong / alias / none fractions of one per_star table."""
    oc = ps.final_outcome
    return dict(n=len(oc), success=oc.isin(SUCC).mean(), wrong=(oc == "wrong").mean(),
                alias=oc.isin(["half", "double"]).mean(), none=(oc == "none").mean(),
                precision=oc.isin(SUCC).sum() / max((oc != "none").sum(), 1))


def _diff(b: pd.DataFrame, x: pd.DataFrame) -> dict:
    """Star-level changes from b to x."""
    j = b[["star_id", "final_outcome"]].merge(x[["star_id", "final_outcome"]], on="star_id", suffixes=("_b", "_v"))
    sb, sv = j.final_outcome_b.isin(SUCC), j.final_outcome_v.isin(SUCC)
    return dict(gained=int((~sb & sv).sum()), lost=int((sb & ~sv).sum()),
                none_to_wrong=int(((j.final_outcome_b == "none") & (j.final_outcome_v == "wrong")).sum()),
                wrong_to_none=int(((j.final_outcome_b == "wrong") & (j.final_outcome_v == "none")).sum()))


def timing() -> pd.DataFrame:
    """Mean serial s/star per variant (production path, total, fit), median over rounds."""
    rows = []
    for f in glob.glob(str(ROOT / "runs/p43/prof/prof_*_r*.csv")):
        v, r = Path(f).stem[len("prof_"):].rsplit("_", 1)
        d = pd.read_csv(f)
        prod = d[[c for c in PROD_STAGES if c in d.columns]].sum(axis=1)
        rows.append(dict(variant=v, round=r, total=d.total.mean(), production=prod.mean(), fit=d.fit.mean()))
    if not rows:
        return pd.DataFrame(columns=["variant", "total", "production", "fit", "n_rounds"])
    t = pd.DataFrame(rows)
    g = t.groupby("variant")
    out = g[["total", "production", "fit"]].median()
    out["n_rounds"] = g.size()
    return out.reset_index()


def summary_table(per_star: dict) -> pd.DataFrame:
    """One row per variant and mode: rates and star changes vs _v7 and vs local_tol."""
    rows = []
    for (v, mode), ps in per_star.items():
        r = dict(variant=v, mode=mode, **rates(ps))
        for base, tag in (("local_screen2", "v7"), ("local_tol", "tol")):
            b = per_star.get((base, mode))
            if b is not None and base != v:
                r.update({f"{k}_vs_{tag}": val for k, val in _diff(b, ps).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def fig_by_bin(per_star: dict, variants: list):
    """Success, alias and wrong by period bin per variant, both modes."""
    fig, ax = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    rows = []
    for j, (mode, _) in enumerate(MODES):
        for v in variants:
            ps = per_star.get((v, mode))
            if ps is None:
                continue
            g = ps.groupby("period_bin").final_outcome
            vals = dict(success=g.apply(lambda x: x.isin(SUCC).mean()).reindex(BINS),
                        alias=g.apply(lambda x: x.isin(["half", "double"]).mean()).reindex(BINS),
                        wrong=g.apply(lambda x: (x == "wrong").mean()).reindex(BINS))
            for b in BINS:
                rows.append(dict(variant=v, mode=mode, period_bin=b, **{k: s[b] for k, s in vals.items()}))
            ls, lw = ("-", 2.5) if v in REFS else ("--", 1.3)
            for i, k in enumerate(vals):
                ax[i, j].plot(BINS, vals[k].values, ls, marker="o", color=COLORS[v], lw=lw, label=LABEL.get(v, v))
        for i, k in enumerate(("success", "alias", "wrong")):
            ax[i, j].set_title(f"{mode}: {k} by true-period bin")
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7.5, ncol=2)
    for j in (0, 1):
        ax[2, j].set_xlabel("true period bin")
    fig.tight_layout()
    fig.savefig(FIG / "fig_by_bin.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "by_bin.csv", index=False)


def harness_candidates() -> pd.DataFrame:
    """All dev_medium harness candidates (every variant, both modes) with success / wrong / alias flags."""
    out = []
    for mode, _ in MODES:
        p = ROOT / f"runs/p43/harness/dev_medium_{mode}.pkl"
        if not p.exists():
            continue
        recs = pickle.loads(p.read_bytes())
        for v in recs[0]["variants"]:
            c = L.candidate_frame(recs, v)
            c["variant"], c["mode"] = v, mode
            out.append(c)
    d = pd.concat(out, ignore_index=True)
    d["succ"] = d.outcome.isin(SUCC)
    d["wrong"] = d.outcome == "wrong"
    d["alias"] = d.outcome.isin(["half", "double"])
    return d


def harness_summary() -> pd.DataFrame:
    """Overall rates per variant and mode from the harness (screening numbers)."""
    rows = []
    for mode, _ in MODES:
        p = ROOT / f"runs/p43/harness/dev_medium_{mode}.pkl"
        if not p.exists():
            continue
        recs = pickle.loads(p.read_bytes())
        for v in recs[0]["variants"]:
            rows.append(dict(mode=mode, **L.summarize(recs, v)))
    return pd.DataFrame(rows)


def _other_ok(d: pd.DataFrame) -> pd.Series:
    """Candidates that fail no gate except (possibly) frac_vertex_tol."""
    f = d.failed.fillna("").str.replace("frac_vertex_tol", "")
    return f.str.strip(";").str.len() == 0


def fig_fvt(c: pd.DataFrame):
    """frac_vertex_tol over all teeth vs positive-height teeth, correct vs wrong (local_tol candidates)."""
    x = c[c.variant == "local_tol"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    rows = []
    bins = np.linspace(0, 1, 21)
    for i, (sel, lab) in enumerate(((np.ones(len(x), bool), "all fitted candidates"),
                                    (_other_ok(x).values, "candidates passing every other gate"))):
        xs = x[sel]
        for j, (col, title) in enumerate((("fvt_all", "frac_vertex_tol, all teeth (local_tol)"),
                                          ("fvt_pos", "frac_vertex_tol, positive-height teeth only (variant a)"))):
            a = ax[i, j]
            for flag, nm, colr in (("succ", "correct/diffrot", "tab:green"), ("wrong", "wrong", "tab:red"),
                                   ("alias", "alias", "tab:orange")):
                v = xs.loc[xs[flag], col].dropna()
                a.hist(v, bins=bins, histtype="step", lw=2, color=colr, density=True, label=f"{nm} (n={len(v)})")
                rows.append(dict(subset=lab, stat=col, group=nm, n=len(v), kept=float((v >= 0.6).mean())))
            a.axvline(0.6, color="k", ls="--")
            ks = (xs.loc[xs.succ, col] >= 0.6).mean()
            kw = (xs.loc[xs.wrong, col] >= 0.6).mean()
            a.set_title(f"{title}\n{lab}: kept at 0.6: correct {ks:.0%}, wrong {kw:.0%}", fontsize=9)
            a.legend(fontsize=8, loc="upper left")
            a.set_xlabel("fraction of teeth whose free vertex is within tol_n of the comb")
    fig.suptitle("dev_medium, both modes pooled; dashed line = min_frac_tol 0.6 (unchanged)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fvt_all_vs_positive.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "fvt_stats.csv", index=False)


def fig_drift(c: pd.DataFrame):
    """drift_a_frac (local_tol candidates) for correct vs alias vs wrong, and diffrot vs not among correct."""
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv", dtype={"star_id": str})[["star_id", "shear"]]
    x = c[c.variant == "local_tol"].merge(man, on="star_id")
    x = x.assign(la=np.log10(np.abs(x.drift_frac).clip(lower=1e-7)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    rows = []
    bins = np.linspace(-6, 0, 49)
    groups = (("succ", "correct/diffrot", "tab:green"), ("alias", "alias", "tab:orange"), ("wrong", "wrong", "tab:red"))
    for flag, nm, colr in groups:
        s = x[x[flag]]
        fin = s[np.isfinite(s.drift_frac)]
        med = float(np.median(np.abs(fin.drift_frac))) if len(fin) else np.nan
        ax[0].hist(fin.la, bins=bins, histtype="step", lw=2, density=True, color=colr,
                   label=f"{nm}: median {med:.1e} ({len(fin)}/{len(s)} finite)")
        rows.append(dict(group=nm, n=len(s), n_finite=len(fin), median_abs=med,
                         q25=float(np.quantile(np.abs(fin.drift_frac), 0.25)) if len(fin) else np.nan,
                         q75=float(np.quantile(np.abs(fin.drift_frac), 0.75)) if len(fin) else np.nan))
    ax[0].set_title("all fitted candidates (local_tol, both modes)", fontsize=10)
    s = x[x.succ & np.isfinite(x.drift_frac)]
    for sub, nm, colr in ((s[s.shear.abs() == 0], "correct, shear = 0", "tab:blue"),
                          (s[s.shear.abs() > 0], "correct, shear != 0", "tab:purple")):
        med = float(np.median(np.abs(sub.drift_frac))) if len(sub) else np.nan
        ax[1].hist(sub.la, bins=bins, histtype="step", lw=2, density=True, color=colr,
                   label=f"{nm}: median {med:.1e} (n={len(sub)})")
        rows.append(dict(group=nm, n=len(sub), n_finite=len(sub), median_abs=med,
                         q25=float(np.quantile(np.abs(sub.drift_frac), 0.25)) if len(sub) else np.nan,
                         q75=float(np.quantile(np.abs(sub.drift_frac), 0.75)) if len(sub) else np.nan))
    ax[1].set_title("correct candidates: differential rotation vs none", fontsize=10)
    for a in ax[:2]:
        a.set_xlabel("log10 |drift_a_frac|  (c_n = q0 + q1 n + a n^2, a / P)")
        a.legend(fontsize=7.5, loc="upper left")
    # selected candidates' |a/P| vs |shear| (does it track the injected shear at all?)
    sel = x[x.selected & x.succ & np.isfinite(x.drift_frac)]
    ax[2].scatter(np.abs(sel.shear) + 1e-3, np.abs(sel.drift_frac), s=8, alpha=0.5)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    rho = sel[["shear", "drift_frac"]].abs().corr(method="spearman").iloc[0, 1] if len(sel) > 3 else np.nan
    ax[2].set_title(f"selected correct answers: |a/P| vs |shear| (+1e-3)\nSpearman rho = {rho:.2f}", fontsize=10)
    ax[2].set_xlabel("|shear| + 1e-3"); ax[2].set_ylabel("|drift_a_frac|")
    fig.tight_layout()
    fig.savefig(FIG / "fig_drift_a_frac.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "drift_a_frac_summary.csv", index=False)


def lost_table(per_star: dict, harness_c: pd.DataFrame, v: str) -> pd.DataFrame:
    """local_tol's lost stars vs _v7 (both modes), whether variant v recovers them, with the correct
    candidate's best local_tol gate values (from the harness)."""
    out = []
    for mode, _ in MODES:
        b, t, x = (per_star.get((k, mode)) for k in ("local_screen2", "local_tol", v))
        if b is None or t is None or x is None:
            continue
        j = b[["star_id", "true_period", "period_bin", "final_outcome"]].merge(
            t[["star_id", "final_outcome"]], on="star_id", suffixes=("_v7", "_tol")).merge(
            x[["star_id", "final_outcome"]].rename(columns={"final_outcome": "final_outcome_var"}), on="star_id")
        j = j[j.final_outcome_v7.isin(SUCC) & ~j.final_outcome_tol.isin(SUCC)].copy()
        j["mode"] = mode
        out.append(j)
    if not out:
        return pd.DataFrame()
    lt = pd.concat(out, ignore_index=True)
    lt["recovered"] = lt.final_outcome_var.isin(SUCC)
    hc = harness_c[(harness_c.variant == "local_tol") & harness_c.succ]
    best = (hc.sort_values("n_dropped").groupby(["star_id", "mode"])
            .agg(snr=("v_mean_height_snr", "max"), n_dropped=("n_dropped", "min"), fvt=("fvt_all", "max"))
            .reset_index())
    return lt.merge(best, on=["star_id", "mode"], how="left")


def fig_lost_scatter(lt: pd.DataFrame, v: str):
    """local_tol's lost stars: true period vs the correct candidate's best mean height SNR, recovered by v or not."""
    if not len(lt):
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    cmap = dict(zip(BINS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))
    for a, (mode, _) in zip(ax, MODES):
        s = lt[lt["mode"] == mode]
        for rec, mk, lab in ((True, "o", f"recovered by {v}"), (False, "x", "still lost")):
            q = s[s.recovered == rec]
            a.scatter(q.true_period, q.snr.fillna(0.05), c=[cmap.get(b, "k") for b in q.period_bin], marker=mk,
                      s=60, label=f"{lab} ({len(q)})")
        a.axhline(1.0, color="k", ls="--", lw=1)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("true period (d)")
        a.set_title(f"{mode}: {len(s)} stars _v7 got right and local_tol does not", fontsize=10)
        a.legend(fontsize=8, loc="lower right")
        for b, col in cmap.items():
            a.plot([], [], "s", color=col, label=b)
    ax[0].set_ylabel("best local_tol mean height SNR among the correct candidates\n(gate at 1.0; 0.05 = no correct candidate)")
    h = [plt.Line2D([], [], marker="s", ls="", color=col, label=b) for b, col in cmap.items()]
    ax[1].add_artist(ax[1].legend(handles=h, title="period bin", fontsize=8, loc="upper left"))
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_lost_scatter_{v}.png", dpi=110)
    plt.close(fig)


def changed_vs_tol(per_star: dict, v: str) -> pd.DataFrame:
    """Stars whose outcome changes between local_tol and v (both modes)."""
    out = []
    for mode, _ in MODES:
        b, x = per_star.get(("local_tol", mode)), per_star.get((v, mode))
        if b is None or x is None:
            continue
        j = b.merge(x, on="star_id", suffixes=("_base", "_var"))
        sb, sv = j.final_outcome_base.isin(SUCC), j.final_outcome_var.isin(SUCC)
        j["change"] = np.where(~sb & sv, "gained", np.where(sb & ~sv, "lost",
                               np.where(j.final_outcome_base != j.final_outcome_var, "other", "same")))
        j["mode"] = mode
        out.append(j[j.change != "same"][["star_id", "mode", "true_period_base", "period_bin_base",
                                          "final_period_base", "final_outcome_base", "decline_gates_base",
                                          "final_period_var", "final_outcome_var", "decline_gates_var", "change"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _draw_comb(a, st, cand, spec, color, label):
    """Draw one fitted candidate's teeth on an ACF axis: fitted parabolae, comb centres, tolerance bands."""
    if cand is None or cand.fit is None:
        return
    fit = cand.fit
    for t in P.tooth_table(cand, spec):
        lag = np.linspace(t["lag_lo"], t["lag_hi"], 40)
        c = fit.per_peak[t["n"]]["center"]
        a.plot(lag, t["h"] - t["A"] * (lag - c) ** 2, color=color, lw=2)
        if not t["lag0"]:
            a.axvspan(t["m"] - t["tol"], t["m"] + t["tol"], color=color, alpha=0.12)
            if np.isfinite(t["v"]):
                a.plot(t["v"], t["h_free"], "v", color=color, ms=6, mec="k", mew=0.5)
    a.plot([], [], color=color, lw=2, label=label)


def fig_examples(ch: pd.DataFrame, v: str, kinds: tuple, fname: str, n: int = 4, seed: int = 0):
    """LC + ACF for n stars of each change kind: local_tol's and v's chosen (or best correct) candidate drawn."""
    x = pd.concat([ch[ch.change == k].sample(min(n, (ch.change == k).sum()), random_state=seed) for k in kinds])
    if not len(x):
        return
    x = x.sort_values(["change", "true_period_base"])
    fig, ax = plt.subplots(len(x), 2, figsize=(15, 2.9 * len(x)), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = np.atleast_2d(ax)
    for i, (_, r) in enumerate(x.iterrows()):
        st, pooled = P.star_and_guesses(r.star_id, r["mode"])
        a0, a1 = ax[i]
        a0.plot(st["t"], st["f"], ",", color="k")
        a0.set_title(f"{r.star_id[-6:]} ({r['mode']}), P_true {r.true_period_base:.3g} d [{r.change}]", fontsize=9)
        a1.plot(st["lags"], st["acf"], color="0.35", lw=0.8)
        Pt = r.true_period_base
        for k in range(1, int(st["lags"][-1] / Pt) + 1):
            a1.axvline(k * Pt, color="tab:green", alpha=0.3, lw=1.2)
        txt = []
        for vv, col in (("local_tol", "tab:red"), (v, "tab:blue")):
            res = P.C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=vv)
            if res.success:
                cand, what = res.candidates[0], f"{P.classify(res.best_fit.P, st)} {res.best_fit.P:.3g}"
            else:  # declined: draw the correct candidate if there is one, else the lowest-redchi one
                ok = [c for c in res.candidates if c.fit is not None]
                corr = [c for c in ok if P.classify(c.fit.P, st) in SUCC]
                cand = corr[0] if corr else (ok[0] if ok else None)
                what = (f"none; drawn {'correct' if corr else 'top'} cand P={cand.fit.P:.3g} fails "
                        f"{','.join(cand.failed_gates)}; dropped {cand.fit.n_peaks_dropped}") if cand else "none"
            _draw_comb(a1, st, cand, P.C.COMB_VARIANTS[vv], col, f"{vv}: {what}")
            txt.append(f"{vv}: {what}")
        a1.set_title(" | ".join(txt), fontsize=7.5)
        a1.legend(fontsize=6.5, loc="upper right")
        a1.set_xlim(0, min(st["lags"][-1], 8.5 * Pt))
        a1.set_ylim(-0.6, 1.05)
    fig.suptitle(f"{v} vs local_tol. Green = P_true multiples; curves = fitted tooth parabolae (comb-tied centre), "
                 f"shaded = tolerance window tol_n around the comb, triangles = free vertices", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=90)
    plt.close(fig)


def fig_runtime(tim: pd.DataFrame):
    """Serial production-path and fit s/star per variant."""
    if not len(tim):
        return
    t = tim.sort_values("production")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t.production, 0.4, color=[COLORS.get(v, "0.5") for v in t.variant], label="production path")
    ax.barh(y + 0.2, t.fit, 0.4, color="0.7", label="comb fit only")
    for yy, (p, f) in enumerate(zip(t.production, t.fit)):
        ax.text(p, yy - 0.2, f" {p:.3f}", va="center", fontsize=8)
        ax.text(f, yy + 0.2, f" {f:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, t.variant)
    ax.set_xlabel("mean serial s / star (profile_stars.txt, dev_medium full-year; median over rounds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_runtime.png", dpi=110)
    plt.close(fig)


def main():
    """Build every table and figure that its inputs allow."""
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    rec = sys.argv[1] if len(sys.argv) > 1 else "local_tol_norej"
    per_star = load_per_star()
    tab = summary_table(per_star)
    tab.to_csv(DATA / "variant_metrics.csv", index=False)
    print(tab.round(4).to_string())
    fig_by_bin(per_star, [v for v in REFS + VARIANTS if (v, "fullyear") in per_star])
    hs = harness_summary()
    hs.to_csv(DATA / "harness_summary.csv", index=False)
    print(hs.round(4).to_string())
    c = harness_candidates()
    fig_fvt(c)
    fig_drift(c)
    lt = lost_table(per_star, c, rec)
    lt.to_csv(DATA / f"local_tol_lost_vs_v7_{rec}.csv", index=False)
    fig_lost_scatter(lt, rec)
    tim = timing()
    tim.to_csv(DATA / "timing_summary.csv", index=False)
    print(tim.round(4).to_string())
    fig_runtime(tim)
    ch = changed_vs_tol(per_star, rec)
    ch.to_csv(DATA / f"changed_stars_{rec}.csv", index=False)
    fig_examples(ch, rec, ("gained",), f"fig_examples_{rec}_recovered.png", n=5)
    fig_examples(ch, rec, ("lost", "other"), f"fig_examples_{rec}_lost_or_wrong.png", n=3)


if __name__ == "__main__":
    main()
