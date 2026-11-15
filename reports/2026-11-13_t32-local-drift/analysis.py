"""T32 analysis: tables and figures from the scorecard runs, the harness pickles and the profile CSVs.

Usage: python reports/2026-11-13_t32-local-drift/analysis.py [recommended_variant]
Reads
  runs/t32/sc/<variant>_{dmfy,dmas}/per_star.csv          (T32 scorecards)
  benchmarks/baselines/dev_medium_<mode>_v6w_tables       (lsq)
  benchmarks/baselines/dev_medium_<mode>_v7_tables        (local_screen2, the default)
  runs/t32/sc/local_{dmfy,dmas}                           (local, rerun here: T31's runs were deleted)
  runs/t32/harness/dev_medium_<mode>.pkl                   (candidate-level gate statistics)
  runs/t32/prof/prof_<variant>_<mode>_r<k>.csv             (serial timing, 15 T12 stars)
Writes data/*.csv and figures/*.png.
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t32lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = L.ROOT
T31 = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/t31-comb-fit-speed-e7e5bb/runs/t31/runs")
MAIN = Path("/Users/raeholcomb/Documents/GitHub/ss_redevelopment/runs/t31_merge")
FIG, DATA = HERE / "figures", HERE / "data"
MODES = (("fullyear", "dmfy"), ("assigned", "dmas"))
REFS = ["lsq", "local", "local_screen2"]
NEW = ["local_track", "local_tol", "local_quad", "local_snrw", "local_scatter", "local_track_tol",
       "local_track_scatter", "local_robust"]
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
COLORS = {"lsq": "#b2182b", "local": "#ef8a62", "local_screen2": "#2166ac", "local_track": "#7fbc41",
          "local_tol": "#8073ac", "local_quad": "#e08214", "local_snrw": "#c51b7d", "local_scatter": "#35978f",
          "local_track_tol": "#542788", "local_track_scatter": "#01665e", "local_robust": "#1a1a1a"}
SUCC = ("correct", "diffrot")
PROD_STAGES = ["load", "acf_long", "acf_short", "guess_pairwise_histogram", "guess_lombscargle", "guess_acf_fft", "fit"]


def run_dir(v: str, tag: str) -> Path:
    """Scorecard output folder for variant v and mode tag."""
    mode = "fullyear" if tag == "dmfy" else "assigned"
    if v == "lsq":  # T31 Part A = lsq with the reproducible wavelet (_v6w baseline)
        return ROOT / f"benchmarks/baselines/dev_medium_{mode}_v6w_tables"
    if v == "local_screen2":  # the default; _v7 baseline
        return ROOT / f"benchmarks/baselines/dev_medium_{mode}_v7_tables"
    return ROOT / "runs/t32/sc" / f"{v}_{tag}"


def load_per_star() -> dict:
    """{(variant, mode): per_star DataFrame} for every variant with a finished run."""
    out = {}
    for v in REFS + NEW:
        for mode, tag in MODES:
            p = run_dir(v, tag) / "per_star.csv"
            if p.exists():
                out[(v, mode)] = pd.read_csv(p)
    return out


def rates(ps: pd.DataFrame) -> dict:
    """success / wrong / alias / none fractions of one per_star table."""
    oc = ps.final_outcome
    n = len(oc)
    return dict(n=n, success=oc.isin(SUCC).mean(), wrong=(oc == "wrong").mean(),
                alias=oc.isin(["half", "double"]).mean(), none=(oc == "none").mean(),
                n_success=int(oc.isin(SUCC).sum()), n_wrong=int((oc == "wrong").sum()))


def timing() -> pd.DataFrame:
    """Mean serial s/star per variant and mode (scorecard path = total, production path), median over rounds."""
    rows = []
    for f in glob.glob(str(ROOT / "runs/t32/prof/prof_*_r*.csv")):
        name = Path(f).stem[len("prof_"):]
        v, mode, r = name.rsplit("_", 2)
        d = pd.read_csv(f)
        prod = d[[c for c in PROD_STAGES if c in d.columns]].sum(axis=1)
        rows.append(dict(variant=v, mode=mode, round=r, total=d.total.mean(), production=prod.mean(),
                         fit=d.fit.mean()))
    if not rows:
        return pd.DataFrame(columns=["variant", "mode", "total", "production", "fit"])
    t = pd.DataFrame(rows)
    return t.groupby(["variant", "mode"])[["total", "production", "fit"]].median().reset_index()


def summary_table(per_star: dict, tim: pd.DataFrame) -> pd.DataFrame:
    """One row per variant and mode: rates, timing, and star-level gains/losses vs local_screen2."""
    rows = []
    for (v, mode), ps in per_star.items():
        r = dict(variant=v, mode=mode, **rates(ps))
        base = per_star.get(("local_screen2", mode))
        if base is not None:
            j = base[["star_id", "final_outcome"]].merge(ps[["star_id", "final_outcome"]], on="star_id",
                                                         suffixes=("_b", "_v"))
            sb, sv = j.final_outcome_b.isin(SUCC), j.final_outcome_v.isin(SUCC)
            r.update(gained=int((~sb & sv).sum()), lost=int((sb & ~sv).sum()),
                     none_to_wrong=int(((j.final_outcome_b == "none") & (j.final_outcome_v == "wrong")).sum()),
                     wrong_to_none=int(((j.final_outcome_b == "wrong") & (j.final_outcome_v == "none")).sum()))
        tt = tim[(tim.variant == v) & (tim["mode"] == mode)]
        if len(tt):
            r.update(s_total=float(tt.total.iloc[0]), s_production=float(tt.production.iloc[0]),
                     s_fit=float(tt.fit.iloc[0]))
        rows.append(r)
    return pd.DataFrame(rows)


def fig_frontier(tab: pd.DataFrame):
    """Success and wrong vs production-path s/star, both modes."""
    if "s_production" not in tab:
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col")
    for j, (mode, _) in enumerate(MODES):
        t = tab[tab["mode"] == mode].dropna(subset=["s_production"])
        for i, met in enumerate(["success", "wrong"]):
            a = ax[i, j]
            for _, r in t.iterrows():
                mk = "*" if r.variant in REFS else "o"
                a.scatter(r.s_production, r[met], s=180 if mk == "*" else 70, marker=mk,
                          color=COLORS.get(r.variant, "k"), zorder=3, edgecolor="k", lw=0.5)
                a.annotate(r.variant, (r.s_production, r[met]), fontsize=7.5, xytext=(4, 3),
                           textcoords="offset points")
            b = t[t.variant == "local_screen2"]
            if len(b):
                a.axhline(float(b[met].iloc[0]), color=COLORS["local_screen2"], ls="--", lw=1)
            a.set_xscale("log")
            a.set_ylabel(f"{met} rate")
            a.set_title(f"{mode}: {met}" + ("  (higher is better)" if met == "success" else "  (lower is better)"))
            a.grid(alpha=0.3)
        ax[1, j].set_xlabel("production-path s/star (serial, 15 T12 stars; log)")
    fig.suptitle("T32 frontier: dev_medium success / wrong vs time (stars = references; dashed = local_screen2, the default)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_frontier.png", dpi=110)
    plt.close(fig)


def fig_by_bin(per_star: dict, variants: list):
    """Success and wrong by period bin per variant, both modes."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    rows = []
    for j, (mode, _) in enumerate(MODES):
        for v in variants:
            ps = per_star.get((v, mode))
            if ps is None:
                continue
            g = ps.groupby("period_bin").final_outcome
            s = g.apply(lambda x: x.isin(SUCC).mean()).reindex(BINS)
            w = g.apply(lambda x: (x == "wrong").mean()).reindex(BINS)
            for b in BINS:
                rows.append(dict(variant=v, mode=mode, period_bin=b, success=s[b], wrong=w[b]))
            ls = "-" if v in REFS else "--"
            lw = 2.5 if v in REFS else 1.3
            ax[0, j].plot(BINS, s.values, ls, marker="o", color=COLORS[v], label=v, lw=lw)
            ax[1, j].plot(BINS, w.values, ls, marker="o", color=COLORS[v], label=v, lw=lw)
        ax[0, j].set_title(f"{mode}: success by period bin")
        ax[1, j].set_title(f"{mode}: wrong by period bin")
        for i in (0, 1):
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7.5, ncol=2)
    ax[1, 0].set_xlabel("true period bin"); ax[1, 1].set_xlabel("true period bin")
    fig.tight_layout()
    fig.savefig(FIG / "fig_by_bin.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "by_bin.csv", index=False)


def harness_candidates() -> pd.DataFrame:
    """All dev_medium harness candidates (every variant, both modes) with success / wrong flags."""
    out = []
    for mode, _ in MODES:
        p = ROOT / f"runs/t32/harness/dev_medium_{mode}.pkl"
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
    return d


def _other_ok(d: pd.DataFrame, vertex_gates=("frac_vertex_in_window", "frac_vertex_tol", "vertex_scatter")):
    """Candidates that pass every gate except the vertex gate(s)."""
    f = d.failed.fillna("")
    for g in vertex_gates:
        f = f.str.replace(g, "")
    return f.str.strip(";").str.len() == 0


def fig_gate_stats(c: pd.DataFrame):
    """Distributions of each vertex-gate statistic for success vs wrong candidates (other gates passed)."""
    panels = [("local", "v_frac_vertex_in_window", 0.8, ">=", "frac_vertex_in_window (local, T31 gate)"),
              ("local_tol", "v_frac_vertex_tol", 0.6, ">=", "frac_vertex_tol, sqrt(n) growth (local_tol)"),
              ("local_track_tol", "v_frac_vertex_tol", 0.8, ">=", "frac_vertex_tol, tracked (local_track_tol)"),
              ("local_scatter", "vertex_scatter", 0.04, "<=", "vertex_scatter (local_scatter)"),
              ("local_track_scatter", "vertex_scatter", 0.04, "<=", "vertex_scatter, tracked (local_track_scatter)"),
              ("local_robust", "vertex_scatter", 0.04, "<=", "vertex_scatter, track+height+drift (local_robust)")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    rows = []
    for a, (v, col, thr, op, title) in zip(ax.flat, panels):
        x = c[(c.variant == v) & _other_ok(c[c.variant == v])] if (c.variant == v).any() else c.iloc[:0]
        if not len(x):
            a.set_visible(False)
            continue
        vals = x[col].replace(np.inf, np.nan)
        if "scatter" in col:
            bins = np.linspace(0, 0.15, 31)
            vals = vals.clip(upper=0.1499)
        else:
            bins = np.linspace(0, 1, 21)
        for flag, lab, colr in (("succ", "success", "tab:green"), ("wrong", "wrong", "tab:red")):
            a.hist(vals[x[flag]].dropna(), bins=bins, histtype="step", lw=2, color=colr, density=True,
                   label=f"{lab} (n={int(x[flag].sum())})")
        a.axvline(thr, color="k", ls="--")
        keep = (x[col] >= thr) if op == ">=" else (x[col] <= thr)
        ks, kw = keep[x.succ].mean(), keep[x.wrong].mean()
        a.set_title(f"{title}\nkept: success {ks:.0%}, wrong {kw:.0%}", fontsize=9)
        a.legend(fontsize=8)
        rows.append(dict(variant=v, stat=col, threshold=thr, op=op, n_success=int(x.succ.sum()),
                         n_wrong=int(x.wrong.sum()), kept_success=ks, kept_wrong=kw,
                         n_inf=int(np.isinf(x[col]).sum()) if "scatter" in col else 0))
    fig.suptitle("Vertex-gate statistics on dev_medium candidates that pass every other gate (both modes pooled; "
                 "scatter: inf = too few vertices, shown in the last bin)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_gate_stats.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "gate_stats.csv", index=False)


def fig_scatter_curve(c: pd.DataFrame):
    """Purity-completeness of the scatter threshold (candidate level) for the tracked scatter variant."""
    v = "local_track_scatter"
    x = c[(c.variant == v) & _other_ok(c[c.variant == v])]
    if not len(x):
        return
    ths = np.linspace(0.01, 0.12, 23)
    comp = [((x.vertex_scatter <= t) & x.succ).sum() / max(x.succ.sum(), 1) for t in ths]
    pur = [((x.vertex_scatter <= t) & x.succ).sum() / max((x.vertex_scatter <= t).sum(), 1) for t in ths]
    fig, a = plt.subplots(figsize=(6.5, 5))
    a.plot(comp, pur, "-o", color=COLORS[v])
    for t, cc, pp in zip(ths, comp, pur):
        if abs(t - 0.04) < 1e-9 or abs(t - 0.02) < 1e-9 or abs(t - 0.08) < 1e-9:
            a.annotate(f"{t:.2f}", (cc, pp), xytext=(5, -10), textcoords="offset points")
    a.set_xlabel("completeness: fraction of success candidates kept")
    a.set_ylabel("purity: success / all kept (candidate level)")
    a.set_title(f"{v}: vertex_scatter threshold\n(dev_medium candidates passing every other gate; 0.04 set on dev_small)",
                fontsize=9)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scatter_threshold_curve.png", dpi=110)
    plt.close(fig)


def fig_drift_a(c: pd.DataFrame):
    """|drift a| / P of success candidates with a fitted drift term, diffrot (shear != 0) vs none.

    a = 0 exactly means no drift term was fit (fewer than 4 teeth with a free
    vertex); those are counted in the legend, not plotted.
    """
    man = pd.read_csv(ROOT / "datasets/dev_medium.csv")[["star_id", "shear"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    rows = []
    for a, v in zip(ax, ["local_quad", "local_robust"]):
        x = c[(c.variant == v) & c.succ].merge(man, on="star_id")
        x = x[np.isfinite(x.drift_a_frac)]
        bins = np.linspace(-6, -0.5, 45)
        for sub, lab, col in ((x[x.shear.abs() == 0], "no diffrot (shear = 0)", "tab:blue"),
                              (x[x.shear.abs() > 0], "diffrot (shear != 0)", "tab:orange")):
            fit = sub[sub.drift_a_frac != 0]
            la = np.log10(np.abs(fit.drift_a_frac))
            med = float(np.median(np.abs(fit.drift_a_frac))) if len(fit) else np.nan
            a.hist(la, bins=bins, histtype="step", lw=2, density=True, color=col,
                   label=f"{lab}: {len(fit)}/{len(sub)} fit a drift term, median |a/P| {med:.1e}")
            rows.append(dict(variant=v, group=lab, n=len(sub), n_drift_fit=len(fit), median_abs=med,
                             q90_abs=float(np.quantile(np.abs(fit.drift_a_frac), 0.9)) if len(fit) else np.nan))
        a.axvline(np.log10(0.05 / 49), color="k", ls=":", lw=1)
        a.text(np.log10(0.05 / 49), a.get_ylim()[1] * 0.9, " a n^2 = 0.05 P at n = 7", fontsize=7)
        a.set_xlabel("log10 |a / P|  (c_n = t0 + n P + a n^2)")
        a.set_title(f"{v}: success candidates with a drift term (both modes)", fontsize=10)
        a.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_drift_a.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DATA / "drift_a_summary.csv", index=False)


def changed_stars(per_star: dict, v: str, base: str = "local_screen2") -> pd.DataFrame:
    """Stars whose success changes between base and v (both modes)."""
    out = []
    for mode, _ in MODES:
        b, x = per_star.get((base, mode)), per_star.get((v, mode))
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


def fig_examples(ch: pd.DataFrame, v: str, kind: str, n: int = 6, seed: int = 0):
    """Light curve + ACF for n gained or lost stars, P_true harmonics and both answers marked."""
    x = ch[ch.change == kind]
    if not len(x):
        return
    x = x.sample(min(n, len(x)), random_state=seed).sort_values("true_period_base")
    fig, ax = plt.subplots(len(x), 2, figsize=(14, 2.6 * len(x)), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = np.atleast_2d(ax)
    tasks = {m: {t["star_id"]: t for t in L.make_tasks(m)} for m in x["mode"].unique()}
    for i, (_, r) in enumerate(x.iterrows()):
        st = L.load_star(tasks[r["mode"]][r.star_id])
        a0, a1 = ax[i]
        a0.plot(st["t"], st["f"], ",", color="k")
        a0.set_title(f"{r.star_id[-6:]} ({r['mode']}), P_true {r.true_period_base:.3g} d", fontsize=9)
        a1.plot(st["lags"], st["acf"], color="0.3", lw=0.8)
        P = r.true_period_base
        for k in range(1, int(st["lags"][-1] / P) + 1):
            a1.axvline(k * P, color="tab:green", alpha=0.35, lw=1.5)
        for col, Pans, oc, lab in (("tab:red", r.final_period_base, r.final_outcome_base, "local_screen2"),
                                   ("tab:blue", r.final_period_var, r.final_outcome_var, v)):
            if pd.notna(Pans):
                for k in range(1, min(int(st["lags"][-1] / Pans), 12) + 1):
                    a1.axvline(k * Pans, color=col, ls="--", lw=1, alpha=0.8)
        dg = r.decline_gates_base if r.final_outcome_base == "none" else r.decline_gates_var
        a1.set_title(f"local_screen2 (red): {r.final_outcome_base} {r.final_period_base if pd.notna(r.final_period_base) else '':.4}"
                     f" | {v} (blue): {r.final_outcome_var} {r.final_period_var if pd.notna(r.final_period_var) else '':.4}"
                     f"\ndecline gates: {dg if isinstance(dg, str) else ''}", fontsize=8)
        a1.set_xlim(0, min(st["lags"][-1], 8.5 * max(P, np.nan_to_num(r.final_period_var, nan=P))))
    fig.suptitle(f"{v}: {kind} vs local_screen2 (green = P_true multiples, red/blue dashed = answers)")
    fig.tight_layout()
    fig.savefig(FIG / f"fig_examples_{v}_{kind}.png", dpi=95)
    plt.close(fig)


def main():
    """Build every table and figure that its inputs allow."""
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    per_star = load_per_star()
    tim = timing()
    tim.to_csv(DATA / "timing_summary.csv", index=False)
    tab = summary_table(per_star, tim)
    tab.to_csv(DATA / "variant_metrics.csv", index=False)
    print(tab.round(4).to_string())
    fig_frontier(tab)
    fig_by_bin(per_star, [v for v in REFS + NEW if (v, "fullyear") in per_star])
    c = harness_candidates()
    fig_gate_stats(c)
    fig_scatter_curve(c)
    fig_drift_a(c)
    rec = sys.argv[1] if len(sys.argv) > 1 else None
    if rec:
        ch = changed_stars(per_star, rec)
        ch.to_csv(DATA / f"changed_stars_{rec}.csv", index=False)
        fig_examples(ch, rec, "gained")
        fig_examples(ch, rec, "lost")


if __name__ == "__main__":
    main()
