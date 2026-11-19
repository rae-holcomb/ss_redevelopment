"""Figures for the P43 clean-rerun report, from the saved tables in data/ only (no pipeline runs).

Run from the repo root:  python reports/2026-11-16_p43-clean-rerun/analysis.py
"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
D, F = HERE / "data", HERE / "figures"
F.mkdir(exist_ok=True)
S = {"correct", "diffrot"}
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
MODES = ("fullyear", "assigned")
VARS = ["local_screen2", "local_tol", "local_tol_pos", "local_tol_trim", "local_tol_trim_pos", "local_tol_cs",
        "local_tol_cs_trim", "local_tol_cs0", "local_tol_cs0_trim"]
BIN_VARS = ["local_screen2", "local_tol", "local_tol_pos", "local_tol_cs", "local_tol_trim", "local_tol_cs_trim"]
LABEL = {"local_screen2": "_v7 (local_screen2)"}
STYLE = {"local_screen2": dict(color="k", lw=2.2, ls="-"), "local_tol": dict(color="0.45", lw=2.2, ls="-"),
         "local_tol_pos": dict(color="#1f77b4", ls="--"), "local_tol_trim": dict(color="#d62728", lw=2),
         "local_tol_trim_pos": dict(color="#ff9896", ls="--"), "local_tol_cs": dict(color="#2ca02c", ls="--"),
         "local_tol_cs_trim": dict(color="#9467bd", lw=1.6), "local_tol_cs0": dict(color="#98df8a", ls=":"),
         "local_tol_cs0_trim": dict(color="#c5b0d5", ls=":")}


def fig_bins():
    """Success / alias / wrong fraction per period bin, every variant, both modes (harness screen)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for j, mode in enumerate(MODES):
        T = pd.read_csv(D / f"screen_{mode}.csv", index_col=0)
        for i, (name, sel) in enumerate([("success", lambda s: s.isin(S)),
                                         ("alias (half/double)", lambda s: s.isin(["half", "double"])),
                                         ("wrong", lambda s: s == "wrong")]):
            ax = axes[i, j]
            for v in BIN_VARS:
                y = [sel(T.loc[T.period_bin == b, v]).mean() for b in BINS]
                ax.plot(BINS, y, marker="o", label=LABEL.get(v, v), **STYLE[v])
            ax.set_ylabel(f"{name} fraction")
            ax.set_title(f"dev_medium_clean {mode}: {name}")
            ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("true period bin")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(F / "fig_bins.png", dpi=110)
    plt.close(fig)


def fig_lost_gates():
    """Why local_tol loses _v7's stars: failed gates of the closest success candidate, rejection drops, recoveries."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1.5]))
    gates = ["mean_height_snr", "frac_vertex_tol", "half_lag_consistency", "n_peaks", "frac_positive_heights",
             "few_peaks_height"]
    ax = axes[0]
    w = 0.38
    for k, mode in enumerate(MODES):
        L = pd.read_csv(D / f"lost_{mode}.csv")
        fails = L.best_failed.fillna("").str.split(";")
        anyc = [sum(g in f for f in fails) for g in gates]
        sole = [sum(f == [g] for f in fails) for g in gates]
        x = np.arange(len(gates)) + (k - 0.5) * w
        ax.bar(x, anyc, w, color=["#1f77b4", "#ff7f0e"][k], alpha=0.35, label=f"{mode}: fails (any)")
        ax.bar(x, sole, w, color=["#1f77b4", "#ff7f0e"][k], label=f"{mode}: sole failed gate")
    ax.set_xticks(np.arange(len(gates)))
    ax.set_xticklabels(gates, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("lost stars")
    ax.set_title("Lost stars (v7 success, local_tol not):\ngates failed by the closest success candidate")
    ax.legend(fontsize=7)

    ax = axes[1]
    for k, mode in enumerate(MODES):
        L = pd.read_csv(D / f"lost_{mode}.csv")
        rng = np.random.default_rng(k)
        jx = rng.uniform(-0.15, 0.15, len(L))
        snr_fail = L.best_failed.fillna("").str.contains("mean_height_snr|n_peaks")
        ax.scatter(L.lsq_n_dropped + jx, L.best_n_dropped + jx[::-1],
                   c=np.where(snr_fail, "#d62728", "0.5"), marker=["o", "s"][k], s=45, alpha=0.8,
                   label=f"{mode}")
    ax.plot([-0.5, 3.5], [-0.5, 3.5], "k:", lw=1)
    ax.set_xlabel("windows dropped by outlier rejection: lsq (_v7's selected fit)")
    ax.set_ylabel("windows dropped: local_tol (closest success cand.)")
    ax.set_title("Outlier rejection drops more teeth under local\n(red: fails mean_height_snr or n_peaks)")
    ax.legend(fontsize=8)

    ax = axes[2]
    cols = ["local_tol_pos", "local_tol_trim", "local_tol_trim_pos", "local_tol_cs", "local_tol_cs_trim",
            "local_tol_cs0", "local_tol_cs0_trim"]
    rows, labels = [], []
    for mode in MODES:
        L = pd.read_csv(D / f"lost_{mode}.csv")
        for _, r in L.iterrows():
            rows.append([{"correct": 2, "diffrot": 2, "none": 0, "wrong": -2}.get(r[f"out_{c}"], -1) for c in cols])
            labels.append(f"{r.star_id[-6:]} {mode[:4]} [{r.best_failed.replace('mean_height_snr', 'snr').replace('frac_vertex_tol', 'fvt').replace('half_lag_consistency', 'half_lag')}]")
    M = np.array(rows)
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#ff9896", "#eeeeee", "white", "#2ca02c"])
    ax.imshow(M, cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("local_tol_", "") for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title("Outcome of each lost star under each variant\n(green success, grey none, red wrong, pink alias)")
    fig.tight_layout()
    fig.savefig(F / "fig_lost_gates.png", dpi=110)
    plt.close(fig)


def fig_fvt():
    """frac_vertex_tol of candidates that pass every other gate: success vs wrong, threshold 0.6 marked."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    bins = np.linspace(0, 1, 12)
    for ax, v in zip(axes, ("local_tol", "local_tol_trim", "local_tol_cs_trim")):
        frames = [pd.read_csv(D / f"cands_{m}.csv.gz") for m in MODES]
        c = pd.concat(frames)
        c = c[c.variant == v]
        okcols = [k for k in c.columns if k.startswith("ok_") and k != "ok_frac_vertex_tol"]
        others = c[okcols].astype("boolean").fillna(True).astype(bool).all(axis=1)
        c = c[others]
        for lab, sel, col in [("success", c.outcome.isin(S), "#2ca02c"), ("wrong", c.outcome == "wrong", "#d62728"),
                              ("alias", c.outcome.isin(["half", "double"]), "#ff7f0e")]:
            x = c.loc[sel, "v_frac_vertex_tol"]
            keep = (x >= 0.6).mean() if len(x) else np.nan
            ax.hist(x, bins=bins, histtype="step", lw=2, color=col, density=True,
                    label=f"{lab} (n={len(x)}, kept {keep:.0%})")
        ax.axvline(0.6, color="k", ls="--", lw=1)
        ax.text(0.61, ax.get_ylim()[1] * 0.92, "min_frac_tol = 0.6", fontsize=8)
        ax.set_xlabel("frac_vertex_tol")
        ax.set_title(f"{v}: candidates passing every other gate\n(both modes pooled)")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(F / "fig_fvt.png", dpi=110)
    plt.close(fig)


def _acf_panel(ax, rec, v, title):
    """ACF with the shown candidate's teeth: fitted parabolae, model centres, tolerance windows, free vertices."""
    lags, acf = rec["lags"], rec["acf"]
    c = rec["variants"][v]
    P = c["P"]
    xmax = min(float(lags[-1]), (max(t["n"] for t in c["teeth"]) + 1.6) * P) if c["teeth"] else float(lags[-1])
    if c["n_trimmed"]:
        xmax = min(float(lags[-1]), (max(t["n"] for t in c["teeth"]) + c["n_trimmed"] + 1.2) * P)
    m = lags <= xmax
    ax.plot(lags[m], acf[m], color="0.55", lw=0.8)
    for t in c["teeth"]:
        if not t["lag0"]:
            ax.axvspan(t["model_c"] - t["tol"], t["model_c"] + t["tol"], color="#2ca02c" if t["tol_ok"] else "#d62728",
                       alpha=0.12, lw=0)
        xx = np.linspace(t["lag_lo"], t["lag_hi"], 60)
        ax.plot(xx, t["height"] - t["curvature"] * (xx - t["center"]) ** 2, color="#1f77b4", lw=1.6)
        ax.text(t["center"], t["height"] + 0.04, f"{t['height']:+.2f}", ha="center", fontsize=6.5, color="#1f77b4")
        if np.isfinite(t["free_vertex"]):
            ax.plot(t["free_vertex"], t["free_height"], marker="v", color="k", ms=5)
        elif not t["lag0"]:
            ax.text(t["model_c"], ax.get_ylim()[0] if False else -0.05, "no vtx", fontsize=6, ha="center", color="#d62728")
    for k in range(1, int(xmax / rec["true_p"]) + 1):
        ax.axvline(k * rec["true_p"], color="orange", lw=0.7, ls=":")
    g = c["gates"]
    gtxt = ", ".join(f"{k.replace('frac_vertex_tol', 'fvt').replace('mean_height_snr', 'snr')}"
                     f"={val:.2f}" for k, (val, ok) in g.items() if k in ("frac_vertex_tol", "mean_height_snr")
                     and val is not None)
    ax.set_title(f"{title}: P={P:.3g} ({c['outcome']}) -> star {c['star_outcome']}\n"
                 f"failed: {', '.join(c['failed']) or 'none'} | {gtxt} | dropped {c['n_dropped']}, trimmed {c['n_trimmed']}",
                 fontsize=7.5)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("lag (d)")


def fig_examples():
    """LC + ACF (local_tol vs local_tol_trim) for recovered, still-lost and newly-wrong example stars."""
    ex = pickle.loads((D / "examples.pkl").read_bytes())
    for group, fname in [("recovered", "fig_examples_recovered.png"), ("lost", "fig_examples_lost.png"),
                         ("newwrong", "fig_examples_newwrong.png")]:
        recs = [r for r in ex if r["group"] == group]
        fig, axes = plt.subplots(len(recs), 3, figsize=(17, 2.9 * len(recs)),
                                 gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(recs):
            ax = axes[i, 0]
            ax.plot(r["t"], r["f"], ",", color="k", alpha=0.5)
            ax.set_title(f"{r['star_id'][-6:]} ({r['mode']}), P_true={r['true_p']:.3g} d", fontsize=8)
            ax.set_xlabel("time (d)")
            _acf_panel(axes[i, 1], r, "local_tol", "local_tol")
            _acf_panel(axes[i, 2], r, "local_tol_trim", "local_tol_trim")
        fig.suptitle({"recovered": "Recovered by local_tol_trim", "lost": "Still lost under local_tol_trim",
                      "newwrong": "Newly wrong under local_tol_trim"}[group] +
                     " -- blue: fitted parabolae (height labelled); shaded: tolerance window per tooth (green = free "
                     "vertex inside, red = not); black triangles: free vertices; orange dotted: k * P_true", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(F / fname, dpi=100)
        plt.close(fig)


def fig_runtime():
    """Median per-star fit seconds and total production-path seconds, from the profile runs (data/profile_*.json)."""
    rows = []
    for p in sorted(D.glob("profile_*.csv")):
        v, mode = p.stem.replace("profile_", "").rsplit("_", 1)
        df = pd.read_csv(p)
        per = df.groupby("star_id")[["fit", "total"]].median()
        rows.append(dict(variant=v, mode=mode, fit=per.fit.median(), total=per.total.median()))
    if not rows:
        return
    R = pd.DataFrame(rows)
    R.to_csv(D / "runtime_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col in zip(axes, ("fit", "total")):
        piv = R.pivot(index="variant", columns="mode", values=col)
        piv.plot.bar(ax=ax, rot=0)
        ax.set_ylabel(f"median s / star ({col})")
        ax.set_title(f"{col} stage, 15 profile stars, median of 3 repeats")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(F / "fig_runtime.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_bins()
    fig_lost_gates()
    fig_fvt()
    if (D / "examples.pkl").exists():
        fig_examples()
    fig_runtime()
    print("figures ->", F)
