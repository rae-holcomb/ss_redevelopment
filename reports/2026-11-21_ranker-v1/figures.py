"""Ranker v1 report figures. Reads only `data/` (written by analysis.py); writes `figures/`."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
BINS = ["<1d", "1-3d", "3-10d", "10-30d", ">30d"]
SETS = [("dev_medium", "fullyear"), ("dev_medium", "assigned"), ("dev_medium_b", "fullyear"),
        ("dev_medium_b", "assigned")]
# categorical slots 1-4 of the dataviz reference palette, fixed order
C = {"hand": "#2a78d6", "matched": "#eb6834", "pick": "#1baf7a", "rank": "#eda100"}
INK, MUTED = "#0b0b0b", "#8a8984"
info = json.load(open(os.path.join(DATA, "run_info.json")))
PICK, MATCHED = info["pick"], info["matched"]
LABEL = {"hand": "hand arbitration", MATCHED: "ranker v1, matched train precision",
         PICK: "ranker v1, pick (per-bin p = 0.86)", "rank_global": "ranker score only (v0-style)"}
COLOR = {"hand": C["hand"], MATCHED: C["matched"], PICK: C["pick"], "rank_global": C["rank"]}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
    return fig


def purity_completeness():
    """Precision vs answer rate (head score and ranker score sweeps) with named cutoffs."""
    cv = pd.read_csv(os.path.join(DATA, "curves.csv"))
    cuts = pd.read_csv(os.path.join(DATA, "cutoffs.csv"))
    panels = [("train_oof", "assigned")] + SETS
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharey=True)
    for ax, (s, m) in zip(axes, panels):
        for col, lab, colr in (("score_v1", "decline-model score sweep", C["matched"]),
                               ("top_score", "ranker-score sweep (v0-style)", C["rank"])):
            c = cv[(cv.set == s) & (cv["mode"] == m) & (cv.score == col)]
            ax.plot(c.answer_rate, c.precision, color=colr, lw=2, label=lab)
        k = cuts[(cuts.set == s) & (cuts["mode"] == m)]
        for rule, mk in (("hand", "*"), (MATCHED, "o"), (PICK, "D"), ("rank_global", "s")):
            r = k[k.rule == rule]
            if len(r):
                ax.plot(r.answer_rate, r.precision, mk, ms=11 if mk == "*" else 8, color=COLOR[rule],
                        mec="white", mew=1.5, label=LABEL[rule], zorder=5)
        neg = "negatives at 1:1" if s in ("train_oof", "dev_medium") else "no negatives (rotators only)"
        ax.set_title(f"{s} {m}\n({neg})", fontsize=10)
        ax.set_xlabel("answer rate (rotators answered / all rotators)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.3, 1.0)
        _style(ax)
    axes[0].set_ylabel("precision (success / answered, weighted)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Purity-completeness: precision vs answer rate; markers = operating points", y=1.04)
    return _save(fig, "fig_purity_completeness.png")


def per_bin():
    """Success / wrong / declined per true-period bin: hand vs ranker (matched, pick), 4 dev tables."""
    s_all = pd.read_csv(os.path.join(DATA, "summary_all.csv"))
    rules = ["hand", MATCHED, PICK]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    for j, (s, m) in enumerate(SETS):
        sub = s_all[(s_all.set == s) & (s_all["mode"] == m)]
        for i, met in enumerate(("success", "wrong", "declined")):
            ax = axes[i, j]
            x = np.arange(len(BINS) + 1)
            for k, r in enumerate(rules):
                v = sub[sub.rule == r].set_index("bin").reindex(BINS + ["all"])[met]
                ax.bar(x + (k - 1) * 0.27, v, 0.25, color=COLOR[r], label=LABEL[r])
            ax.set_xticks(x, BINS + ["all"])
            if j == 0:
                ax.set_ylabel(met)
            if i == 0:
                ax.set_title(f"{s} {m}")
            _style(ax)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per true-period bin: ranker v1 vs hand arbitration (dev sets, both coverage modes)", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig_per_bin.png")


def lt1d():
    """<1 d success (and wrong) per dev table at matched precision and at the pick."""
    s_all = pd.read_csv(os.path.join(DATA, "summary_all.csv"))
    rules = ["hand", "rank_global", MATCHED, PICK]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, met in zip(axes, ("success", "wrong")):
        x = np.arange(len(SETS))
        for k, r in enumerate(rules):
            v = [s_all[(s_all.set == s) & (s_all["mode"] == m) & (s_all.rule == r) & (s_all.bin == "<1d")][met].iloc[0]
                 for s, m in SETS]
            b = ax.bar(x + (k - 1.5) * 0.2, v, 0.18, color=COLOR[r], label=LABEL[r])
            ax.bar_label(b, fmt="%.2f", fontsize=7, color=INK)
        ax.set_xticks(x, [f"{s}\n{m}" for s, m in SETS])
        ax.set_title(f"< 1 d {met}")
        _style(ax)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig_lt1d.png")


def tuning():
    """Two-set protocol: dev success and wrong vs per-bin precision target, hand wrong rate as a line."""
    s_all = pd.read_csv(os.path.join(DATA, "summary_all.csv"))
    a = s_all[(s_all.bin == "all") & s_all.rule.str.startswith("head_perbin_p")].copy()
    a["p"] = a.rule.str[-4:].astype(float)
    fig, axes = plt.subplots(2, 4, figsize=(18, 6.5), sharex=True)
    for j, (s, m) in enumerate(SETS):
        g = a[(a.set == s) & (a["mode"] == m)].sort_values("p")
        h = s_all[(s_all.set == s) & (s_all["mode"] == m) & (s_all.rule == "hand") & (s_all.bin == "all")].iloc[0]
        for i, met in enumerate(("success", "wrong")):
            ax = axes[i, j]
            ax.plot(g.p, g[met], "-o", ms=4, color=C["pick"], lw=2, label="ranker v1 per-bin")
            ax.axhline(h[met], color=C["hand"], lw=2, ls="--", label="hand (= _v9 baseline)")
            ax.axvline(0.86, color=MUTED, lw=1)
            ax.axvline(0.84, color=MUTED, lw=1, ls=":")
            if i == 0:
                ax.set_title(f"{s} {m}" + ("  (tuning set)" if s == "dev_medium" else "  (judging set)"))
            if j == 0:
                ax.set_ylabel(met)
            if i == 1:
                ax.set_xlabel("per-bin precision target (OOF, negatives 1:1)")
            _style(ax)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Choosing the precision target: solid grey = 0.86 (tuned on dev_medium), dotted = 0.84 "
                 "(tuned on dev_medium_b)", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig_tuning.png")


def importance():
    """Grouped permutation importance (ranker top-1, decline-model AUC) and top gain features."""
    p = pd.read_csv(os.path.join(DATA, "importance_permutation.csv"))
    g = pd.read_csv(os.path.join(DATA, "importance_gain.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    p = p.sort_values("drop_head_auc")
    y = np.arange(len(p))
    axes[0].barh(y, p.drop_top1_success, color=C["hand"])
    axes[0].set_yticks(y, p.group, fontsize=8)
    axes[0].set_title("ranker: drop in OOF top-1 success\nwhen the group is permuted")
    axes[1].barh(y, p.drop_head_auc, color=C["matched"])
    axes[1].set_yticks(y, [""] * len(p))
    axes[1].set_title("decline model: drop in OOF AUC\nwhen the group is permuted")
    top = g.sort_values("head_gain", ascending=False).head(20).iloc[::-1]
    yy = np.arange(len(top))
    axes[2].barh(yy, top.head_gain / g.head_gain.sum(), color=C["pick"])
    axes[2].set_yticks(yy, [f"{f}  [{fam}]" for f, fam in zip(top.feature, top.family)], fontsize=8)
    axes[2].set_title("decline model: top 20 features by gain share")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    return _save(fig, "fig_importance.png")


def failure_modes():
    """Where the ranker (pick) loses to hand: counts by kind and bin, and the declined-right scores."""
    f = pd.read_csv(os.path.join(DATA, "failures.csv"))
    thr = info["thresholds"]["rules"][PICK]
    kinds = ["declined, top-1 right", "declined, top-1 wrong", "answered alias", "answered wrong"]
    kc = dict(zip(kinds, [C["matched"], C["rank"], C["hand"], "#e87ba4"]))
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.5]))
    for ax, (s, m) in zip(axes[:4], SETS):
        sub = f[(f.set == s) & (f["mode"] == m)]
        lose = sub[sub.side == "ranker loses"]
        wins = sub[sub.side == "ranker wins"].groupby("period_bin").size().reindex(BINS).fillna(0)
        bottom = np.zeros(len(BINS))
        x = np.arange(len(BINS))
        for k in kinds:
            v = lose[lose.kind == k].groupby("period_bin").size().reindex(BINS).fillna(0).to_numpy()
            ax.bar(x - 0.2, v, 0.38, bottom=bottom, color=kc[k], label=f"loss: {k}", edgecolor="white", lw=1)
            bottom += v
        ax.bar(x + 0.2, wins, 0.38, color=C["pick"], label="ranker wins (hand doesn't)")
        ax.set_xticks(x, BINS)
        ax.set_title(f"{s} {m}: {len(lose)} losses, {int(wins.sum())} wins", fontsize=10)
        ax.set_ylabel("stars")
        _style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.4, -0.06))
    ax = axes[4]
    d = f[(f.side == "ranker loses") & (f.kind == "declined, top-1 right")]
    for (s, m), mk in zip(SETS, ("o", "s", "^", "D")):
        q = d[(d.set == s) & (d["mode"] == m)]
        ax.scatter(q.top_period, q.score_v1, marker=mk, s=28, color=C["matched"], edgecolor="white",
                   label=f"{s} {m}")
    edges = [0.1, 1, 3, 10, 30, 400]
    for b, lo, hi in zip(BINS, edges[:-1], edges[1:]):
        ax.hlines(thr[b], lo, hi, color=INK, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("top-1 period (d)")
    ax.set_ylabel("decline-model probability")
    ax.set_title("losses with a right top-1 but declined:\nscore vs per-bin threshold (black)", fontsize=10)
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "fig_failure_modes.png")


def examples():
    """Representative full-year losses: light curve + every candidate's ranker score vs period."""
    e = pd.read_csv(os.path.join(DATA, "examples_candidates.csv"))
    lc = np.load(os.path.join(DATA, "examples_lc.npz"))
    ids = list(dict.fromkeys(e.star_id))
    fig, axes = plt.subplots(len(ids), 2, figsize=(15, 2.8 * len(ids)), gridspec_kw=dict(width_ratios=[2, 1]))
    axes = np.atleast_2d(axes)
    for (a0, a1), sid in zip(axes, ids):
        q = e[e.star_id == sid]
        t, fl = lc[f"{sid}__t"], lc[f"{sid}__f"]
        a0.plot(t, fl, ",", color=INK, alpha=0.6)
        a0.set_title(f"{sid} ({q.set.iloc[0]}, full-year): {q.kind.iloc[0]}; P_true = {q.true_period.iloc[0]:.2f} d",
                     fontsize=9)
        a0.set_xlabel("time (d)")
        _style(a0)
        col = np.where(q.relevance > 0, C["pick"], MUTED)
        a1.scatter(q.P_eval, q.score, c=col, s=30, edgecolor="white", zorder=3)
        sel = q[q.selected.astype(str).str.lower() == "true"]
        a1.scatter(sel.P_eval, sel.score, s=120, facecolor="none", edgecolor=C["hand"], lw=2, label="hand's pick",
                   zorder=4)
        a1.axvline(q.true_period.iloc[0], color=C["pick"], lw=1, ls="--", label="true P")
        a1.set_xscale("log")
        a1.set_xlabel("candidate period (d)")
        a1.set_ylabel("ranker score")
        _style(a1)
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_examples.png")


ALL = [purity_completeness, per_bin, lt1d, tuning, importance, failure_modes, examples]

if __name__ == "__main__":
    for fn in ALL:
        fn()
        print("wrote", fn.__name__)
