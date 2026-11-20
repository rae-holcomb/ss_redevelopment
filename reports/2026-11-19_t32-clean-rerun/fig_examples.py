"""Example LC + ACF panels: _v7 (local_screen2) vs a T32 variant, with windows, fitted teeth and vertices.

Usage: python fig_examples.py VARIANT KIND OUT.png star:mode [star:mode ...]
KIND is a title word (gained / lost / newly wrong). Reads the Stage-2 caches.
"""
import pickle
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "claude_files"))
import determinism  # noqa: E402
determinism.pin_blas_threads()
import numpy as np  # noqa: E402
import comb_fit as cf  # noqa: E402
import evaluation as ev  # noqa: E402
from style import plt, C, INK2  # noqa: E402


def run(rec, variant):
    """fit_rotation_period on a cached star -> (result, candidate shown: selected or best declined).

    For a variant that declines a star _v7 answered, main() swaps in the candidate with _v7's seed."""
    res = cf.fit_rotation_period(rec["lags"], rec["acf"], rec["pooled"], acf_counts=rec["counts"],
                                 comb_variant=variant)
    cand = res.candidates[0] if res.candidates else None
    return res, cand


def draw_comb(ax, rec, cand, color, label, yoff):
    """Shade windows, draw fitted teeth, mark vertices (filled = vertex flag passes)."""
    lags = rec["lags"]
    fit = cand.fit
    for w in fit.windows:
        ax.axvspan(w.lag_lo, w.lag_hi, ymin=yoff, ymax=yoff + 0.06, color=color, alpha=0.35, lw=0)
    for n, p in fit.per_peak.items():
        w = next(x for x in fit.windows if x.n == n)
        x = np.linspace(w.lag_lo, w.lag_hi, 60)
        ax.plot(x, p["height"] - p["curvature"] * (x - p["center"]) ** 2, color=color, lw=1.6)
        ax.plot(p["center"], p["height"], "o", ms=6, mfc=color if p["vertex_in_window"] else "none",
                mec=color, mew=1.5)
    ax.plot([], [], color=color, lw=1.6, label=label)


def main():
    """Draw one row per star."""
    warnings.simplefilter("ignore")
    variant, kind, out = sys.argv[1], sys.argv[2], sys.argv[3]
    items = [s.split(":") for s in sys.argv[4:]]
    fig, axes = plt.subplots(len(items), 2, figsize=(13, 2.5 * len(items)), gridspec_kw=dict(width_ratios=[1, 1.6]))
    axes = np.atleast_2d(axes)
    for (sid, mode), (a0, a1) in zip(items, axes):
        with open(ROOT / f"runs/t32_clean/cache_{mode}/{sid}.pkl", "rb") as fh:
            rec = pickle.load(fh)
        Pt = rec["true_period"]
        a0.plot(rec["time"], rec["flux"], ",", color=INK2)
        a0.set_title(f"{sid[-6:]} ({'full-year' if mode == 'fy' else 'assigned'}), P_true = {Pt:.3g} d", fontsize=8)
        a0.set_xlabel("time [d]", fontsize=7)
        lags, acf = rec["lags"], rec["acf"]
        a1.plot(lags, acf, color="#8a8984", lw=0.8)
        txt = []
        for k, (v, col, lab) in enumerate([("local_screen2", C[1], "v7 (local_screen2)"), (variant, C[0], variant)]):
            res, cand = run(rec, v)
            if k == 0:
                p0_v7 = cand.period if (res.success and cand is not None) else None
            elif not res.success and p0_v7 is not None:
                # declined: show the candidate seeded like _v7's answer, so its failing gates are visible
                same = [r for r in res.candidates if abs(r.period - p0_v7) < 1e-9]
                cand = same[0] if same else cand
            if cand is not None and cand.fit is not None:
                draw_comb(a1, rec, cand, col, lab, 0.02 + 0.07 * k)
            P = res.best_fit.P if res.success else None
            o = ev.classify_period(P, Pt, ev.REL_TOL, rec["prange"])
            fg = "" if res.success else ",".join(g[:12] for g in (cand.failed_gates if cand else []))
            txt.append(f"{lab}: {o}" + (f" P={P:.3g}" if P else f" (cand. P={cand.fit.P:.3g} fails {fg})"
                                          if cand and cand.fit else ""))
        top = max(cand.fit.P if cand and cand.fit else Pt, Pt) * 8.5
        for j in range(1, 9):
            a1.axvline(j * Pt, color=C[2], lw=0.8, ls=":")
        a1.set_xlim(0, min(lags[-1], top))
        a1.text(0.99, 0.97, "\n".join(txt), transform=a1.transAxes, ha="right", va="top", fontsize=7)
        a1.set_xlabel("lag [d]  (dotted green: n·P_true; filled vertex = vertex flag passes)", fontsize=7)
    axes[0, 1].legend(loc="upper center", fontsize=7, ncol=2)
    fig.suptitle(f"{kind}: _v7 vs {variant}", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / out)


if __name__ == "__main__":
    main()
