"""P59 example combs (dev_medium_clean, judging set; diagnosis only): the selected candidate under the _v9
default (local_tol_trim_l0x) and under local_tol_trim_l0x2, for two stars l0x2 gains and two it newly gets
wrong (one per mode each, the first by star_id). Writes data/examples.pkl and figures/fig_examples.png."""
import pickle, sys, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports" / "2026-11-20_p44-lag0-out"))
import p44lib as P  # noqa: E402
L, C, np = P.L, P.C, P.np
import pandas as pd  # noqa: E402
import evaluation as ev  # noqa: E402
S = {"correct", "diffrot"}
VS = ("local_tol_trim_l0x", "local_tol_trim_l0x2")


def pick():
    out = []
    for mode, tag in (("fullyear", "dmfy"), ("assigned", "dmas")):
        o = pd.read_csv(ROOT / f"benchmarks/baselines/dev_medium_clean_{mode}_v9_tables/per_star.csv").set_index("star_id")
        n = pd.read_csv(HERE / f"data/per_star/local_tol_trim_l0x2_{tag}.csv.gz").set_index("star_id")
        a, b = o.final_outcome, n.final_outcome.reindex(o.index)
        out.append((mode, "gained", sorted(o.index[b.isin(S) & ~a.isin(S)])[0]))
        out.append((mode, "new wrong", sorted(o.index[(b == "wrong") & (a != "wrong")])[0]))
    return out


def collect():
    warnings.simplefilter("ignore")
    recs = []
    for mode, why, sid in pick():
        t = [t for t in L.make_tasks(mode, "datasets/dev_medium_clean.csv") if t["star_id"] == sid][0]
        st = L.load_star(t)
        pooled = L.pooled_guesses(st)
        rec = dict(mode=mode, why=why, star_id=sid, true_p=st["true_p"], lags=st["lags"], acf=st["acf"], fits={})
        for v in VS:
            res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
            r = res.candidates[0]
            f = r.fit
            rec["fits"][v] = dict(
                P=f.P, success=res.success, outcome=ev.classify_period(f.P if res.success else None, st["true_p"],
                                                                       ev.REL_TOL, st["prange"]),
                failed=r.failed_gates, n_required=r.n_peaks_required, few=r.few_peaks,
                gates={k: g["value"] for k, g in r.gate_results.items() if g.get("enabled", True)},
                teeth=[dict(n=n, center=pk["center"], height=pk["height"], curvature=pk["curvature"],
                            lo=float(w.lag_lo), hi=float(w.lag_hi), lag0=C._is_lag0_tooth(pk["center"], f.P))
                       for (n, pk), w in zip(sorted(f.per_peak.items()), sorted(f.windows, key=lambda w: w.n))])
        recs.append(rec)
    pickle.dump(recs, open(HERE / "data/examples.pkl", "wb"))
    return recs


def plot(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(recs), 2, figsize=(12, 2.9 * len(recs)))
    for i, rec in enumerate(recs):
        for j, v in enumerate(VS):
            ax, f = axes[i, j], rec["fits"][v]
            xmax = min(rec["lags"][-1], max(1.5 * max(t["hi"] for t in f["teeth"]), 3 * rec["true_p"]))
            m = rec["lags"] <= xmax
            ax.plot(rec["lags"][m], rec["acf"][m], color="0.55", lw=0.8)
            for t in f["teeth"]:
                xx = np.linspace(t["lo"], t["hi"], 60)
                ax.plot(xx, t["height"] - t["curvature"] * (xx - t["center"]) ** 2,
                        color="#9467bd" if t["lag0"] else "#1f77b4", lw=1.6)
                ax.text(t["center"], t["height"] + 0.04, f"{t['height']:+.2f}", ha="center", fontsize=6.5)
            for k in range(1, int(xmax / rec["true_p"]) + 1):
                ax.axvline(k * rec["true_p"], color="orange", lw=0.6, ls=":")
            ax.set_xlim(0, xmax)
            g = f["gates"]
            ax.set_title(f"{rec['star_id'][-6:]} {rec['mode']} ({rec['why']} under l0x2) P_true={rec['true_p']:.2f} | "
                         f"{v.replace('local_tol_trim_', '')}: top P={f['P']:.2f} -> {f['outcome']}\n"
                         f"n_peaks {g['n_peaks']}/{f['n_required']} req, snr {g['mean_height_snr']:.2f}, "
                         f"failed: {', '.join(f['failed']) or 'none'}", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("lag (d)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    plot(collect(), HERE / "figures/fig_examples.png")
