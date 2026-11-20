"""P44 example stars (dev_medium_clean, judging set; diagnosis only): the closest correct candidate's comb fit
under local_tol_trim (_v8) and local_tol_trim_l0x. Writes data/examples.pkl (ACF + per-tooth fits) and
figures/fig_examples.png. One recovered star and three of the P43 lag-0 losses that stay lost."""
import pickle, sys, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p44lib as P  # noqa: E402
L, C, np = P.L, P.C, P.np
import evaluation as ev  # noqa: E402
S = {"correct", "diffrot"}
STARS = [("fullyear", "smarts-tess-v1.0-046873"), ("fullyear", "smarts-tess-v1.0-125098"),
         ("assigned", "smarts-tess-v1.0-010328"), ("assigned", "smarts-tess-v1.0-028773")]


def collect():
    warnings.simplefilter("ignore")
    out = []
    for mode, sid in STARS:
        t = [t for t in L.make_tasks(mode, "datasets/dev_medium_clean.csv") if t["star_id"] == sid][0]
        st = L.load_star(t)
        pooled = L.pooled_guesses(st)
        rec = dict(mode=mode, star_id=sid, true_p=st["true_p"], lags=st["lags"], acf=st["acf"], fits={})
        for v in ("local_tol_trim", "local_tol_trim_l0x"):
            res = C.fit_rotation_period(st["lags"], st["acf"], pooled, acf_counts=st["counts"], comb_variant=v)
            sel = ev.classify_period(res.best_fit.P, st["true_p"], ev.REL_TOL, st["prange"]) if res.success else "none"
            for r in res.candidates:
                if ev.classify_period(r.fit.P, st["true_p"], ev.REL_TOL, st["prange"]) in S:
                    f = r.fit
                    teeth = [dict(n=n, center=pk["center"], height=pk["height"], curvature=pk["curvature"],
                                  lo=float(w.lag_lo), hi=float(w.lag_hi), lag0=C._is_lag0_tooth(pk["center"], f.P))
                             for (n, pk), w in zip(sorted(f.per_peak.items()), sorted(f.windows, key=lambda w: w.n))]
                    rec["fits"][v] = dict(P=f.P, teeth=teeth, dropped=f.n_peaks_dropped, failed=r.failed_gates,
                                          snr=r.diagnostics["mean_height_snr"], acf_std=float(np.std(st["acf"])),
                                          star_outcome=sel)
                    break
        out.append(rec)
    pickle.dump(out, open(HERE / "data/examples.pkl", "wb"))
    return out


def plot(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(recs), 2, figsize=(12, 2.8 * len(recs)))
    for i, rec in enumerate(recs):
        for j, (v, lab) in enumerate((("local_tol_trim", "_v8 local_tol_trim"), ("local_tol_trim_l0x", "l0x"))):
            ax = axes[i, j]
            f = rec["fits"][v]
            xmax = min(rec["lags"][-1], (max(t["n"] for t in f["teeth"]) + 1.5) * f["P"])
            m = rec["lags"] <= xmax
            ax.plot(rec["lags"][m], rec["acf"][m], color="0.55", lw=0.8)
            for t in f["teeth"]:
                xx = np.linspace(t["lo"], t["hi"], 60)
                ax.plot(xx, t["height"] - t["curvature"] * (xx - t["center"]) ** 2,
                        color="#9467bd" if t["lag0"] else "#1f77b4", lw=1.6)
                ax.text(t["center"], t["height"] + 0.04, f"{t['height']:+.2f}", ha="center", fontsize=6.5)
            ax.axhline(f["acf_std"], color="#d62728", ls=":", lw=0.8)
            ax.axhline(1.02 * f["acf_std"], color="#d62728", ls="--", lw=0.6)
            for k in range(1, int(xmax / rec["true_p"]) + 1):
                ax.axvline(k * rec["true_p"], color="orange", lw=0.6, ls=":")
            ax.set_xlim(0, xmax)
            ax.set_ylim(min(-0.3, rec["acf"][m].min() - 0.05), 1.05)
            ax.set_title(f"{rec['star_id'][-6:]} {rec['mode']} P_true={rec['true_p']:.2f} | {lab}: P={f['P']:.2f}, "
                         f"dropped {f['dropped']}, snr={f['snr']:.2f} -> star {f['star_outcome']}\n"
                         f"failed: {', '.join(f['failed']) or 'none'}", fontsize=7.5)
    axes[-1, 0].set_xlabel("lag (d)")
    axes[-1, 1].set_xlabel("lag (d)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    plot(collect(), HERE / "figures/fig_examples.png")
