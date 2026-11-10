"""Figures and extra statistics for the quiet-noise-periods report, from the tables in data/ only.

Run from the repo root: ``$PY reports/2026-11-10_quiet-noise-periods/make_figures.py``.
Writes figures/*.png and data/extra_stats.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).resolve().parent
D, F = R / "data", R / "figures"
F.mkdir(exist_ok=True)
REL_TOL = 0.10


def near_any(P, refs, tol):
    """True where P is within fractional tol of any value in refs."""
    P = np.asarray(P, float)[:, None]
    r = np.asarray(refs, float)[None, :]
    return (np.abs(P / r - 1) < tol).any(axis=1)


def main():
    """Make every figure and write extra_stats.json."""
    summ = json.loads((D / "summary.json").read_text())
    s = pd.read_csv(D / "quiet_sources.csv")
    q = s[s.classification == "quiet"].copy()
    w = pd.read_csv(D / "wrong_stars.csv")
    per = pd.read_csv(D / "dev_medium_fullyear_v5_per_star.csv")
    cmp = pd.read_csv(D / "common_mode_peaks.csv")
    nb = pd.read_csv(D / "gaia_neighbour_periods.csv")
    stats = {}

    # 1. shuffled-source null
    null = np.load(D / "null_counts.npy")
    obs = summ["null"]["quiet_after_reclass"]["observed"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(null, bins=np.arange(null.max() + 2) - 0.5, color="0.6", label="shuffled-source null (5000)")
    ax.axvline(obs, color="C3", lw=2, label=f"observed = {obs}")
    ax.set(xlabel="quiet-class wrong answers within 10% of P_ref", ylabel="draws",
           title="dev_medium full-year, 30 quiet-class wrongs (after reclass)")
    ax.legend()
    fig.tight_layout(); fig.savefig(F / "null_test.png", dpi=120); plt.close(fig)

    # 2. time-split S/N vs P_ref, population
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = s[s.classification == "periodic_low_amplitude"]
    ax.scatter(q.P_ref, q.tsplit_min_snr, s=5, c="0.6", label=f"quiet (n={len(q)})")
    ax.scatter(lo.P_ref, lo.tsplit_min_snr, s=12, c="C0", label=f"periodic_low_amplitude (n={len(lo)})")
    m = s[s.matched_star_source]
    ax.scatter(m.P_ref, m.tsplit_min_snr, s=60, facecolor="none", edgecolor="C3", lw=1.5,
               label="sources behind dev_medium noise-period wrongs")
    ax.axhline(20, color="C0", ls="--", lw=1); ax.axvline(5, color="C0", ls="--", lw=1)
    ax.set(xscale="log", yscale="log", xlabel="P_ref (d)", ylabel="min over halves of local LS S/N at P_ref",
           title="Time-split coherence: are P_ref peaks present in both halves of the year?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "time_split_population.png", dpi=120); plt.close(fig)
    q["bin"] = pd.cut(q.P_ref, [0.1, 1.5, 5, 10, 20, 61])
    stats["tsplit_quiet_by_bin"] = q.groupby("bin", observed=True).tsplit_min_snr.describe(
        percentiles=[.5, .95, .99]).round(2).reset_index().astype(str).to_dict("records")
    stats["matched_tsplit_percentile_in_bin"] = {
        str(r.tic): float((q[q.bin == r.bin].tsplit_min_snr < r.tsplit_min_snr).mean())
        for _, r in q[q.matched_star_source].iterrows()}

    # 3. matched-source half-year periodograms
    z = np.load(D / "matched_power.npz")
    n = len(z["tic"])
    fig, axs = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    for i, ax in enumerate(axs):
        P = 1 / z["freq"]
        ax.plot(P, z["half_power"][i, 0], lw=0.7, label="t < 182.5 d")
        ax.plot(P, z["half_power"][i, 1], lw=0.7, label="t >= 182.5 d")
        ax.plot(P, z["full_power"][i], lw=0.7, c="k", alpha=0.5, label="full year")
        ax.axvline(z["P_ref"][i], c="C3", ls="--", lw=1)
        for a in w[w.noise_tic == z["tic"][i]].final_period:
            ax.axvline(a, c="C2", ls=":", lw=1.5)
        ax.set_xscale("log"); ax.set_xlim(0.1, 60)
        cls = s.set_index("tic").classification[z["tic"][i]]
        ax.text(0.01, 0.8, f"TIC {z['tic'][i]}  P_ref={z['P_ref'][i]:.2f} d  ({cls})", transform=ax.transAxes, fontsize=7)
    axs[0].legend(fontsize=6, loc="upper right"); axs[-1].set_xlabel("period (d)")
    fig.suptitle("Sources behind the 12 quiet-class noise-period wrongs: red = P_ref, green = star's answer", fontsize=9)
    fig.tight_layout(); fig.savefig(F / "matched_periodograms.png", dpi=110); plt.close(fig)

    # 4. P_ref distribution vs TESS timescales and common modes
    pcs = cmp[["P1", "P2", "P3"]].values.ravel()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    bins = np.logspace(np.log10(0.1), np.log10(60), 120)
    a1.hist(q.P_ref, bins=bins, color="0.5")
    for v in (13.7, 27.4):
        a1.axvline(v, c="C1", lw=1.5)
    for v in cmp.P1:
        a1.axvline(v, c="C4", ls=":", lw=1)
    a1.set(ylabel="quiet sources", title="Quiet P_ref (orange: TESS orbit 13.7/27.4 d; purple: common-mode top peaks)")
    a2.scatter(q.P_ref, q.cm_retained, s=5, c="0.6")
    a2.scatter(m.P_ref, m.cm_retained, s=50, facecolor="none", edgecolor="C3")
    a2.axhline(0.5, ls="--", c="k", lw=1)
    a2.set(xscale="log", yscale="log", xlabel="P_ref (d)", ylabel="LS power retained after\nremoving 8 common modes")
    fig.tight_layout(); fig.savefig(F / "pref_systematics.png", dpi=120); plt.close(fig)
    # TESS-orbit / common-mode excess vs log-shifted reference sets
    long = q[q.P_ref >= 5].P_ref.values
    def excess(refs, tol=0.03):
        """Observed count near refs vs the mean over 40 log-shifted copies of refs (+-30%)."""
        o = int(near_any(long, refs, tol).sum())
        sh = [int(near_any(long, np.asarray(refs) * f, tol).sum()) for f in np.exp(np.linspace(-0.3, 0.3, 41)) if abs(np.log(f)) > 0.05]
        return dict(observed=o, shifted_mean=float(np.mean(sh)), shifted_max=int(np.max(sh)), n=len(long))
    stats["near_tess_orbit"] = excess([13.7, 27.4])
    stats["near_common_mode_peaks"] = excess(pcs)
    stats["cm_retained_lt_0.5"] = dict(quiet_ge5=float((q[q.P_ref >= 5].cm_retained < 0.5).mean()),
                                       matched=int((m.cm_retained < 0.5).sum()), n_matched=len(m))

    # 5. why wrong answers land on P_ref: injected amplitude vs noise
    qw = w[w.class_new == "quiet"]
    qs = per.merge(s[["tic", "classification", "amp1_ppm_all"]].rename(columns={"tic": "noise_tic"}), on="noise_tic")
    qs = qs[qs.classification == "quiet"]
    succ = qs[qs.final_outcome.isin(["correct", "diffrot"])]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for d_, lab, c in ((succ, "quiet: success", "C0"), (qw[~qw.noise_match], "quiet: wrong, other", "0.4"),
                       (qw[qw.noise_match], "quiet: wrong = noise P_ref", "C3")):
        ax.scatter(d_.true_period, d_.flux_amp, s=14 if c != "C3" else 40, c=c, label=f"{lab} (n={len(d_)})", alpha=0.7)
    ax.set(xscale="log", yscale="log", xlabel="true (injected) period (d)", ylabel="injected flux_amp",
           title="Noise-period wrongs are weak injected signals")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "amplitude_context.png", dpi=120); plt.close(fig)
    stats["flux_amp_median"] = dict(success=float(succ.flux_amp.median()),
                                    wrong_other=float(qw[~qw.noise_match].flux_amp.median()),
                                    wrong_noise=float(qw[qw.noise_match].flux_amp.median()))

    # 6. Gaia neighbour period matches: shuffled null over quiet sources with a neighbour period
    rng = np.random.default_rng(0)
    nq = nb[nb.tic.isin(q.tic)].copy()
    def nmatch(pref):
        r = nq.P_nb / nq.tic.map(pref)
        ok = (np.abs(r - 1) < .03) | (np.abs(r - .5) < .015) | (np.abs(r - 2) < .06)
        return nq.tic[ok].nunique()
    has = q[q.tic.isin(nq.tic)].set_index("tic").P_ref
    nul = [nmatch(pd.Series(rng.permutation(has.values), index=has.index)) for _ in range(500)]
    stats["gaia_nb"] = dict(n_with_period=len(has), observed=int(nmatch(has)), null_mean=float(np.mean(nul)),
                            null_p95=float(np.percentile(nul, 95)),
                            matched_tics=q[q.gaia_nb_match].tic.tolist())
    stats["near_lmc"] = dict(quiet=float(q.near_lmc.mean()), matched=int(m.near_lmc.sum()))
    stats["member_injection_match"] = dict(mean=float(q.frac_member_P_match.mean()), null=float(q.frac_member_P_match_null.mean()))
    stats["ffi_tested_quiet"] = int(q.ffi_tested.sum())
    stats["ffi_confirmed_quiet"] = int(q.ffi_confirmed.sum())
    (D / "extra_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
