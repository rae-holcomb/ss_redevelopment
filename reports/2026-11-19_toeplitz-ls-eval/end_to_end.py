#!/usr/bin/env python3
"""end_to_end.py -- share of per-star time in Lomb-Scargle, and the saving if LS used Toeplitz-LS (P50).

Inputs (data/): profile_{assigned,fullyear}.csv from benchmarks/profile_pipeline.py (5 stars x 3 repeats,
serial), timing.csv from timing.py (real-star grids). Output: data/end_to_end.csv.

Per mode, two paths:
  production  load + acf_long + acf_short + the 3 default guess methods (pairwise_histogram, lombscargle,
              acf_fft) + fit  -- LS slice = guess_lombscargle
  scorecard   every stage profile_pipeline times (all guess_* methods) -- LS slice = guess_lombscargle +
              guess_lombscargle_short
Saving = LS slice x (1 - median[backend / astropy_fast], backend = tls_native or tlsf_native, on that mode's real grids, per grid type).
The LS stage time includes peak picking and InitialGuess building, which Toeplitz-LS would not speed up,
so applying the periodogram-only ratio to the whole stage slightly overstates the saving.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
PROD = ["load", "acf_long", "acf_short", "guess_pairwise_histogram", "guess_lombscargle", "guess_acf_fft", "fit"]


def main():
    tm = pd.read_csv(DATA / "timing.csv").query("exp == 'real'")
    piv = tm.pivot_table(index=["star_id", "mode", "grid"], columns="backend", values="seconds").reset_index()
    rows = []
    for mode in ("assigned", "fullyear"):
        prof = pd.read_csv(DATA / f"profile_{mode}.csv")
        med = prof.groupby("star_id").median(numeric_only=True).mean()  # mean over stars of per-star medians
        for path in ("production", "scorecard"):
            total = med[PROD].sum() if path == "production" else med["total"]
            parts = {"lombscargle": med["guess_lombscargle"]}
            if path == "scorecard":
                parts["lombscargle_short"] = med["guess_lombscargle_short"]
            ls = sum(parts.values())
            row = dict(label=f"{mode}, {path}", mode=mode, path=path, total_ms=1e3 * total, ls_ms=1e3 * ls,
                       ls_share=ls / total)
            for b in ("tls_native", "tlsf_native"):
                r = {g: (piv[b] / piv.astropy_fast)[(piv["mode"] == mode) & (piv.grid == g)].median() for g in parts}
                new_ls = sum(parts[g] * r[g] for g in parts)
                tag = b.split("_")[0]
                row[f"ls_ms_{tag}"] = 1e3 * new_ls
                row[f"saved_frac_{tag}"] = (ls - new_ls) / total
                for g in parts:
                    row[f"ratio_{tag}_{g}"] = r[g]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "end_to_end.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
