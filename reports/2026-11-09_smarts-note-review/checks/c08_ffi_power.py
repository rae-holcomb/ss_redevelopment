"""Does FFI confirmation depend on signal amplitude / target brightness, and do the classes differ in them?
Tests the 'too faint in other reductions' explanation for source_specific_unexplained."""
import numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
T = f"{A}/reports/2026-10-21_t20-noise-sources-phase2/data"
f = pd.read_csv(f"{T}/ffi_compare_all.csv")
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv").set_index("noise_tic")
f = f.join(S[["Tmag", "near_lmc"]], on="tic")
t = f[f.testable & ~f.control]
print("Tmag all sources median", S.Tmag.median(), "range", S.Tmag.min(), S.Tmag.max(), "frac Tmag>15", (S.Tmag > 15).mean())
print(t.groupby("classification").agg(n=("tic", "size"), conf=("confirmed", "mean"), amp_med=("smarts_amp_ppm", "median"),
      Tmag_med=("Tmag", "median"), P_med=("P_ref", "median")).round(2).to_string())
t["amp_bin"] = pd.cut(t.smarts_amp_ppm, [0, 300, 600, 1500, 1e6])
print("confirmation by SMARTS amplitude bin (all classes):"); print(t.groupby("amp_bin", observed=True).confirmed.agg(["size", "mean"]).round(2).to_string())
print("by amp bin x class:"); print(t.groupby(["amp_bin", "classification"], observed=True).confirmed.agg(["size", "sum"]).unstack(fill_value=0).to_string())
lo = t[t.smarts_amp_ppm < 1500]
print("amp<1500 ppm: known_var", lo[lo.classification == "contamination_known_variable"].confirmed.agg(["sum", "size"]).tolist(),
      "unexplained", lo[lo.classification == "source_specific_unexplained"].confirmed.agg(["sum", "size"]).tolist())
t["Tbin"] = pd.cut(t.Tmag, [0, 13, 14, 15, 20])
print("by Tmag:"); print(t.groupby("Tbin", observed=True).confirmed.agg(["size", "mean"]).round(2).to_string())
# 12,244 vs 336 example
print(f[f.tic == 10000584619][["classification", "smarts_amp_ppm", "ffi_amp_at_P_ppm", "authors", "confirmed"]].to_string())
print("ratio FFI amp / SMARTS amp among confirmed, median by class:",
      (t[t.confirmed].assign(r=lambda d: d.ffi_amp_at_P_ppm / d.smarts_amp_ppm).groupby("classification").r.median().round(2).to_dict()))
