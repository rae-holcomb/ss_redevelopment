"""Check the 1M-row per-file table: keys, uniqueness, consistency with the per-source table, and the
file->noise mapping rebuilt independently from Zach's simulation_properties_sorted.csv + galaxy_files.csv."""
import numpy as np, pandas as pd
A = "/Users/raeholcomb/Documents/GitHub/ss_redevelopment/.claude/worktrees/smarts-contamination-note-969cc1"
D = "/Volumes/Data_Vault/smarts_data"
F = pd.read_csv(f"{D}/extracted/release_note/smarts_noise_source_flags.csv.gz")
S = pd.read_csv(f"{A}/docs/smarts_release_note/smarts_noise_sources.csv")
print("rows", len(F), "cols", list(F.columns))
print("file_number 0..999999 unique:", F.file_number.is_unique, F.file_number.min(), F.file_number.max(),
      "sim_id unique", F.sim_id.is_unique, "filename ok",
      (F.filename == F.file_number.map(lambda r: f"smarts-tess-v1.0-{r:06d}.fits")).all())
print("bool dtypes:", {c: str(F[c].dtype) for c in F.columns if F[c].dtype == bool or F[c].dtype == object})
# per-file vs per-source consistency
cols = [c for c in F.columns if c in S.columns and c != "noise_tic"]
M = F.drop_duplicates("noise_tic").merge(S, on="noise_tic", suffixes=("_f", "_s"))
bad = {}
for c in cols:
    a, b = M[c + "_f"], M[c + "_s"]
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        ok = np.isclose(a.astype(float), b.astype(float), rtol=1e-5, equal_nan=True)
    else:
        ok = (a.astype(str) == b.astype(str))
    bad[c] = int((~ok).sum())
print("mismatches per column (unique sources):", bad)
cnt = F.noise_tic.value_counts()
print("n_files consistent:", (S.set_index("noise_tic").n_files == cnt.reindex(S.noise_tic).values).all())
print("files by class:", F.classification.value_counts().to_dict())
# Independent mapping from Zach's files
sp = pd.read_csv(f"{D}/simulation_properties_sorted.csv", usecols=["Published Simulation Number", "Original Simulation Number"])
gf = pd.read_csv(f"{D}/galaxy_files.csv")
print("galaxy_files cols", list(gf.columns), len(gf))
key = [c for c in gf.columns if "Simulation" in c][0]; fcol = [c for c in gf.columns if c != key][0]
m = F.merge(sp, left_on="file_number", right_on="Published Simulation Number").merge(gf, left_on="Original Simulation Number", right_on=key)
print("sim_id == Original:", (m.sim_id == m["Original Simulation Number"]).all(), len(m))
tic_from_gf = m[fcol].str.extract(r"TIC(\d+)")[0].astype(np.int64)
print("noise_tic == galaxy_files TIC:", (tic_from_gf == m.noise_tic).mean())
names = set(gf[fcol].unique()); print("noise_lc_file in galaxy_files names:", S.noise_lc_file.isin(names).mean(), "example", next(iter(names)))
# what a user would get if they joined galaxy_files on file_number (the trap the reference warns about)
w = F.merge(gf, left_on="file_number", right_on=key)
print("WRONG join on file_number would agree on:", (w[fcol].str.extract(r"TIC(\d+)")[0].astype(np.int64) == w.noise_tic).mean())
