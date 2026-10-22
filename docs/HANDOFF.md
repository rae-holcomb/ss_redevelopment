# Handoff note (2026-10-22, end of day)

> **Update 2026-10-22 (new coordinator):** T20 is **merged** (`cd0e5ba`), with the full FFI check done and the flags file regenerated. Running now: scoring-v4 (T22+T23, Sonnet) and T16c (Opus). T24/T25 go to the scoring agent after it finishes; T16b's replay excluding flagged sources is unblocked.

For a **fresh coordinator session** (and a fresh T20 session if needed). The previous coordinator
conversation got very large, which made every turn expensive on Rae's Pro plan, so we're restarting lean.

**Start here:** read `CLAUDE.md`, the tail of the changelog in `docs/PROJECT_CONTEXT.md`, the statuses in
`docs/TASKS.md`, and this note. Don't re-read whole reports unless a task needs them.

## State of `main`
- **Merged:** T0–T4, T9, T14, T15, T16a, T16b, T16e, T18, T21, T1 phase 1, T2, and the D4 decision.
- **Open session:** only **T20**, on branch `claude/t20-galaxy-quietness-c3bb9b`; see below. All other task
  sessions are archived and their worktrees removed.
- **Baselines** (`benchmarks/baselines/README.md`):
  - `synth200_s0` = post-T18 synthetic (success 0.730, wrong 0);
  - `dev_{small,medium}_{assigned,fullyear}` = real SMARTS (T2). dev_medium full-year: success 0.248, wrong 0.088.
- **Data:**
  - dev files are local in the main checkout's `data/smarts_dev/`; worktrees must symlink them (CLAUDE.md);
  - the corpus, catalog, bundle index, galaxy-pass outputs and Zach's provenance files are on the Data_Vault
    drive (`docs/SMARTS_REFERENCE.md` s7).

## Waiting on Rae
1. **T20:** plug in the drive, run the pending FFI check (below), then review and merge T20.
2. **Zach:** were the 242 near-LMC "galaxies" intended? (T20 open question.)
3. Approve the restated goals once the scoring agent proposes them.

## Plan for the next working session (agreed in principle 2026-10-22)
Run **at most two agents at once**:
- **Scoring-update agent:** T22 (band-width breakdown) + T23 (searchable flag/metrics, from D4) + T24
  (`noise_period` outcome) + T25 (quiet-subset metrics).
  - T24 and T25 exist only on the T20 branch until it merges, and need T20's flags, so start with T22 + T23.
  - One `SCORECARD_VERSION` bump, one re-baseline (dev_medium full-year + quiet subset), proposed goals.
  - Small and well specified: a cheaper model (Sonnet) is fine.
- **T16c agent** (pipeline `searchable_range`, beyond-reach tagging, R/2 validation per coverage class,
  `min_valid_frac` test, per-method mechanisms). Touches `coverage.py`, `comb_fit.py` and ML features; no
  overlap with scoring.
- **After T20 merges:** rerun T16b's offline gate replay **excluding flagged noise sources** → decide T16d.
- **Later:** T16h (batch-wise search; needs T16c), T12 (efficiency; after T16c), T19a (first hygiene pass,
  once this round has merged), T17 (small).

## Usage hygiene (new working conventions)
- **Fresh session per task/phase,** with a focused brief; archive finished sessions promptly. Don't let one
  conversation run for days.
- **Agents end their turn only when done or genuinely blocked:** no "still running" status turns. Every
  wake-up of a large session costs a full-context turn.
- Use **Sonnet/Haiku** for small, well-specified tasks; keep Opus for design and analysis.
- **≤ 2 parallel agents** on the Pro plan.
- Avoid dumping long diffs or logs into context; read only the needed sections.

---

## T20 handoff (noise-galaxy quietness)

- **Branch:** `claude/t20-galaxy-quietness-c3bb9b` @ `f30ff7a`.
  - Worktree: `.claude/worktrees/t20-galaxy-quietness-c3bb9b`.
  - Includes `main` up to `cdbd74d`. **Not merged or pushed:** Rae is still working on it.
- **Done:**
  - Galaxy pass over all 180 bundles (outputs on the drive in `extracted/galaxy_pass/`).
  - Ground-truth galaxy IDs from Zach's `galaxy_files.csv`, which is keyed by the original SIM_ID.
  - Amplitude-aware stacking, periodicity search and TIC/Gaia lookups.
  - **Injection is multiplicative.**
  - **206/1,719 noise sources are periodic,** mostly contamination (Gaia variables, crowded/LMC fields). 51 of
    dev_medium's 88 wrong answers equal the noise source's own period.
  - Rae's decisions are recorded on the branch:
    - flags everywhere;
    - exclude contamination + unexplained from ranker training, keep the 8 systematics sources;
    - new tasks T24 `noise_period` and T25 quiet-subset metrics;
    - the flags table carries signal-strength metrics.
  - A 19-source independent-FFI pilot confirmed the period in 9/12 periodic sources and 0/4 quiet controls.
- **Pending (needs the drive):** the full independent-FFI check, covering all 206 periodic sources + 40 quiet
  controls at shuffled periods.
  - About 740 light curves, ~170 MB to the drive (`extracted/ffi_check`).
  - Runs in the **`codeastro`** env (has lightkurve).
  - Command: see the docstring of `tools/audit/ffi_lc_check.py` (selection file prepared in commit `8d9eb04`),
    then `tools/audit/ffi_lc_compare.py`.
  - Then update the phase-2 report, and Rae reviews and merges.
- **If continuing in a fresh session:** work in the existing worktree (same branch), symlink `data/` per
  CLAUDE.md, check the drive is mounted first, and read only the phase-2 report's TL;DR + open questions
  plus this section.
- **On merge:** expect small `docs/` conflicts. Keep both sides; T24/T25 numbering is already reconciled with
  main.
