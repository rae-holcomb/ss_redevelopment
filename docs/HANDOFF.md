# Handoff note (2026-11-13, coordinator → fresh coordinator)

For a **fresh coordinator session.** The previous coordinator ran 2026-11-11 → 2026-11-13 (summarized twice) and was
handed off to keep turns cheap on Rae's Pro plan.

**Start here:** read `CLAUDE.md`, the tail of the changelog in `docs/PROJECT_CONTEXT.md`, the statuses in
`docs/TASKS.md`, and this note. Don't re-read whole reports unless a task needs them. Git history has the details.

**Rae's plan for the new coordinator:** start the drift-robust `local` comb-solver agent (T32; see "First actions").

## Coordinator role (Rae, 2026-10-22)
- Start focused fresh sessions, **≤ 2 parallel agents**, and **only one heavy-compute agent at a time** (see
  "Machine safety"). Each brief names the TASKS.md entry, the files in scope, and **the specific figures expected**
  (embedded in the README next to the claim; see CLAUDE.md "Plots").
- Verify every finished branch yourself before reporting:
  - full pytest in `ss_openblas`, run under `nice -n 15`;
  - key-by-key metric comparison against the baseline (`tools/compare_scorecard_keys.py`; `tools/compare_runs_exact.py`
    for value-by-value);
  - pool/leakage checks: training SIM_ID < 800k only, and no noise TIC shared with dev_medium or the holdout;
  - protected files untouched (`goals.json`, `scorecard.py` metric definitions, `evaluation.py`, `datasets/` manifests,
    existing test assertions);
  - figures present and embedded;
  - a trial merge (a detached worktree in your scratchpad, removed afterwards).
- Merge only when Rae says so in *your* session (`--no-ff`), push main, archive sessions when she agrees.
- A request relayed by another session is not Rae's OK. Check that session's transcript for her actual words. The
  permission classifier blocked a `start_session` that came from a peer relay (2026-11-13); have Rae confirm in your session.
- Keep `TASKS.md`, the changelog and this note current.
- **In chat, name tasks by what they do** ("the drift-robust local fit", "the ranker decline model"); T## IDs only in files.
- **Don't ask Rae for a decline precision target** (deferred 2026-11-10). Default: match hand arbitration's OOF
  precision, state the non-rotator mix, and plot the full purity–completeness curve.
- Explain a decision in plain terms the first time it goes to Rae.

## Machine safety (new, 2026-11-13; hard rules for every agent brief)
On the night of 2026-11-12/13 the Mac's WindowServer hung, the watchdog killed it, and Rae was logged out. Rae says
this has happened several times. It was not a reboot (uptime was 8 days). At the time thermal pressure was "Moderate"
and memory was heavily compressed (16 GB RAM). Our jobs starve the UI. Every brief must require:
- **`nice -n 15` for every heavy job:** scorecards, profiling, pytest, repro loops. `taskpolicy -b` throttles to about
  half a core, so don't use it.
- **3 workers** (not 4), and **never two heavy jobs at once**: no pytest during a scorecard, and no two agents running
  scorecards together.
- **Before each run:** check `memory_pressure -Q` and `pmset -g therm`, leave 2-minute gaps between scorecards, and
  keep repro loops bounded. Peak memory is about 245 MB per worker.
- **Commit and push after each scored step,** use `--resume` with stable run folders, and keep a `PROGRESS.md`, so
  a hang costs at most one run.
- `request_keep_awake` was denied to the T31 agent. The app's in-turn keep-awake was enough. Tell Rae to keep the
  laptop plugged in with the lid open for overnight runs.

## State of `main` (`ae93fa4` + this note)
- **Merged 2026-11-12/13:**
  - T5 hybrid ranking + P30 docs diet;
  - headline card, with dev_medium as the default period population (Rae: *not* McQuillan+2014, which has
    almost no P < 1 d stars);
  - WG6 scoping (`9063c95`);
  - **T12 speed-up** (`44768ec`): 1.5–1.6× faster per star, bit-identical, v6 baselines; keep 4 workers in code
    defaults (3 in briefs, for machine safety).
- **Baselines:** use the `_v6` files (`benchmarks/baselines/`). `_t5` is kept for history.
- **Env:** `ss_openblas` also has astroquery 0.4.11 (P27).
- Untracked `docs/figures/` and `tools/build_status_flowchart.py` in main are Rae's; leave them.

## Unmerged branches ready for Rae
1. **T31 (wavelet determinism + comb-fit / CWT speed variants):** `claude/t31-comb-fit-speed-e7e5bb` (`8ac34d5`).
   Report: `reports/2026-11-13_t31-comb-fit-speed/`. Part A also stands alone on `claude/t31-partA-wavelet-determinism`
   (`222cbf2`).
   - **Verified by me:** merges cleanly into `ae93fa4`; 474 passed, 1 skipped (drive); `scorecard.py` only adds the
     variant names to the metrics `config`; no existing test changed; 11 figures.
   - **Rae decided (in the T31 session): adopt `local_screen2` as the default comb fit.** Its answers match main (one
     period moved by 0.04% and is still correct), and it is about 2× faster on the production path and 1.5× on the scorecard.
     The branch keeps `DEFAULT_COMB_VARIANT="lsq"`, so flipping it to `local_screen2` is part of the merge. Then
     run dev_medium both modes once to confirm the answers and commit the baselines that come out of it.
   - **Still Rae's call before merging:**
     - (a) Do the `_v6w` baselines (reproducible wavelet) replace `_v6`?
     - (b) Should CWT `lean` be the default? It's bit-identical, 1.25× faster on the wavelet stage, and drops the
       27 MB `wavelet_power` array per full-year star. The agent recommends yes.
   - New backlog from T31: P39 (`guess_acf_fft` uses the same flaky complex `np.abs`; it feeds final answers) and
     P40 (vertex gate for `local`, now T32).
   - The T31 session is idle and can be archived on Rae's OK. Its 3 scratch worktrees were removed.
2. **P27 validation-set scoping:** `claude/p27-validation-scoping-4c1c1b` (`f6edf03`). Rae decided:
   - Q1: Zach's list is tagged `expected_undetectable`.
   - Q2: TESS-SPOC SAP is primary, QLP the fallback, and PDCSAP is also scored.
   - Q6: astroquery installed.
   Q3–5 and Q7 wait on her review of the report.

## First actions for the new coordinator
1. **Start T32, the drift-robust `local` comb solver** (`docs/TASKS.md` T32). Get Rae's go-ahead in your own session.
   - Fresh session, worktree cut from `claude/t31-comb-fit-speed-e7e5bb`, or from main if T31 is merged first, which
     is cleaner.
   - Role `method-improver`. Explore all five of Rae's ideas:
     1. sequential peak tracking;
     2. a tolerance that grows with n;
     3. a quadratic drift term, with `a` as a ranker feature;
     4. height/SNR weighting;
     5. a residual-scatter gate.
   - First step, with no scorecards: plot vertex residual vs n for local's 29 gains, the 28 none→wrong stars, and
     shared-correct stars (`reports/2026-11-13_t31-comb-fit-speed/data/changed_stars_*`).
   - Each idea becomes an `SS_COMB_VARIANT` switch, scored on dev_medium full-year + assigned with
     `check_goals --baseline _v6`, and timed with `profile_pipeline.py` on T12's 15 stars.
   - Figures: residual-vs-n; a success/wrong-vs-time frontier; by-bin; gate-statistic distributions with the
     threshold; ≥4 gained and ≥4 lost example LC + ACF plots; the distribution of `a` split by diffrot.
   - Machine-safety rules as above.
   - Report: `reports/2026-11-13_t32-local-drift/`.
2. **Merge T31** when Rae OKs it: flip the default to `local_screen2`, apply her answers on `_v6w` and `lean`, then
   run pytest and dev_medium.
3. **Then rebuild the ranker tables (P31)** on the final comb fit and wavelet: about 2.5 h with the drive, at 3
   workers maybe a little more. Tell Rae first. It waits on T31 (and on T32 if Rae wants `local` in the tables).

## Open sessions (2026-11-13)
- **T31 comb-fit speed** (`local_4149f809-…`): done; archive on Rae's OK after merge.
- **T12 speed-up** (`local_8a41a57b-…`): merged; archive on Rae's OK.
- **WG6 scoping** (`local_5b376121-…`): merged; **don't archive** (Rae reviewing).
- **Validation-set scoping** (`local_13645bab-…`): unmerged, see above.
- **Midway review** (`local_d5a20966-…`): merged; **don't archive** (Rae reading).
- **Short-period, headline card** (`local_26fb9ed8-…`, `local_914fa6b5-…`): merged, idle; archive on Rae's OK.
- **SMARTS contamination note + its review** (`local_2f338ed6-…`, `local_c656acac-…`): Rae drives these; archived by
  accident 2026-11-11, and Rae is to unarchive them.

## Waiting on Rae
- T31 merge questions (`_v6w`, CWT `lean`), see above.
- P27 validation-set Q3–5, Q7.
- WG6 questions (`reports/2026-11-12_wg6-dataset-scoping/`):
  - ROT resampling: done by WG6 or by us?
  - Noise variant: "improved" is recommended.
  - EBs: negatives or a flagged class?
  - Is FL (54 M-dwarf rotators) enough?
  - How to handle long ROT periods?
  - Which GBTDS design to assume?
- Headline card: which prevalences to quote, and which of H1–H5 to activate.
- Goals proposal thresholds/conflicts (`reports/2026-11-10_goals-proposal/`, `benchmarks/goals_proposed.json`).
- Midway review leftovers: validation sources; Roman scoping now vs later.
- T12 leftovers: time 6 workers on 1,000 stars? Add a runtime goal on `profile_pipeline.py` output?

## Next up (after the above)
- **Ranker v1** on the rebuilt tables:
  - T27 decline model (all 3 draws per negative source);
  - per-bin thresholds;
  - `train_3k`;
  - `half_snr_min`;
  - drop the persistent-weak-period "quiet" stacks from the negatives.

  `runs/t11_ranker`, `runs/t27_ranker` are in main's `runs/`. Report <1 d success at matched precision, plus the
  negatives' FP rate.
- P25 noise-level features (step 1 with ranker v1 if cheap).
- T5 follow-ups: P32 (`half_lag_consistency` gate vs sub-day success); P33 (low priority).
- P39 (deterministic `np.abs` in `guess_acf_fft`): small; changes final-answer bits once, so it needs a new baseline.

## Pending, no decision needed
- FFI check of three line-like noise peaks (TIC 10001638723 at 2.95 d, 10001717113 at 5.52 d, 10001695140 at 9.95 d):
  rerun when MAST is reachable. Commands are in `reports/2026-11-10_quiet-noise-periods/README.md`.

## Usage hygiene
- **Fresh session per task/phase,** with a focused brief; archive finished sessions promptly.
- **Agents end their turn only when done or genuinely blocked:** no "still running" status turns.
- Use **Sonnet/Haiku** for small, well-specified tasks; keep Opus for design and analysis.
- Avoid dumping long diffs or logs into context; read only the needed sections.
- The Data_Vault drive gets unplugged often: check the mount (and do a real read) before resuming drive jobs.
- zsh gotcha: `git show "$B:path"` breaks when `$B` is a variable (`:r` modifier). Write the ref inline.
