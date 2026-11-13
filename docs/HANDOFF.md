# Handoff note (2026-11-13, coordinator → fresh coordinator)

For a **fresh coordinator session.** The previous coordinator ran 2026-11-11 → 2026-11-13 (summarized twice) and was
handed off to keep turns cheap on Rae's Pro plan.

**Start here:** `CLAUDE.md` is already loaded; read the last few changelog entries in `docs/PROJECT_CONTEXT.md`,
the Roadmap and statuses in `docs/TASKS.md`, and this note. Open `TASKS_PROPOSED.md` / `TASKS_DISCUSSIONS.md`
only for the item you need (search by its `### P##.` / `### D##.` heading). Don't re-read whole reports unless a task needs them. Git history has the details.

**Update (2026-11-13, coordinator #2):** T31 merged with Rae's answers (`local_screen2` + CWT `lean` defaults, `_v6w`
replaces `_v6`), dev_medium + synth200 rerun and committed as **`_v7`, the judging baseline**. Next: start T32 from main.

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

## State of `main`
- **Merged 2026-11-12/13:**
  - T5 hybrid ranking + P30 docs diet;
  - headline card, with dev_medium as the default period population (Rae: *not* McQuillan+2014, which has
    almost no P < 1 d stars);
  - WG6 scoping (`9063c95`);
  - **T12 speed-up** (`44768ec`): 1.5–1.6× faster per star, bit-identical, v6 baselines; keep 4 workers in code
    defaults (3 in briefs, for machine safety).
  - **T31** (`fd6e716` merge, `5d14f3f` default flip): wavelet reproducible; default comb fit `local_screen2`, CWT `lean`.
- **Baselines:** use the `_v7` files (`benchmarks/baselines/`). `_v6w`, `_v6`, `_t5` are history.
- **Env:** `ss_openblas` also has astroquery 0.4.11 (P27).
- Untracked `docs/figures/` and `tools/build_status_flowchart.py` in main are Rae's; leave them.

## Unmerged branches ready for Rae
1. ~~T31~~ merged 2026-11-13 (see State of main). Backlog from it: P39 (`guess_acf_fft` complex `np.abs`), T32.
2. **P27 validation-set scoping:** `claude/p27-validation-scoping-4c1c1b` (`f6edf03`). Rae decided:
   - Q1: Zach's list is tagged `expected_undetectable`.
   - Q2: TESS-SPOC SAP is primary, QLP the fallback, and PDCSAP is also scored.
   - Q6: astroquery installed.
   Q3–5 and Q7 wait on her review of the report.

## First actions for the new coordinator
1. **Start T32, the drift-robust `local` comb solver** (`docs/TASKS.md` T32). Get Rae's go-ahead in your own session.
   - Fresh session, worktree cut from main (T31 is merged). Judge against `_v7`.
   - Role `method-improver`. Explore all five of Rae's ideas:
     1. sequential peak tracking;
     2. a tolerance that grows with n;
     3. a quadratic drift term, with `a` as a ranker feature;
     4. height/SNR weighting;
     5. a residual-scatter gate.
   - First step, with no scorecards: plot vertex residual vs n for local's 29 gains, the 28 none→wrong stars, and
     shared-correct stars (`reports/2026-11-13_t31-comb-fit-speed/data/changed_stars_*`).
   - Each idea becomes an `SS_COMB_VARIANT` switch, scored on dev_medium full-year + assigned with
     `check_goals --baseline _v7`, and timed with `profile_pipeline.py` on T12's 15 stars.
   - Figures: residual-vs-n; a success/wrong-vs-time frontier; by-bin; gate-statistic distributions with the
     threshold; ≥4 gained and ≥4 lost example LC + ACF plots; the distribution of `a` split by diffrot.
   - Machine-safety rules as above.
   - Report: `reports/2026-11-13_t32-local-drift/`.
2. ~~Merge T31~~ done 2026-11-13.
3. **Then rebuild the ranker tables (P31)** on the final comb fit and wavelet: about 2.5 h with the drive, at 3
   workers maybe a little more. Tell Rae first. T31 is merged; it waits on T32 only if Rae wants `local` in the tables. Note: screened-out candidates' diagnostics now
   come from the `local` fit (changelog 2026-11-13), so candidate features shift for those.

## Open sessions (2026-11-13)
- **T32 drift-robust local fit** (`local_d7e7d540-…`): running, started 2026-11-13 from main.
- Archived 2026-11-13 (Rae): T31, T12, headline card, short-period sessions.
- **WG6 scoping** (`local_5b376121-…`): merged; **don't archive** (Rae reviewing).
- **Validation-set scoping** (`local_13645bab-…`): unmerged, see above.
- **Midway review** (`local_d5a20966-…`): merged; **don't archive** (Rae reading).
- **SMARTS contamination note + its review** (`local_2f338ed6-…`, `local_c656acac-…`): Rae drives these; archived by
  accident 2026-11-11, and Rae is to unarchive them.

## Waiting on Rae
- Answered 2026-11-13 (recorded in the task files): P27 mix + `pulsation` outcome; WG6 (resample ROT ourselves,
  improved noise, EBs flagged, more varied rotators than FL, dev uses the 3 early seasons); headline card (f = 0.1;
  H1/H3/H4 active, H2/H5 guards); P29 70 d run deferred; 6 workers ruled out (4 already strains the machine).
- P27: pick a long-period (70–180 d) block from the options in `TASKS_PROPOSED.md` P27; then freeze (Q7).
- WG6: how to handle long ROT periods.
- Goals proposal thresholds/conflicts (`reports/2026-11-10_goals-proposal/`, `benchmarks/goals_proposed.json`).
- Midway review leftovers: validation sources; Roman scoping now vs later.

## Next up (after the above)
- **Runtime baseline for goal RT2** (no Rae input needed): when the machine is idle (after T32), run
  `nice -n 15 $PY benchmarks/profile_pipeline.py --manifest datasets/dev_medium.csv --data-root data/smarts_dev
  --ignore-manifest-sectors --star-ids-file benchmarks/profile_stars.txt --features --repeats 3
  --summary-json benchmarks/baselines/profile15_fullyear_v7.json` (~2–3 min), and commit it. RT2 (≤ 10% slower) is
  run only occasionally, before speed-relevant merges. RT1a/b are free draft checks on every scorecard.
- **Batch the baseline-changing work** (Rae, 2026-11-13: avoid dead time): P39 and any adopted T32 variant go into one
  baseline refresh (~15 min of scorecards), then the ranker-table rebuild (P31, ~2.5 h, needs the drive) runs
  **overnight**. Neither is urgent.
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
