# Handoff note (2026-11-16, coordinator #2 → coordinator #3)

For a **fresh coordinator session.** Coordinator #2 ran 2026-11-13 → 2026-11-16 (summarized once) and hands off to
keep turns cheap on Rae's Pro plan.

**Start here:** `CLAUDE.md` is already loaded. Read this note, the last few changelog entries in
`docs/PROJECT_CONTEXT.md`, and the Roadmap/statuses in `docs/TASKS.md`. Open `TASKS_PROPOSED.md` /
`TASKS_DISCUSSIONS.md` only for the item you need (search its `### P##.` / `### D##.` heading). Git history has details.

## Coordinator #3 progress (2026-11-19)
- **P43 clean rerun done** (`local_dfcdea18-…`, branch `claude/p43-clean-rerun-1340c3`, report
  `reports/2026-11-16_p43-clean-rerun/`; verified: trial merge into main 528 pass + the 1 expected failure, protected files
  untouched, metrics match). Same mechanism as the first P43 run (rejection drops the misfit lag-0 / tallest teeth; SNR
  gate, not vertex gate). Recommends `local_tol_trim` (fy 0.265 / wrong 0.024; as 0.153 / 0.030). Its branch and the
  first P43 branch conflict in `comb_fit.py` + changelog (both add variants): resolve at merge.
- **Rae (2026-11-19): before deciding, rerun T32 blind on clean stars** to see whether the P43 fixes are needed at all.
  Base `claude/t32-clean-base` (pushed) = `aa6af76` (main just before the T32 merge) + `97a01c8`'s noise-clean changes,
  minus its HANDOFF/changelog edits; 484 pass. Session `local_52307def-…` (`t32-clean-rerun`), report due at
  `reports/2026-11-19_t32-clean-rerun/`. Compare it against the first T32 report and both P43 runs before the default
  decision.

- **Rae (2026-11-19): merged P35, P27, P34** (decisions in TASKS_PROPOSED P27/P34/P35, P46-P49, D5 and the changelog).
  WG6 manifest needs a CEP rebuild before freezing (zip added to the drive 2026-11-19). P27 freeze is next (70-180 d block
  skipped). Explained "best-achievable period" (P34 Q6) to Rae in chat; her decision pending.
- **Archived 2026-11-19 (Rae):** P35, P34, P27 and midway-review sessions. The 11/10 coordinator `local_ba4679a3-…` could not be
  archived: the app keeps its side session WG6 scoping `local_5b376121-…` (pinned or in use); Rae can archive both from the
  sidebar. Branches `claude/{p27-validation-scoping-4c1c1b,t28-wg6-scoping-43bfaa,midway-review-d397c3,smarts-note-review-2d28d8}` pushed first.

- **T32 clean rerun done** (`claude/t32-clean-rerun-2c0779`, `reports/2026-11-19_t32-clean-rerun/`; protected files
  untouched; `local_screen2` reproduces clean `_v7` exactly). Kept every existing threshold (0.8), so its √n-tolerance
  variant scored only 0.215 fy; the first T32's `local_tol` uses `min_frac_tol=0.6`, set on dev_small. Leader
  `local_seq_trim` (tracking + a trailing-tooth trim, found independently): 0.251/0.021 fy, 0.142/0.026 as; recommends
  keeping `local_screen2`. `local_tol_trim` (P43 rerun) 0.265/0.024, 0.153/0.030 remains the best balance.
  Rae's default decision pending.
- **CEP rebuild merged** (`480c90a`; 24 CEP stars, only that many exist). Session archive blocked (pinned/open).
- **Toeplitz-LS (P50)** `claude/toeplitz-ls-eval-975317`: don't adopt for speed (LS is 1-3% of per-star time; no gain at
  our grid sizes), exact where astropy `fast` is ~1e-4. **P27 freeze** `claude/p27-freeze-27f7e1`:
  `datasets/tess_validation_v0.csv` + `claude_files/tess_lc.py` + download script (1.69 GB) + `tessval_v0_*` strings;
  72 of 480 stars have no MAST product; 7 questions for Rae. Neither merged.

- **Rae (2026-11-19): default comb fit = `local_tol_trim`.** Merged `claude/p43-clean-rerun-1340c3` (which carries
  `claude/local-tol-default`) and, for the `drift_a_frac` ranker feature only, `claude/p43-decayed-tail-22a749`:
  `comb_fit.comb_drift_frac` + ml_features + docs + `tests/test_drift_a_frac.py`; its `local_tol_tail`/`norej`/`snr0`
  variants stay on its branch (report merged as history). Its P45 became P51; its P44 text folded into P44.
  `DEFAULT_COMB_VARIANT = "local_tol_trim"`; default-asserting tests updated. **`_v8` baselines** installed
  (`dev_medium_clean_{fullyear,assigned}_v8`, `synth200_s0_v8`; bit-identical to the rerun's runs; CLAUDE.md points
  to them). `tests/test_coverage.py`'s exact synthetic check now reads `synth200_s0_v8_tables` (the old v2/v3 table only
  matched `local_screen2`). 557 pass. T32 rerun branch not merged (report-only value; Rae can say if she wants it kept).
- **Toeplitz-LS merged as a report** (Rae: not adopted; P50 notes it for sparser surveys).
- **P27 freeze: Rae's answers sent to its session** (30% stratified locked holdout; replace the 72 unscorable stars;
  new outcome definitions approved (additive, `_v7`/`_v8` numbers unchanged); hot stars by Gaia class; QLP fallback
  only; drop the 7 crowded EBs; Zach question still open). Merge when it reports back.
- **Next:** rebuild the P31 ranker tables under `local_tol_trim` (overnight, drive plugged in), refresh the RT2 profile
  baseline, then ranker v1. Still to archive from the sidebar: 11/10 coordinator + WG6 scoping, CEP session.

- **Later 2026-11-19:** merged the P27 freeze (validation set v1), the T32 rerun as a report only, and the P49 Roman noise
  survey (Rae's decisions in P34/P49/P56/D5). `check_goals` guard ignores one-star drops. **Rae: leave the lag-0 tooth out
  (P44)**, with the cutoff set on `dev_medium_b` (P57).
- **Queue for when Rae plugs the drive in (she will say so):** (1) rebuild the P31 ranker feature tables under
  `local_tol_trim` (overnight); (2) refresh the RT2 profile baseline; (3) P52: download the TESS validation light curves
  (1.69 GB, ~35-55 min, to the drive) and score the unlocked 70%; (4) P57: build `dev_medium_b`, a second ~1,000-star noise-clean dev set
  disjoint from the others, with its own `_v8` baselines; (5) P44: set the lag-0-free height-SNR cutoff on `dev_medium_b`,
  judge on dev_medium_clean vs `_v8`. One heavy job at a time. Then ranker v1,
  and the `romrot_v0` build (P34 decisions: 1.3 ppt floor, blend fraction, ×0.7, `p_best_achievable`).
- **2026-11-20 drive session:** (1) done: P31 tables rebuilt under `local_tol_trim`, 0 errors: `runs/p31_dev_medium_
  {assigned,fullyear}` (874 stars each) and drive `extracted/p31_tables/{train_3k (3000), neg_dev_medium (1736),
  neg_train (2972)}`. The drive dropped twice mid-run (USB; Rae reseated it); `--resume` recovered. (2) done:
  `benchmarks/baselines/profile15_fullyear_v8.json`, mean 0.739 s/star (repeat 0.738).
  (3) done: TESS validation downloaded and the unlocked 70% scored (numbers in P52); local copy `data/tess_validation/`.
  Drive dropped a third time under 3-worker reads: run drive-reading jobs with 1 worker until the connection is fixed.
  (4) done: P57 `dev_medium_b_clean` merged (1,000 stars; `_v8` 0.274 / 0.034 full-year, 0.167 / 0.020 assigned; no
  overtuning sign). Next: (5) P44 lag-0 cutoff tuned on B, judged on dev_medium_clean vs `_v8`.

## Your first activity (Rae, 2026-11-16): a clean, blind rerun of the decayed-tail fix task

**Why.** The decayed-tail fix agent (P43, session `local_f120e7df-…`, branch `claude/p43-decayed-tail-22a749`, report
`reports/2026-11-15_p43-decayed-tail/` on that branch) did its diagnosis and scoring on all 1,000 dev_medium stars,
including the 126 whose real-TESS noise source was contaminated before injection (`noise_exclude`). Those stars hold 55
of the 88 full-year `_v7` wrongs, and the agent's reasoning about losses and wrong answers leaned on them. Rae wants
**the same agent task run again from scratch on the clean stars only**, to see whether the reasoning and the
recommendation end up in the same place as before (and as coordinator #2's quick re-score, below).

**Set-up (keep it blind).**
- Start ONE fresh session (Rae's go-ahead is already given for this activity; confirm with her in your session that she
  still wants it started now, and check the machine is idle). Role `method-improver`. Opus (design/analysis work).
- Worktree cut from **`claude/local-tol-default`** (pushed; default comb fit `local_tol`, gate-test updates, main's
  noise-clean manifests and guard merged in at `db64668`; 517 pass, 1 expected failure:
  `test_default_scorecard_rows_bit_identical_to_baseline`, which waits for the synth200 baseline refresh).
- **Do not give the agent the earlier answers.** Its brief must say: do not open `reports/2026-11-15_p43-decayed-tail/`,
  the branch `claude/p43-decayed-tail-22a749`, its worktree or `runs/p43/`, this HANDOFF note, or the 2026-11-16 entries of
  the `PROJECT_CONTEXT.md` changelog, until its own report is written. It may read everything the first agent had: the
  T32 report (`reports/2026-11-13_t32-local-drift/`), P43 in `docs/TASKS_PROPOSED.md`, `comb_fit.py`.
- Brief = the original P43 brief, re-pointed at clean data:
  - Reference runs of the branch default (`local_tol`) on `datasets/dev_medium_clean.csv` full-year
    (`--ignore-manifest-sectors`) and assigned, plus `--synthetic 200 --seed 0`, each with `check_goals.py --baseline`
    against `benchmarks/baselines/dev_medium_clean_{fullyear,assigned}_v7.json` (and `synth200_s0_v7.json`).
  - Diagnose `local_tol`'s lost stars vs `_v7` (clean stars only) before fixing: which gate fails, and why.
  - Try principled variants as new `SS_COMB_VARIANT` entries (the P43 ideas: count `frac_vertex_tol` over positive-height
    teeth; apply the tolerance test after the decayed-tail trim; T31's robust per-window height fit; plus whatever its own
    diagnosis suggests). **No new thresholds, no threshold scans or tuning on dev_medium** (Rae, 2026-11-15).
  - Full scorecards for serious candidates, both modes, against the step-1 runs and the clean `_v7`; time the winner
    with `benchmarks/profile_pipeline.py --manifest datasets/dev_medium_clean.csv --data-root data/smarts_dev
    --ignore-manifest-sectors --star-ids-file benchmarks/profile_stars.txt --features --repeats 3`.
  - Leave out the `drift_a_frac` ranker feature: it is already implemented on the first P43 branch.
  - Figures: success/alias/wrong by period bin (`_v7`, `local_tol`, each variant, both modes); the gate-failure breakdown
    of the lost stars; `frac_vertex_tol` distributions for correct vs wrong candidates with the 0.6 threshold; ≥4 recovered
    and ≥4 newly lost/wrong example LC + ACF plots with teeth, heights and tolerance windows; runtime bars.
  - Report `reports/<date>_p43-clean-rerun/` via `write-report`; "Open questions for Rae" names the default it
    recommends. Don't change `DEFAULT_COMB_VARIANT`, protected files, or baselines; commit + push its branch; don't merge.
  - Machine-safety rules below, verbatim, and "end your turn only when done or genuinely blocked".
- **After it reports, you compare** (not the agent): its diagnosis, variant ranking and recommendation vs (a) the first
  P43 report and (b) coordinator #2's clean re-score below. Tell Rae in plain terms where they agree and differ, with a
  figure or table, before any default decision.

**Coordinator #2's clean re-score, for your comparison only** (per-star tables of the first P43 runs filtered by
`~noise_exclude`; 874 stars; success / wrong, full-year | assigned):

| variant | success fy | wrong fy | success as | wrong as |
|---|---|---|---|---|
| `_v7` (`local_screen2`) | 0.254 | 0.038 | 0.146 | 0.029 |
| `local_tol` | 0.252 | 0.019 | 0.143 | 0.024 |
| `local_tol_tail` | 0.262 | 0.023 | 0.152 | 0.029 |
| `local_tol_tail_snr0` | 0.255 | 0.019 | 0.149 | 0.024 |
| `local_tol_norej_tail_snr0` | 0.262 | 0.016 | 0.150 | 0.024 |

The first P43 agent (all 1,000 stars) recommended keeping `local_tol` because `local_tol_tail` added about one wrong per
recovered star. On clean stars coordinator #2 leaned to `local_tol_tail` (beats `_v7` on success in both modes with
equal-or-lower wrong; vs `local_tol` +0.010 success for +0.004/+0.005 wrong). All `local_tol` variants lose 0.02-0.03
success at 10-30 d vs `_v7` (`norej_tail_snr0` -0.066 full-year). Its first-run diagnosis: the step-4 window-rejection
loop drops the tallest early teeth of decaying ACFs, so the mean-height-SNR gate (not the vertex gate) causes most losses.

**Rae's default decision (`local_tol` vs `local_tol_tail` or another) waits for this rerun.**

## After the rerun: the baseline-changing batch
1. Rae picks the default. Merge `claude/local-tol-default` (and the chosen variant's branch: the first P43 branch also
   carries `drift_a_frac`, which Rae wants as a ranker feature; merge it even if its variants are not adopted).
   If the default is not `local_tol`, flip `DEFAULT_COMB_VARIANT` and the default-asserting tests in the same change.
2. `_v8` baselines: `dev_medium_clean_{fullyear,assigned}_v8.json` (+ tables). If the rerun agent's runs of the chosen
   default were made on code identical to merged main for the default path, derive them with `tools/rescore_subset.py`
   (exact; see `benchmarks/baselines/README.md`); otherwise rerun the two scorecards (~13 min at 3 workers). Regenerate
   `synth200_s0` tables on the new default (~2 min); that fixes the one failing test. RT2 runtime baseline:
   `benchmarks/baselines/profile15_fullyear_v8.json` from the profile command above with `--summary-json` (~2 min);
   point goal RT2 at it. Update `benchmarks/baselines/README.md`, CLAUDE.md's baseline names, changelog.
3. **P31 ranker-table rebuild, overnight, Data_Vault drive plugged in** (~2.5 h): dev_medium tables from
   `datasets/dev_medium_clean.csv` (both modes, `runs/p31_dev_medium_{assigned,fullyear}`), then on the drive
   `extracted/p31_tables/{train_3k,neg_dev_medium,neg_train}` (`datasets/train_v0.csv`; `datasets/negatives_v0.csv
   --nonrotator --only-set dev_medium|train`). Put any queue in a script FILE: the 2026-11-13 queue deadlocked because a
   watcher's command line contained "pytest", matching the queue's own idle check (`pgrep -f`).
4. Ranker v1 on the rebuilt tables (see "Next up").

## Noise-clean rule (Rae, 2026-11-16; done on main `97a01c8`)
- `datasets/dev_{small,medium}_clean.csv` (47 / 874 stars) = frozen manifests minus `noise_exclude` sim_ids
  (`tools/build_clean_manifests.py`, `--check`). Frozen manifests untouched.
- `scorecard.build_tasks` refuses manifests holding such stars (covers scorecard, `profile_pipeline.py`,
  `build_feature_tables.py`, harnesses built on it); `--allow-noise-excluded` for audits, labelled in `config.dataset`.
- `headline_card.py` defaults to the clean manifest and restricts systems to its stars (period-mix population still
  from the full manifest's injected periods).
- `goals.json` / `goals_proposed.json` dataset strings name the clean manifests (targets unchanged).
- Judging baselines `dev_medium_clean_{fullyear,assigned}_v7.json` derived from the `_v7` tables (`tools/rescore_subset.py`;
  unfiltered it reproduces `_v7` exactly). Three profile stars swapped for clean ones (`benchmarks/profile_stars.txt`).
- `train_v0` and the negatives were already clean. **The holdout has no sim_id: filter it when Rae opens it.**
- Old reports (T31, T32, first P43 and earlier) were scored on all 1,000 stars; their per-star tables can be re-scored on
  the clean subset with `tools/rescore_subset.py` instead of rerunning.

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
- **Noise-clean rule (Rae, 2026-11-16):** every brief says: develop and judge only on `datasets/dev_{small,medium}_clean.csv`
  (the frozen sets minus `noise_exclude` stars), and filter any per-star table by `~noise_exclude`. The scorecard refuses
  the old manifests (`--allow-noise-excluded` only for audits of those stars).

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

## State of `main` (2026-11-16, `97a01c8` + this note)
- Merged 2026-11-15: T32 drift-robust `local` variants (`ec03830`; its follow-up "P41" is **P43** on main, since P41 is the
  methodology reviewer); P39 reproducible |c|^2 in `guess_acf_fft` / `ml_features` (`80fa98c`; changed nothing measurable).
- Default comb fit on main is still `local_screen2`; CWT `lean`. `celerite2==0.3.2` pinned (P34).
- Untracked `docs/figures/` and `tools/build_status_flowchart.py` in main are Rae's; leave them.

## Branches (pushed, not merged)
- `claude/local-tol-default` — default -> `local_tol` + gate-test updates; base for the rerun.
- `claude/p43-decayed-tail-22a749` — first P43 run: variants `local_tol_pos/_tail/_norej*/_snr0/_rh`, `drift_a_frac`
  feature, report. Trial merge into main: 543 pass + the 1 expected failure. Merge after Rae's default decision.
- `claude/p35-wg6-loader-ef0d37` — WG6 Roman loader; trial-merge pytest 487 pass. **Needs Rae's OK** (manifest freeze).
- `claude/p34-roman-rotators-accd50` — Roman rotator scoping (celerite2 GP generator, proposed `romrot_v0` set of 3,375
  stars, ~2-3 CPU-h as Rae's local job); trial-merge pytest 510 pass, changelog conflict only. **Needs Rae's OK.**
- `claude/p27-validation-scoping-4c1c1b` — real-TESS validation-set scoping; Rae decided Q1, Q2, Q6, mix + `pulsation`
  outcome; open: a 70-180 d block (options in P27), then freeze.
- `runs/overnight_2026-11-13/` holds a stale RT2 profile on `local_screen2` and a P39 full-year scorecard (all 1,000
  stars); ignore both.

## Open sessions
- Idle, linked to coordinator #2 (archive only on Rae's OK): t32-local-drift `local_d7e7d540-…`, wg6-loader
  `local_e1f3a83b-…`, roman-rotators `local_56d3fdb8-…`, p43-decayed-tail `local_f120e7df-…` (keep until the comparison
  is done; Rae talked to it about the contamination).
- Don't archive: WG6 scoping `local_5b376121-…`, midway review `local_d5a20966-…` (Rae reviewing); validation scoping
  `local_13645bab-…` (unmerged). SMARTS contamination note + review (`local_2f338ed6-…`, `local_c656acac-…`) are Rae's.

## Waiting on Rae
- Default comb fit after the clean rerun.
- OK to merge P35 (WG6 loader) and P34 (Roman rotators).
- P27: pick a long-period (70-180 d) block, then freeze. WG6: long ROT period handling (she pursues Cepheids herself).
- Goals-proposal thresholds (`reports/2026-11-10_goals-proposal/`, `benchmarks/goals_proposed.json`).
- Midway review leftovers: validation sources; Roman scoping timing.
- Decided 2026-11-15/16: adopt `local_tol`-family default (variant pending); `drift_a_frac` becomes a ranker feature; no
  threshold scans on dev_medium; noise-clean rule.

## Next up (after the batch above)
- **Ranker v1** on the rebuilt tables: T27 decline model (all 3 draws per negative source), per-bin thresholds,
  `train_3k`, `half_snr_min`, drop the persistent-weak-period "quiet" stacks from the negatives. `runs/t11_ranker`,
  `runs/t27_ranker` are in main's `runs/`. Report <1 d success at matched precision plus the negatives' FP rate.
- P44 (proposed on the first P43 branch; lands on main when it merges): joint redesign of window rejection + the SNR gate, judged on a fresh dev set, not
  dev_medium. P25 noise-level features. P32 / P33 (T5 follow-ups). P41 (skeptical methodology reviewer) and P42
  (top-N candidates per method) when Rae says.

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
