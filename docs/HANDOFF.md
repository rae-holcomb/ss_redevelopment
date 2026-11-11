# Handoff note (2026-11-11, coordinator; updated late 2026-11-11)

For a **fresh coordinator session.** The previous coordinator ran from 2026-10-22 to 2026-11-11 and was summarized
once; it was handed off to keep turns cheap on Rae's Pro plan.

**Start here:** read `CLAUDE.md`, the tail of the changelog in `docs/PROJECT_CONTEXT.md`, the statuses in
`docs/TASKS.md`, and this note. Don't re-read whole reports unless a task needs them. Git history has the details
of everything merged.

## Coordinator role (Rae, 2026-10-22)
- Start focused fresh sessions, **≤ 2 parallel agents**. Each brief names the TASKS.md entry, the files in scope, and
  **the specific figures expected** (embedded in the README next to the claim; see CLAUDE.md "Plots").
- Verify every finished branch yourself before reporting: full pytest in `ss_openblas`, key-by-key metric comparison
  against the baseline (NaN-aware), pool/leakage checks (training SIM_ID < 800k only, no shared noise TIC with
  dev_medium or the holdout), protected files untouched, figures present, trial merge.
- Merge only when Rae says so (`--no-ff`), push main, archive the session when she agrees.
- Keep `TASKS.md`, the changelog and this note current.
- **In chat, name tasks by what they do** ("the non-rotator examples", "the ranker decline model"); T## IDs only in files.
- **Don't ask Rae for a decline precision target.** Deferred by her (2026-11-10) until more of the pipeline is in
  place. Default: match hand arbitration's OOF training precision, state the non-rotator mix (1:1 now), and plot
  the full purity–completeness curve with candidate cutoffs marked.
- Explain a decision in plain terms the first time it goes to Rae: what it means and why it matters.

## State of `main` (2026-11-11)
- **Merged this stretch:** T20 (noise sources), T16c (searchable range), scorecard v4 and v5 (noise-period scoring),
  T10 feature tables, quiet-noise reclassification (`periodic_low_amplitude`, 34 sources), T11 ranker v0,
  T27 non-rotator negatives (`857f6e1`), quiet-noise follow-ups (see below).
- **Plots requirement** strengthened in `CLAUDE.md`, the `write-report` skill and the reviewer checklist (`55bd504`).
- **Env:** `ss_openblas` now has xgboost 3.4.1, scikit-learn 1.9.1, llvm-openmp (`environment.yml`).
- **Main checkout's `runs/`** holds `t10_dev_medium_{assigned,fullyear}` (dev_medium rotator feature tables) and is
  the stable path ranker work reads from.
- Untracked `docs/figures/` and `tools/build_status_flowchart.py` in main are Rae's; leave them.

## Open sessions (2026-11-11 night)
- **Short-period recovery** (T5, `local_26fb9ed8-…`, branch prefix `claude/t5-short-period`): diagnose the full-year <1 d
  deficit, then pick optional method(s) to pool by recall / compute / non-rotator FP. Expect 2–4 h; needs v6 baselines.
- **Midway review** (`local_d5a20966-…`, branch prefix `claude/midway-review`): read-only science/plan/wheel-spinning
  review, report `reports/2026-11-11_midway-review/`.
- **SMARTS contamination note + its review** (`local_2f338ed6-…`, `local_c656acac-…`): Rae drives these. They were
  archived by accident on 2026-11-11 (side sessions of the old coordinator); worktrees intact; Rae to unarchive.

## Waiting on Rae
- Thresholds/conflicts in the goals proposal (`reports/2026-11-10_goals-proposal/`, `benchmarks/goals_proposed.json`).
- T5's choice of which optional method(s) to pool, when it reports.

## Next up (decided)
- **Ranker v1** after T5 settles the candidate pool: T27 decline model (all 3 draws per negative source), per-bin
  thresholds, train on `train_3k` (drive `extracted/t10_tables/train_3k`; learning curve plateaued at 3k), `half_snr_min`,
  drop persistent-weak-period "quiet" stacks from the negatives. `runs/t11_ranker`, `runs/t27_ranker` are in main's `runs/`.
- **Efficiency pass** (performance-optimizer) alone on the machine, after T5 merges.
- P25 noise-level features (step 1 with ranker v1 if cheap; step 2 per-sector weighting is its own task).

## Pending, no decision needed
- FFI check of three line-like noise peaks (TIC 10001638723 at 2.95 d, 10001717113 at 5.52 d, 10001695140 at
  9.95 d): rerun when MAST is reachable; commands in `reports/2026-11-10_quiet-noise-periods/README.md`.
- `half_snr_min` (half-coherence feature) is in `ml_features.py` but not in the feature tables; include it in ranker v1.

## Next candidates (roadmap)
- Ranker v1: decline model from T27, `half_snr_min`, `beyond_reach` rows (P23), per-bin thresholds (P24).
- Detectability ceiling (T26), Roman scoping with Rae (T28).

## Usage hygiene
- **Fresh session per task/phase,** with a focused brief; archive finished sessions promptly.
- **Agents end their turn only when done or genuinely blocked:** no "still running" status turns.
- Use **Sonnet/Haiku** for small, well-specified tasks; keep Opus for design and analysis.
- Avoid dumping long diffs or logs into context; read only the needed sections.
- The Data_Vault drive gets unplugged often: check the mount (and do a real read) before resuming drive jobs.
