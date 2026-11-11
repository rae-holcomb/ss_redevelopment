# Handoff note (2026-11-11, coordinator)

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

## Open sessions
- **SMARTS contamination release note** (`local_2f338ed6-…`, branch `claude/smarts-contamination-note-969cc1`) and its
  **independent review** (`local_c656acac-…`). Rae drives these herself. Remaining: rebuild tables after the
  2026-11-11 flag relabel (435 quiet → common-mode), M6 (segment table and clipping rule), should-fixes S2/S4/S5/S7,
  PDF render, questions for Zach Claytor (Q1–Q3, Q6). Keep both until the note is delivered.
- **Quiet-noise follow-ups** (`local_ad2c3dac-…`): see "Waiting" below for its one leftover.

## Waiting on Rae
- Her local run of `tools/run_t10_train_tables.sh` (~51 min) for the ranker's 3k learning-curve point.
- Whether to build a full-year training table for the ranker.
- Whether to adopt T27's star-level decline model as ranker v1's decline (needs learning curve, feature ablation,
  per-period-bin thresholds, P24) and how many negatives per rotator to train with.
- T27 open questions 4–7: weak-signal "quiet" sources in the negatives, common-mode full-year FPs (T29), the 125
  spike-dropped stacks, synthetic red noise.
- Photometric-uncertainty proposal (empirical noise features, then per-sector weighting): not yet filed; Rae hasn't
  responded.
- Goals proposals not yet in `goals.json` (v4's five, T16c drafts, v5's quiet-subset guards): offer to collect them
  into one proposal, now that the quiet subset's definition is settled.

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
