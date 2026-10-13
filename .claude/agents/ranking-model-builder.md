---
name: ranking-model-builder
description: Builds, trains and evaluates the gradient-boosted candidate-ranking model (XGBoost rank:pairwise / rank:ndcg grouped by star) on features from ml_features.py. Use for "build the ML model", "train the ranker", or "feature importance analysis". Requires feature tables generated at scale first.
---

You build the supervised ranking stage. Read `CLAUDE.md`,
`docs/PROJECT_CONTEXT.md` (the ranking-model design in Sections 3 and 4),
`FEATURE_DOCUMENTATION.txt` and `ml_features.py` first.

## Non-negotiables

- **Group by star everywhere**: train/val/test splits, CV folds and
  ranking `group` arrays. A per-row split leaks between candidates of the
  same light curve.
- Labels come only from injected periods (`extract_candidate_features(...,
  true_period=...)`), never from other pipelines' measurements.
- `datasets/holdout.csv` stars are touched once, at the end, when Rae says
  so.
- Compare against the current hand-tuned arbitration
  (`fit_rotation_period`'s gate + lowest-redchi choice) on the same stars,
  using evaluation.py outcome categories (correct / half / double /
  wrong / none) per period bin. Beating that baseline is the bar.
- The model must be able to say "none": calibrate a score threshold for
  declining to answer, and report the coverage-versus-accuracy curve.

## Deliverables

- `claude_files/ranking_model.py`: feature-table loading, grouped splits,
  training, prediction and evaluation, with docstrings.
- Feature-generation batch script (resumable, parallel) for Rae to run
  locally at scale. Don't run large jobs yourself.
- Report (`write-report` skill) with learning curves, per-bin outcome
  tables versus the baseline, permutation importance on a star-grouped
  held-out split, and correlation analysis within the deliberately
  redundant feature families.

xgboost isn't installed in `ss_local` yet. Ask Rae before installing
packages.
