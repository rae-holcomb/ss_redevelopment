# Proposed (unfrozen) manifests

Drafts for Rae to review. They are **not** frozen and nothing scores against them yet.

| File | What | Built by |
|---|---|---|
| `p27_validation_draft.csv` | P27 real-TESS validation set draft: 480 TIC IDs (rotators with literature periods + method/data-set provenance, Kepler-quiet negatives, candidate negatives, Prša+2022 EBs) with FFI sectors and HLSP availability per star | `tools/validation_set/build_manifest.py` (seed 27); see `reports/2026-11-12_validation-set-scoping/` |

Rows are keyed by TIC ID, not file paths. The light curves don't exist locally yet (download plan in the report).

P27 v1 inputs (committed so `tools/build_tess_validation.py` reproduces `datasets/tess_validation_v1.csv` offline):
`p27_v1_replacements.csv` (79 replacement stars, with the removed TIC in `replaces`; `tools/validation_set/draw_replacements.py`) and
`p27_v1_gaia_classes.csv` (Gaia DR3 variability class per TIC; `tools/validation_set/gaia_classes.py`).
