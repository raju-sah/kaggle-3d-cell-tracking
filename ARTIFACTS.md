# Artifact Record

This repository tracks code, reports, and provenance only. The local run directory was
deleted after the competition closed; everything below records what existed and how to
recover or regenerate it.

## Final submission

| Field | Value |
| --- | --- |
| File | `submission.csv` (12.6 MB, 241,557 rows) |
| SHA-256 | `443add6dc3f5c0b1ae61f4f74863b941ccc7862693fae588c13cd3157e7d546a` |
| MD5 | `726d905640a5abc28195ca1e15c8ce56` |
| Selected config | `tight55` (`MOTION_RELINK_TIGHT_UM = 5.5`) |
| Public LB score | 0.94741 |
| Rank | 1344 / 3950 |

Recovery: download from the competition's **My Submissions** page, or rerun the
advanced notebook with the tracked config. Submission identity (hash, row count,
selected config) is stored in
[`provenance/resume/final_submission_state.json`](provenance/resume/final_submission_state.json).

## Model weights

Checkpoints were public Kaggle datasets, not trained here:

| Role | Kaggle dataset |
| --- | --- |
| Primary graph generator (400ep snapshot) | `pilkwang/biohub-tracking-support-pack-50ep-v1` |
| Secondary seed (seed 314159, all-train) | `pilkwang/biohub-temporal-unet3d-seed314159-v1` |
| DeepCenter center prior | `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` |

Attach them as notebook inputs; `notebooks/biohub_advanced_model/kernel-metadata.json`
lists the exact slugs. Checkpoint hashes and training config are preserved in
[`provenance/metadata/`](provenance/metadata/) and
[`research_sources_2026-09-18/primary_artifact_manifest.json`](research_sources_2026-09-18/primary_artifact_manifest.json).

## Competition data

```bash
kaggle competitions download -c biohub-cell-tracking-during-development
```

## Regenerable, not worth keeping

| Local path (deleted) | Size | Rebuild |
| --- | --- | --- |
| `logs_v13/`, `logs_v14/` | ~97 MB | rerun `scripts/pipeline_manager.py` (presets are in the script) |
| `secondary_seed_weights/` | 33 MB | re-download secondary dataset |
| `submission.csv` | 13 MB | rerun notebook / download submission |
| `tracking_repo/predictions/` | ~3 MB | rerun notebook inference |
| `data/` | 4 MB | Kaggle download command above |

Run-state snapshots from the tuning and submission rounds are kept in
[`provenance/resume/`](provenance/resume/) so the selected thresholds and pipeline
signatures remain reproducible without the logs.
