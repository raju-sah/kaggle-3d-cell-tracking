# kaggle-3d-cell-tracking

3D cell detection and tracking pipeline for the Kaggle competition
[**Biohub - Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

Detect and track zebrafish cells through 3D space and time from volumetric microscopy,
scoring predicted node/edge graphs against held-out annotations.

## Result

| Metric | Value |
| --- | --- |
| Public leaderboard rank | **1344 / 3950** (top ~34%) |
| Best public score | **0.94741** |
| Submissions used | 12 |
| Last submission | 2026-09-16 |
| Kaggle team | `rajucode` (Raju Sah) |

Leaderboard source:
[competition leaderboard, search "raju sah"](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard?search=raju+sah).

## Approach

Local proxy selection on top of a public 0.947 baseline:

1. **Detection** - dual-seed TemporalUNet3D ensemble plus a DeepCenter center-prior model,
   fused with harmonic bidirectional probabilities.
2. **Association** - ILP global linking with motion-aware relinking, node-transformer
   association features, and spatial TTA.
3. **Repair** - safe division recovery, gap closing, short-track rescue, and retention
   guards so detections are not silently dropped.
4. **Tuning** - held-out parameter sweeps scored with a local reimplementation of the
   official metric; selected config in [`ppsweep_selected.json`](ppsweep_selected.json).

The shipped kernel is [`notebooks/biohub_advanced_model/biohub_advanced_pipeline.ipynb`],
derived from [Reyhan Ksatria's public 0.947 notebook](https://www.kaggle.com/code/reyhanksatria/biohub-cell-tracking-0-947-lb)
with public checkpoint datasets attached. The baseline starter notebook is original.

The local sweep improved the stored proxy from `0.948689` to `0.950749`
(`MOTION_RELINK_TIGHT_UM = 5.5`); see [`ppsweep_results.csv`](ppsweep_results.csv).

## Layout

```
src/metric.py                  official metric reimplementation
scripts/pipeline_manager.py    iteration presets, kernel push, submission dispatch
scripts/prepare_iteration.py   notebook config guard synchronization
notebooks/                     baseline starter + advanced submission kernel
experiments/sep18/             Colab bootstrap and parameter range probe
research_sources_2026-09-18/   reference metric implementations and split audit
reference_kernels/             public notebooks used as evidence (credited in research notes)
research_2026-09-18.md         competition/metric research and experiment rationale
*.jsonl, *.csv, *.json         small diagnostic reports and sweep results
```

## Reproducing

```bash
pip install numpy scipy pandas blosc2
python -m scripts.prepare_iteration     # sync notebook config
python -m scripts.pipeline_manager      # push kernel, monitor, submit
```

The advanced notebook needs the competition data plus the three public checkpoint
datasets listed in its `kernel-metadata.json`. Inference runs offline on Kaggle.

## Not in the repo

Model weights (`*.pth`), run logs (`logs_v13/`, `logs_v14/`), prediction zarrs
(`tracking_repo/`), downloaded competition data (`data/`), and `submission.csv`
are regenerable artifacts and are gitignored. The local copies were deleted after
the competition; checkpoint sources, submission hash, and regeneration steps are
recorded in [`ARTIFACTS.md`](ARTIFACTS.md).

## Metric

Score = adjusted edge Jaccard + 0.1 x division Jaccard, with 7 um node matching.
See [`src/metric.py`](src/metric.py) and the
[host metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).
