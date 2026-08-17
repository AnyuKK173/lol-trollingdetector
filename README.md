# LoL Player-State Risk Pipeline

This repository contains the complete code path for collecting Riot Match-V5
and Timeline-V5 data, building patch-locked Gold reference curves, producing
five-minute future-event labels, and training the v4 replay-free teacher/review
system.

The system does **not** claim to detect intentional trolling. Riot Timeline data
cannot reveal chat, player intent, skill-shot execution, or every contextual
reason behind a decision. The v4 layer combines independent current-state
signals to identify reviewable/intervenable risk states, while the actual
forecasters keep using observable future events such as deaths and resource
collapse as targets.

## Architecture

```text
Riot API -> PostgreSQL -> v3 Gold behavior windows
                              |
retained matches, all 10 players -> role-specific checkpoint outcome teacher
                              |
Gold subject P-score + teacher -> weak review state / confidence gate
                              |
future *_h5 event labels -> A/B/C calibrated risk forecasters
```

The teacher is trained on all ten participants in every retained match because
real games contain mixed visible rank/MMR contexts. Final review output remains
restricted to the Gold-verified v3 subject windows, preserving the stated
population boundary of the current study.

## Safety and leakage contracts

- A whole match belongs to either train or test. If v3 Gold subjects imply both,
  that match is excluded before any player from it can enter training.
- Teacher features use only frames/events at or before the checkpoint timestamp.
- `final_win` is a teacher target only and is absent from `FEATURE_COLUMNS`.
- Teacher percentiles are fit only on train rows, grouped by role and minute.
- One-minute review windows receive only the latest past teacher checkpoint,
  with a default maximum age of two minutes. Future backfill is forbidden.
- Every other `*_h5` future label is excluded when training one risk head.
- Review states are weak supervision and intervention gates, never intent truth.

## Repository contents

| File | Purpose |
|---|---|
| `collector.py`, `riot_api.py` | Patch-locked Riot collection with retry/rate-limit handling |
| `schema.sql`, `storage.py` | PostgreSQL schema and transactional persistence |
| `parsers.py` | Pure Match/Timeline payload parsers |
| `validate_collection.py`, `export_parquet.py` | Data quality checks and Parquet export |
| `build_baselines.py`, `p_score.py` | Gold role/champion/minute P25/P50/P75 baselines and P-score |
| `build_behavior_dataset.py` | v3 Gold-subject windows and independent future `*_h5` weak labels |
| `train_pattern_forecaster.py` | Original v3 event-risk forecasters |
| `build_timeline_teacher_dataset.py` | All-player, three-minute, cutoff-safe teacher training rows |
| `train_timeline_teacher.py` | Five role-specific calibrated outcome-proxy teachers |
| `build_auto_review_dataset.py` | Past-only teacher/P-score merge and evidence-backed review gate |
| `train_review_gated_forecaster.py` | A/B/C comparison, grouped CV, calibration and held-out metrics |
| `run_v4_pipeline.py` | End-to-end v4 orchestration |
| `tests/` | Unit, leakage, data-contract and optional PostgreSQL integration tests |

Generated Parquet datasets and model binaries are intentionally ignored. They
are derived artifacts, can contain persistent player identifiers, and do not
belong in a public code repository.

## Setup

Requires Python 3.10+ and PostgreSQL (Docker Compose is included).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env            # Windows: Copy-Item .env.example .env
docker compose up -d
```

Edit `.env` with a fresh Riot development/production key and the database URL.
Never commit `.env`. The example is locked to patch `16.14` only as the current
study configuration; change it deliberately for a new collection.

Initialize and smoke-test collection:

```bash
python collector.py --schema-only
python collector.py --players-per-division 2 --matches-per-player 2 --stop-after-matches 3
python validate_collection.py
```

Run the full intended collection batch:

```bash
python collector.py --players-per-division 25 --matches-per-player 10
```

Collection is resumable. Complete matches are skipped and a match bundle is
written in one transaction.

## Build v3 and v4

Build the frozen v3 behavior dataset and all v4 extensions:

```bash
python run_v4_pipeline.py
```

If `output_v3/patch=16.14/behavior_windows_v3.parquet` and its train-only
baselines are already present:

```bash
python run_v4_pipeline.py --skip-v3-build
```

Equivalent explicit commands:

```bash
python build_behavior_dataset.py

python build_timeline_teacher_dataset.py \
  --v3-windows output_v3/patch=16.14/behavior_windows_v3.parquet \
  --output output_v4/patch=16.14/timeline_teacher_training_v2.parquet

python train_timeline_teacher.py \
  --checkpoints output_v4/patch=16.14/timeline_teacher_training_v2.parquet \
  --output-dir output_v4/patch=16.14/timeline_teacher_v2

python build_auto_review_dataset.py \
  --windows output_v3/patch=16.14/behavior_windows_v3.parquet \
  --teacher-checkpoints output_v4/patch=16.14/timeline_teacher_v2/timeline_teacher_checkpoints_v2.parquet \
  --output output_v4/patch=16.14/auto_review_windows_v2.parquet

python train_review_gated_forecaster.py \
  --review-windows output_v4/patch=16.14/auto_review_windows_v2.parquet \
  --output-dir output_v4/patch=16.14/review_gated_models
```

The forecaster manifest reports for each target:

- A: original v3 features;
- B: A plus teacher probability/percentile/trend/age;
- C: B plus review state and confidence gate;
- match-grouped cross-validation mean/standard deviation;
- held-out ROC-AUC, PR-AUC, Brier score, log loss and calibration bins;
- metrics on the high-confidence intervention gate where both classes exist.

## Tests

```bash
pytest -q
```

Most tests are pure and run in CI. PostgreSQL integration tests automatically
skip unless `DATABASE_URL` is configured. Generated-artifact contract tests
also skip on a clean checkout until the corresponding outputs exist.

## Current research boundaries

- Match-V5 Timeline is post-game data. A real-time client must map Live Client
  Data API fields into the same current-state contract.
- Gold is a reference population, not a claim that every participant in a
  retained match is Gold at match time.
- Overlapping five-minute windows are correlated; manifests therefore include
  match-grouped CV and absolute positive counts rather than treating every row
  as independent evidence.
- The v4 teacher predicts outcome association. It is not a replacement for
  replay review and must not be described as a validated behavior label.
