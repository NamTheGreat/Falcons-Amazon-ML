# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Amazon ML Challenge 2026 entry: **Business Entity Resolution**. For every Source 1 (S1) record, find all matching Source 2/3 records (zero, one, or many). The records come from three noisy sources with no shared IDs. Scoring is **macro-averaged F₀.₅** per S1 entity, which weights precision over recall. Singletons (5.6% of train) score 1.0 only if you predict an empty list, so the rule is "when unsure, do NOT merge."

`MASTER_REFERENCE.md` is the living project doc. It covers dataset stats, observed noise patterns, the strategy, the iteration plan, and submission count. Read it before making design decisions, and update it as the challenge progresses (e.g. submissions used, scores). `student_resource/README.md` is the official problem statement.

## Git rules

- Never commit automatically. Commit only when the user asks.
- **Never push.**

## Layout

- `code/business_entity_resolution/src/`: all pipeline source goes here. It is currently empty. It ships as-is in the submission zip, together with a `README.md` (exact end-to-end reproduction steps) and a pinned `requirements.txt` in `code/business_entity_resolution/`.
- `student_resource/dataset/{train,test}/`: `*_source{1,2,3}.tsv` (columns `entity_id, business_name, business_address, country`) and `train_ground_truth.tsv` (`source1_entity_id, matched_entity_ids`, comma-separated, empty for singletons). The files are large: ~12.5M train rows and ~11.7M test rows.
- `student_resource/utils/validate_submission.py`: the official stdlib-only format validator.
- `student_resource/Documentation_template.md`: the methodology write-up that goes in the zip.
- `requirements.txt` (root): dev dependencies (pandas, sklearn, lightgbm/xgboost/catboost, rapidfuzz, sentence-transformers, faiss-cpu).

## Commands

```bash
pip install -r requirements.txt

# Validate outputs before spending one of the 15 submissions (5/day). Run from student_resource/
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
# add --check-ids to also verify IDs exist in test S2/S3 (uses several GB of RAM)
```

Heavy training and inference runs on Google Colab Pro (T4/A100, 51–83GB RAM). The local macOS machine can't comfortably hold full-scale data in memory, so develop and debug on samples locally.

## Intended pipeline architecture

Data loading → preprocessing/normalization → **blocking** (union of several strategies: same-country filter, TF-IDF char n-gram kNN on name, same on address, token overlap, shared numeric address tokens) → per-pair features (rapidfuzz name/address similarities, numeric token overlap, cross name↔address features for URL-style names, script flags) → LightGBM classifier → threshold tuned for F₀.₅ on a held-out train split → outputs.

Blocking sets the recall ceiling and is the top priority. Address-based blocking is needed to catch DBA/trade names and transliterated Indian names (Devanagari/Tamil/Kannada) whose names share no characters with S1.

## Hard constraints that affect code

- Every file is TSV. Always read and write with `sep="\t"`. Addresses and ID lists contain commas.
- **The test set contains France, which never appears in train.** Treat `country` as an open set. Never hard-code, filter, or one-hot to `{US, India}`. Features must be language-agnostic.
- S2/S3 have ~3.3% empty addresses and literal `"null"` strings inside addresses. Name blocking must cover these records.
- Two outputs, each with one row for every test S1 entity (1,732,544 rows) and no duplicate IDs within a list:
  - `matching_results.tsv` (header `source1_entity_id\tmatched_entity_ids`): the only scored file.
  - `candidate_pairs.tsv` (header `source1_entity_id\tcandidate_entity_ids`): the exact final candidate set fed to the model. It must be a superset of the matches.
- Models must be ≤8B params and MIT/Apache-2.0 licensed. **No external data, APIs, geocoding, or lookup databases** (disqualification risk). Transliteration must use local libraries only.
- Self-evaluation: hold out part of train and compute macro F₀.₅ per S1 entity. Include singletons: an empty prediction for a singleton scores 1.0 and any prediction scores 0.0.
