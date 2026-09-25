# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: normalize → blocking (inverted index over hashed keys) → two-stage LightGBM matcher → one-S1-per-record assignment with a tuned threshold.

No external data, APIs or pretrained models are used. Everything is learned from the provided training data. Libraries are MIT, BSD or ISC licensed (see `requirements.txt`).

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM needs OpenMP
brew install libomp
```

Tested on a MacBook (Apple M5, 10 cores, 16 GB RAM). End to end, the pipeline takes about 1.5 h.

## Reproduce end to end

```bash
./run_all.sh <dataset_dir> <work_dir> <output_dir>
# e.g. ./run_all.sh ../../student_resource/dataset ../../work ../../output
```

The script runs these steps (all scripts live in `src/`):

| Step | Command | Output |
|---|---|---|
| 1. Normalize | `prepare.py --data-dir D --work-dir W` | `W/{train,test}_s{1,2,3}.parquet` |
| 2. Blocking | `run_blocking.py --work-dir W --split train` (then `test`) | `W/{split}_cands.parquet` |
| 3. Train | `train.py --work-dir W --gt D/train/train_ground_truth.tsv --model-dir W/models` | `stage1.txt`, `stage2.txt`, `config.json` (threshold) |
| 4. Predict | `predict.py --work-dir W --model-dir W/models --out-dir OUT` | `OUT/matching_results.tsv`, `OUT/candidate_pairs.tsv` |

`train.py` prints the validation macro F0.5 on a held-out 2% of Source 1 entities.

## Method

**Key fact.** Every S2/S3 record matches at most one S1 entity. About 27% of S2/S3 records match nothing, and many of those are deliberate near-copies of an S1 record. So the problem is solved from the S2/S3 side: for each S2/S3 record ("query"), find its best S1 entity or none.

1. **Normalization** (`normalize.py`)
   - Romanize any script with `anyascii`, so Devanagari, Tamil, Telugu, Bengali, Kannada and accented French all become ASCII.
   - Lowercase and strip punctuation.
   - Map abbreviation variants to one canonical form: legal forms, street types, US/Indian state names, ordinals, and romanized spellings such as `praivet`→`pvt`.
   - Remove legal/filler tokens to get a core name, and extract the DBA part of the name.
   - Build a compact name without spaces (for URL-style names like `kempgloba.com`).
   - Build a crude phonetic skeleton per token, so `sophtveyr` and `software` both become `sftvr`.
   - No country is hard-coded. `country` is only used to scope keys, so France (absent from train) works as-is.
2. **Blocking** (`blocking.py`). Each record emits hashed, country-scoped keys:
   - name tokens, bigrams, phonetic skeletons, compact-name prefix/suffix
   - address words, (number, neighbouring word) pairs, unordered address word pairs
   - name/phonetic token × address word/number combos
   
   Keys are IDF-weighted over S1, and keys with document frequency above 400 are dropped. Each query probes only its rarest keys per key type. The top 10 S1 candidates per query are kept, or 24 when the query has no address or a non-Latin name. The result is about 9 candidates per query, with 98.2% pair recall on a train sample.
3. **Stage 1** (`features.py`). LightGBM on blocking scores and about 30 rapidfuzz similarities: name, full name, DBA name, compact and phonetic forms, address, and number overlap. It scores every candidate pair. Pairs with p1 < 0.005 are dropped: that keeps about 13% of pairs and loses under 0.1% of true pairs.
4. **Stage 2** (`token_features.py`, `pipeline.py`). Adds two feature groups:
   - Features that describe *how* two records differ: unmatched name tokens in each direction, minimum per-token similarity, legal-form agreement or conflict, and house-number edit distance / numeric difference / digit permutation / range containment.
   - Competition context from stage-1 scores: the pair's rank within its query, the margin to the best alternative, and each S1 entity's strongest score and number of strong matches.
5. **Decision.** Each query is assigned to its highest-p2 S1 entity if p2 ≥ threshold. The threshold is tuned for macro F0.5 on the hold-out.

**Validation protocol.** 2% of S1 entities (by hash) are held out, and every query with a candidate among them is moved to validation. The remaining queries are split into disjoint sets for stage 1 and stage 2. Stage-1 scores on the stage-2 and validation sets are therefore out-of-sample, exactly as on test.

`candidate_pairs.tsv` is the full blocking output that stage 1 scores. `matching_results.tsv` is always a subset of it.

## Generalizing to an unseen country (e.g. France)

The test set includes France, which never appears in training. Don't guess a lower decision threshold for it from unlabeled test scores -- comparing score *distributions* across countries is misleading unless you compare the same statistic (e.g. the top-1 pick's score, not the score of every surviving candidate, which is dominated by how many distractors a country's blocking happens to produce).

Instead, get labeled evidence for "unseen country" generalization directly:

```bash
# Train with an entire country held out completely (not just a random 2% of S1),
# so validation measures generalization to a country the model never trained on --
# the same situation France is in at test time.
python3 train.py --work-dir W --gt D/train/train_ground_truth.tsv \
    --model-dir W/models_holdout_india --holdout-country India
```

This prints a real F0.5-vs-threshold curve for India-as-if-unseen. If it shows a materially different optimal threshold than the global model's, write the finding to a small JSON and apply it only where there's evidence for it:

```bash
echo '{"France": 0.55}' > country_thresholds.json
python3 predict.py --work-dir W --model-dir W/models --out-dir OUT \
    --country-thresholds country_thresholds.json
```

Countries not listed silently fall back to the global threshold (`config.json`) -- this is never a hardcoded closed set, so an untested country is never mishandled.
