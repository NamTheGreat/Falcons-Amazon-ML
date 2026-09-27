# Amazon ML Challenge 2026 — Master Reference

> **Last Updated**: 2026-09-27 21:30 IST
> **Challenge Window**: Sep 25, 12:00 AM IST → Sep 27, 11:59 PM IST (Final Day)
> **Target Score**: F₀.₅ > 98.0
> **Submissions**: #1 (v2, thr 0.68) = 0.966843 public LB · #2 (v3, thr 0.73) = **0.973** public LB, 09/27 5:05 PM IST · #3 (v4, thr 0.68) = **0.973** public LB, 09/27 9:17 PM IST
> **Best Public LB Score**: **0.973** (Submissions #2 and #3 — identical at 3-decimal display precision; see §13 for why). **3/15 total submissions used (2 used today, 3 remaining today, 12 remaining overall).**

---

## 1. Problem Statement

**Business Entity Resolution** — given business records from 3 independent noisy sources with no shared identifiers, determine which records refer to the same real-world business entity.

- **Source 1** is the deduplicated reference. For each S1 entity, find all matching records from S2 and S3.
- A Source 1 entity may match **zero, one, or many** records from S2 and S3.
- Fields: `entity_id`, `business_name`, `business_address`, `country`
- Source is determined by ID prefix: `S1-`, `S2-`, `S3-`

---

## 2. Dataset Details

### 2.1 Scale

| Split | Source 1 | Source 2 | Source 3 | Ground Truth |
|:------|--------:|--------:|--------:|------------:|
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 | 2,206,821 |
| **Test** | 1,732,544 | 4,887,273 | 5,082,316 | — |

Total training records: ~12.5M. Total test records: ~11.7M.

### 2.2 Country Distribution

| Country | Train S1 | Train S2 | Train S3 | Test S1 | Test S2 | Test S3 |
|:--------|--------:|--------:|--------:|-------:|-------:|-------:|
| **US** | 60.0% | 59.9% | 60.0% | 38.3% | 38.3% | 38.3% |
| **India** | 40.0% | 40.1% | 40.0% | 46.8% | 47.3% | 47.3% |
| **France** | 0% | 0% | 0% | **15.0%** | **14.4%** | **14.4%** |

> [!IMPORTANT]
> France is **completely unseen** in training. The pipeline must be language-agnostic and not hardcode country-specific logic.

### 2.3 Ground Truth Statistics

| Matches per S1 entity | Count | % |
|:---|---:|---:|
| 0 (singletons) | 123,247 | 5.6% |
| 1 | 119,157 | 5.4% |
| 2 | 375,212 | 17.0% |
| 3 | 530,841 | 24.1% |
| 4 | 484,115 | 21.9% |
| 5 | 321,957 | 14.6% |
| 6 | 164,868 | 7.5% |
| 7 | 63,968 | 2.9% |
| 8 | 18,680 | 0.8% |
| 9 | 4,205 | 0.2% |
| 10 | 534 | 0.0% |
| 11 | 37 | 0.0% |

- **Median matches**: ~3-4 per S1 entity
- **Singletons only 5.6%** — most S1 entities have matches
- **Max 11 matches** per S1 entity

### 2.4 Missing Values

| Source | Missing Name | Missing Address | Missing Country |
|:-------|:-----------:|:--------------:|:--------------:|
| Train S1 | 0% | 0% | 0% |
| Train S2 | 0% | **3.36%** (~169K) | 0% |
| Train S3 | 0% | **3.33%** (~176K) | 0% |

> S1 is clean. S2/S3 have ~3.3% missing addresses. Country and name are never missing.

---

## 3. Noise Patterns Observed

### 3.1 Name Variations
From actual matched pair analysis:

| Pattern | Example (S1 → S2/S3) |
|:--------|:---------------------|
| **Abbreviation** | "Acme Robotics Inc." → "Acme Robotics Incorporated" |
| **Suffix dropped** | "Acme Robotics Inc." → "Acme Robotics" |
| **Typos** | "Payne Enterprises" → "Payne Enterpires", "Payne Etrepndiels", "PAYNE-ENRTPRMISES" |
| **Accented chars** | "Payne Enterprises" → "Payne Énterprises" |
| **URL-style** | "Maure Williams Colombier Inc" → "maurewilliamscolombier.com" |
| **DBA/trade name** | "Maure Williams Colombier Inc" → "Dréxkor" (completely different name!) |
| **Extra words** | "Maure Williams Colombier Inc" → "Maure Williams Inc Center" |
| **Transliteration** | "Raj Investments LLP" → "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" (Tamil) |
| | "Ss Food Private Limited" → "एसएस फूड प्राइवेट लिमिटेड" (Hindi) |
| **Mixed script** | "Raj Investments எல்எல்பி" (English + Tamil) |
| **Case changes** | "Dahlia Power Reliable Scientific LLC" → all caps, mixed case |

### 3.2 Address Variations

| Pattern | Example |
|:--------|:--------|
| **Abbreviation** | "Street" → "St", "Avenue" → "Ave", "Road" → "Rd" |
| **Component reordering** | "630 45th Terrace, Kansas City, MO" → "KANSAS CITY, MO, 630 45ND TERRACE, null" |
| **State expansion** | "NC" → full state name, "TN" → "தமிழ்நாடு" (Tamil for Tamil Nadu) |
| **Completely different** | "500 Market St, San Jose" → "Nr. City Hall, San Jose" (landmark-based!) |
| **Null values** | Literal "null" strings in addresses |
| **Missing** | 3.3% of S2/S3 have empty address |
| **Typos in address** | "Ticonderoga" → "Ticonderoga Townshiip" (double i) |
| **Number variations** | "45th" → "45ND" |

### 3.3 Script/Language Patterns

| Country | Scripts Observed |
|:--------|:-----------------|
| US | English only |
| India | English + Devanagari (Hindi) + Tamil + Kannada + possibly others |
| France | French (with accents: é, è, ê, ë, à, ç, etc.) |

Indian sources frequently have **both** an English name AND a transliterated version in S2/S3. This means for India:
- S1 is in English
- S2 may have the name in Devanagari/regional script OR English
- S3 may have the name in regional script OR mixed English+regional

---

## 4. Evaluation Metric

### F₀.₅ (Precision-Heavy)

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Macro-averaged**: computed per S1 entity, then averaged across ALL S1 entities
- **Precision weighted 2× over recall**: a false merge (merging different businesses) costs ~2× a missed match
- **Singletons included**: correctly predicting empty → 1.0; predicting any match for a singleton → 0.0
- **Rule of thumb**: "When unsure, do NOT merge"

### What 98+ Means
- Essentially zero false positives allowed
- Can afford ~5% false negatives (missed matches)
- Must correctly identify nearly all singletons
- Must handle transliteration, DBA names, wild typos

---

## 5. Output Format

### 5.1 matching_results.tsv (SCORED on leaderboard)
```
source1_entity_id\tmatched_entity_ids
S1-00001\tS2-00047,S2-00193,S3-00812
S1-00002\tS3-00004
S1-00003\t
```
- One row per test S1 entity (all 1,732,544 must be present)
- Empty `matched_entity_ids` for singletons
- No duplicate IDs within a list
- Only S2-/S3- IDs allowed

### 5.2 candidate_pairs.tsv (NOT scored, audit only)
```
source1_entity_id\tcandidate_entity_ids
S1-00001\tS2-00047,S2-00193,S3-00812,S3-00999
S1-00002\tS3-00004
S1-00003\t
```
- Same format as matching_results.tsv
- Should be a superset of matching_results.tsv
- Represents what was fed to the matching model for inference

### 5.3 Final Submission Package
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

### 5.4 Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Use `--check-ids` for full validation (memory-heavy).

---

## 6. Constraints

1. Model ≤ **8 Billion parameters**, **MIT/Apache 2.0** license only
2. **No external data** — no APIs, no geocoding, no business lookup databases
3. **5 submissions/day** max (15 total across 3 days)
4. All files **tab-separated** (.tsv), read with `sep="\t"`
5. Every test S1 entity must appear in submission
6. No self-matches (S1 IDs in matched list)
7. No duplicate IDs

---

## 7. Strategy

### 7.1 Pipeline Architecture

```
Data Loading → Preprocessing → Blocking → Feature Engineering → ML Matching → Threshold Tuning → Output
```

### 7.2 Stage Details

#### Stage 1: Preprocessing
- Lowercase, strip whitespace
- Expand abbreviations (Pvt→Private, Ltd→Limited, Corp→Corporation, etc.)
- Handle French abbreviations (SARL, SAS, SCI, SA, Sté)
- Remove punctuation
- Collapse whitespace
- For transliteration: consider `indic-transliteration` or `transliterate` libraries, OR rely on address overlap for cross-script matches

#### Stage 2: Blocking (MOST CRITICAL — sets recall ceiling)
Multiple strategies, **union** the results:
1. **Country filter** — only compare within same country (instant reduction)
2. **TF-IDF char n-grams on name** — top-K nearest neighbors per S1
3. **TF-IDF char n-grams on address** — catches DBA/trade name cases
4. **Token overlap on name** — fast pre-filter for shared words
5. **Shared numeric tokens in address** — street numbers are strong signals

Target: 10-50 candidates per S1 entity.

> [!IMPORTANT]
> The official video explicitly says: "Blocking sets your recall ceiling — invest there first."

#### Stage 3: Feature Engineering (per candidate pair)
**Name features:**
- Jaro-Winkler similarity
- Levenshtein ratio (rapidfuzz)
- Jaccard on character 3-grams
- Token set ratio (rapidfuzz)
- Partial ratio (rapidfuzz)
- Name length ratio
- Common token count

**Address features:**
- Same set as name features
- Shared numeric tokens (street numbers, PIN codes)
- Token overlap ratio

**Cross features:**
- Name-to-address overlap (catches URL-style names like "maurewilliamscolombier.com")
- Combined text similarity

**Script/language features:**
- Same-script flag
- Country match flag

**Target: 15-20 features per pair**

#### Stage 4: ML Matching
- **LightGBM** binary classifier (match probability)
- Train on labeled training pairs (positive from ground truth, negative from blocking non-matches)
- Cross-validation on training data

#### Stage 5: Threshold Tuning
- Optimize threshold for F₀.₅ on validation split
- Expected optimal threshold: 0.7-0.9 (high, because precision matters)
- **When unsure, do NOT merge**

### 7.3 Key Technical Challenges

| Challenge | Impact | Mitigation |
|:----------|:-------|:-----------|
| **Multi-script transliteration** | Blocks matching Hindi/Tamil names to English | Address-based blocking as fallback; transliteration library |
| **DBA/trade names** | Completely different business names | Address-based blocking catches these |
| **Scale (2.2M × 10M)** | Can't brute-force | Efficient TF-IDF blocking with sparse matrices |
| **France (unseen)** | No training data | Language-agnostic features; character-level similarity works across languages |
| **Missing addresses (3.3%)** | Can't use address for blocking | Name-only blocking as fallback for missing-address records |
| **Address reordering** | Token order varies | Token-set-based similarity (order-independent) |

---

## 8. Iteration Plan

| # | Target F₀.₅ | Focus | When |
|:--|:-----------|:------|:-----|
| **Sub 1** | 88-93 | Working baseline: TF-IDF blocking + string features + LightGBM | Day 1 (today) |
| **Sub 2** | 93-96 | Fix blocking gaps, add transliteration, tune threshold | Day 1 night |
| **Sub 3** | 96-97 | Error analysis → fix specific failure modes, more features | Day 2 |
| **Sub 4-5** | 97-98+ | Fine-tune everything, ensemble, edge case handling | Day 2-3 |

---

## 9. Compliant Libraries

| Library | License | Purpose |
|:--------|:--------|:--------|
| `rapidfuzz` | MIT | Fast string similarity (Levenshtein, Jaro-Winkler, etc.) |
| `sentence-transformers` | Apache 2.0 | Semantic embeddings (`all-MiniLM-L6-v2`, ~23M params) |
| `faiss-cpu` / `faiss-gpu` | MIT | ANN search for blocking |
| `lightgbm` | MIT | Gradient boosted tree classifier |
| `xgboost` | Apache 2.0 | Alternative GBDT |
| `catboost` | Apache 2.0 | Alternative GBDT |
| `scikit-learn` | BSD-3 | TF-IDF, evaluation metrics |
| `pandas` | BSD-3 | Data manipulation |
| `numpy` | BSD-3 | Numerical operations |
| `tqdm` | MIT/MPL | Progress bars |

All ≤ 8B parameters. All MIT/Apache 2.0/BSD compatible.

---

## 10. File Paths

### Project
- **Workspace**: `/Users/namankalia/Documents/amazon ML challenge/AML_Codebase/`
- **Dataset**: `/Users/namankalia/Documents/amazon ML challenge/AML_Codebase/student_resource/dataset/`
- **Validation script**: `/Users/namankalia/Documents/amazon ML challenge/AML_Codebase/student_resource/utils/validate_submission.py`
- **Doc template**: `/Users/namankalia/Documents/amazon ML challenge/AML_Codebase/student_resource/Documentation_template.md`

### Challenge Resources
- **Problem Statement PDF**: `/Users/namankalia/Documents/amazon ML challenge/6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`
- **Guidelines PDF**: `/Users/namankalia/Documents/amazon ML challenge/6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`
- **Video**: `/Users/namankalia/Documents/amazon ML challenge/6ab509c5b7036_ml_challenge_2026_video.mp4`

### Data Files
```
dataset/train/train_source1.tsv    (200MB, 2.2M rows)
dataset/train/train_source2.tsv    (467MB, 5.0M rows)
dataset/train/train_source3.tsv    (480MB, 5.3M rows)
dataset/train/train_ground_truth.tsv (121MB, 2.2M rows)
dataset/test/test_source1.tsv      (167MB, 1.7M rows)
dataset/test/test_source2.tsv      (486MB, 4.9M rows)
dataset/test/test_source3.tsv      (483MB, 5.1M rows)
```

---

## 11. User Context

- **Team**: 4 people — user is lead technical, one more technical person, two for docs/submission
- **Compute**: Google Colab Pro (T4/A100 GPU, 51-83GB RAM)
- **Local machine**: macOS
- **Git rules**: Never auto-commit, NEVER push
- **Submissions used so far**: 1 / 15 (Submission #1: 0.966843 public LB, 2026-09-25 16:36 IST)

---

## 12. Sample Matched Pairs (for reference)

### US Example — Typos & Variations
```
S1: Payne Enterprises | 3315 Fremont Street, Peoria, IL | US
  → S2: Payne Énterprises      | 3315 FREMONT ST, PEORIA, IL
  → S2: Payne Enterpires        | 3315 FREMONT ST, PEORIA, IL
  → S2: PAYNE-ENRTPRMISES       | 3315 FREMONT SAINT, PEORIA, IL
  → S3: Payne Etrepndiels       | 3315 Fremont St, Peoria, Illinois
  → S3: Payne Énterprises       | 3315 Fremont Street, Peoria, Illinois
  → S3: Payne Enterprises  LLC  | Fremont St, Peoria, Illinois
```

### India Example — Transliteration
```
S1: Raj Investments LLP | 6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu | India
  → S2: ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி | 6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu
  → S2: Raj Investments LLP                   | same address, different case
  → S3: Raj Investments எல்எல்பி              | same address, TN abbreviation
  → S3: ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி     | same address, தமிழ்நாடு (Tamil for Tamil Nadu)
```

### US Example — DBA / Trade Name + Missing Address
```
S1: Maure Williams Colombier Inc | 85 Wayne Avenue, Ticonderoga, NY | US
  → S2: Maure Wilblims Colombier Inc | (empty address!)
  → S2: Maure Williams Colombier     | (empty address!)
  → S3: Dréxkor                      | 85 Wanye Avenue, Ticonderoga Townshiip, New York  ← COMPLETELY DIFFERENT NAME!
  → S3: maurewilliamscolombier.com   | Wayne Ave, Ticonderoga Townshiip, New York        ← URL-style name
  → S3: Maure Williams Inc Center    | (empty address!)
```

---

## 13. Diagnostics Log

### 2026-09-25 — France gap: not a threshold problem, verified from Submission #1's scores

Local hold-out validation (random 2% of S1, US+India only) scored 0.9766, but the real leaderboard (Submission #1, includes France) came back at **0.9668**. Hypothesis: the ~0.01 gap is concentrated in France, since France has zero training exposure.

A follow-up plan proposed lowering France's decision threshold from 0.68 to ~0.50, citing "French candidates have a median probability of 0.109 vs 0.99 for US/India." Checked directly against `test_scores.parquet`:

| Country | median **top-1** p2 (what gets submitted) | median p2 across **all** surviving candidates |
|:--|--:|--:|
| US | 0.9996 | 0.989 |
| India | 0.9983 | 0.990 |
| France | 0.9994 | **0.109** |

**Finding**: the 0.109 figure is real, but it's the median over every candidate a query has (distractors included), not the model's confidence in its actual top pick. France's top-1 confidence is statistically indistinguishable from US/India, and a similar fraction of French queries already clear the current 0.68 threshold (65.3% vs 65.7% US / 63.9% India). France does have a fatter tail of medium-scoring near-miss candidates that stage 1 doesn't cleanly reject (avg surviving candidates/query: 1.36 for France vs ~0.96–1.0 for US/India) -- a real, worth-fixing weakness in candidate *discrimination*, but not evidence that the true match itself scores too low.

**Conclusion**: don't lower France's threshold blindly -- with the top-1 pick already this confident, a lower threshold mainly pulls in the risky 0.50-0.68 band, which is exactly the "unsure" territory F0.5 penalizes 2x for a false merge. No labeled France data exists to verify a new threshold before spending a submission on it.

**Follow-up built**: `train.py --holdout-country <country>` holds out an entire country from training (e.g. train on US-only, validate on India-as-if-unseen) to get a real, labeled F0.5-vs-threshold curve for an "unseen country" scenario, instead of guessing from unlabeled test-score distributions. `predict.py --country-thresholds <json>` applies per-country overrides with a safe default fallback for any country not listed (never a hardcoded closed set). Also fixed two genuinely-missing French address terms in `normalize.py`: `cedex` (postal-routing suffix, now stripped) and `bp`/`boite postale` (now canonicalized like `post office`). French legal forms (`sarl`, `sas`, `sci`, `eurl`, ...) were already handled before this fix.

**Next step (to run on Colab, not locally)**: use `--holdout-country` to get evidence-based per-country thresholds before spending another submission on threshold changes, and re-run blocking + training with the improved blocker (98.2% recall on a train sample vs 97.2% previously) and the frequency features already in `features.py`.

### 2026-09-27 — Submission #2 generated: v3 pipeline with 0.98195 validation F0.5

- **Execution**: Ran end-to-end on an Azure Ubuntu VM (`Standard_E4as_v4`, 4 vCPUs, 32 GB RAM) via automated runner `run_pipeline_end_to_end.py`.
- **Key Pipeline Improvements**:
  - Multi-family blocking (8 families + phonetic skeleton token collision) capped with memory-sliced key generation.
  - Enhanced French normalization (`cedex`, `bp`, `boite postale` canonicalization).
  - 66 features computed across 2-stage LightGBM architecture.
  - Out-of-sample Stage 2 training protocol with early stopping.
- **Results**:
  - Candidate pairs: 113,366,071 pairs for 9,969,589 queries.
  - Tuned decision threshold: **0.73** (up from 0.68).
  - Validation macro-$F_{0.5}$: **0.98195** (stage 2 best iteration 2117).
  - Test predictions: 5,774,573 matches across 1,633,992 S1 entities; 98,552 singletons (0 matches).
  - Validation: Passed all official competition validator checks (`validate_submission.py`).
- **Artifacts**: Downloaded to `submissions/v3_val098195/`, `output/matching_results.tsv`, and `~/Downloads/matching_results.tsv`.
- **Status: not yet uploaded to the leaderboard as of this entry.** Upload it as Submission #2 — it's already validated and it's the safe fallback while further work continues.

### 2026-09-27 — Decision-rule / country-threshold experiments (pre-existing v3 model, no retraining)

Quick follow-ups run directly on v3's `val_frame.parquet` (no retraining):
- **`decide_expected_f`** (Poisson-binomial expected-F0.5 selection, built earlier but never wired in) **underperforms the flat threshold**: reported flat τ=0.73 F0.5 0.98773 vs. `decide_expected_f` 0.98680. Plausible mechanism: LightGBM scores aren't well-calibrated probabilities at the tails, and the rule's independence-across-candidates assumption doesn't hold (candidates for the same query share context features). **Decision: keep the flat threshold, don't wire this in.**
- **Country-specific thresholds aren't indicated**: slicing the existing (US+India) validation set by country, both independently peak at τ=0.73 (US F0.5 0.98873, India F0.5 0.98624). Weaker evidence than a true unseen-country test would be (both countries were trained on), but sufficient given the time budget. **Decision: one global threshold, no `country_thresholds.json`.**
- **Reconciliation complete**: `scratch/verify_v3_decision_rule.py` was executed directly on the Azure VM against `work/val_frame.parquet` using the canonical `evaluate.macro_f05()`:
  - Flat threshold $\tau = 0.73$: exactly **`0.98195`** ($P = 0.99596, R = 0.95541$), perfectly matching `config.json`. (US-only: `0.98475`, India-only: `0.97779`).
  - `decide_expected_f`: **`0.98100`** ($P = 0.99576, R = 0.95444$), strictly inferior to flat thresholding.
### 2026-09-27 — v4 Model Retrained with 8M Stage-2 Queries (New Best Model)

- **Training**: `--n-stage2-queries 8000000` (76,461,553 Stage-2 candidate pairs; 7,526,194 Stage-2 training rows, 5,133,430 positives).
- **Validation**:
  - Macro $F_{0.5}$: **`0.98240`** (calibrated threshold $\tau = 0.68$, best iteration 2084).
  - Outperformed v3 (`0.98195`) and baseline v2 (`0.9766`).
  - Threshold curve: plateau between 0.60 and 0.80 (0.98215 - 0.98240), confirming stability across confidence regions.
- **Test Inference**:
  - Scored all 113,366,071 test candidate pairs.
  - Generated **5,806,496 matches** across **1,633,886 Source 1 entities**; **98,658 singletons**.
  - Validation script output: `PASS — no blocking issues found. Safe to submit.`
- **Artifacts**:
  - Saved to `output/matching_results.tsv` (93 MB), `output/matching_results.zip` (40 MB), and `~/Downloads/matching_results.zip`.
  - Model checkpoints stored in `submissions/v4_val09824/models/` and `work/models/`.

### 2026-09-27 — Submissions #2 (v3) and #3 (v4) confirmed on the real leaderboard: both 0.973

Checked live (SSH into the Azure VM, plus the Unstop submissions panel):

- **v4's `predict.py` finished cleanly**: `scoring done 3083s`, threshold 0.680 → 5,806,496 matches, 98,658/1,732,544 S1 entities with no match. But the automated watcher script (`run_v4_predict_when_ready.sh`) called `validate_submission.py` with the **wrong flag names and a wrong path** (`--submission`/`--test-queries` instead of `--matching`/`--candidate`/`--test-dir`, and a nonexistent `test_queries.tsv`), so it errored out immediately without actually validating anything. **Re-ran it with the correct flags directly: PASS — no blocking issues found, safe to submit.** (Fix the watcher script if it's reused — the real validator's flags are `--matching`, `--candidate`, `--test-dir`.)
- **Leaderboard**: Submission #2 (v3, threshold 0.73) = **0.973**. Submission #3 (v4, threshold 0.68) = **0.973** — displayed identically because the real underlying difference (local val 0.98195 vs 0.98240, a gap of 0.00045) is smaller than the leaderboard's 3-decimal display precision. **Not a bug — this is exactly what the flat local-validation threshold curve predicted**: doubling stage-2 training data (the whole point of v4) is now confirmed dead on *both* local validation and the real leaderboard. Don't spend further time pushing `--n-stage2-queries` higher.
- **The local-val-vs-real-LB gap has stayed roughly constant in absolute terms across both jumps**: v2 was 0.9766 local → 0.966843 real (gap ≈0.0098); v3/v4 are ~0.982 local → 0.973 real (gap ≈0.009). The France-normalization fixes and the extra training data raised the *whole* model (local and real both went up together), but did **not** shrink the France-specific gap itself — it's still sitting there at almost the same size. To actually close it, per the 2026-09-25 diagnosis, needs work specifically targeting France's near-miss-candidate discrimination, not further generic model improvements.
- **State as of this entry**: 3/15 submissions used (2 today, 3 remaining today). Current best/live submission: 0.973 (v3 or v4, tied). No further experiments in progress.

---

*End of reference document. Update this file as the challenge progresses.*
