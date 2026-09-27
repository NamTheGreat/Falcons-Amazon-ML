# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Falcons  
**Team Members:** Naman Kalia, Pawan Jain, Aarohi Tyagi, Tvisha Choudhary  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

We resolve business entities across three noisy sources using a language-agnostic, inverted-index blocking architecture feeding a two-stage LightGBM classifier optimized for the macro-$F_{0.5}$ metric. Key innovations include multi-strategy hashed-key blocking with phonetic skeleton token collisions to bridge cross-script/transliteration variations without dictionaries, out-of-sample two-stage classification with competition context margins, and automated metric-driven threshold calibration. Our solution achieves a validated **0.98240 macro-$F_{0.5}$** on held-out data and **0.973 on the public leaderboard**, with zero external data or APIs.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Core Asymmetry:** Source 1 is deduplicated; every S2/S3 record maps to at most one S1 entity, while ~27% are singletons/distractors matching nothing. Inverting the matching direction (finding the single best S1 entity for each S2/S3 query) drastically simplifies search.
- **Multilingual & Country Distribution:** Training data contains US (60%) and India (40%), while the test set introduces **France (15%) with zero training exposure**. Indian entities frequently exhibit regional script transliterations (Devanagari, Tamil, Telugu, Kannada) and DBA variations; French entries feature unique administrative and postal abbreviations (`cedex`, `bp`, `sarl`, `sas`).
- **Metric Dynamics:** Macro-$F_{0.5}$ weights precision twice as heavily as recall and scores singletons strictly as binary (1.0 if empty, 0.0 if any false match). Thus, false merges are penalized severely, dictating a conservative, precision-focused decision threshold.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Inverted-Index Multi-Strategy Blocking + Two-Stage LightGBM Classifier.  
**Core Innovation:**
1. **Phonetic Skeleton Collision:** A rule-based consonant skeleton reduces transliterated/misspelled tokens to common collision keys (`software` and `sophtveyr` → `sftvr`), enabling cross-script blocking without language-specific translation models.
2. **Two-Stage Cascaded Inference:** A fast Stage-1 classifier scores all candidates; Stage-2 computes deep token-alignment and cross-entity margin features only on surviving candidates ($p_1 \ge 0.005$), achieving linear scalability over 113M candidate pairs.
3. **Language-Agnostic Country Scoping:** All blocking keys and priors are open string keys scoped by country, allowing automatic generalization to unseen countries like France without hardcoded rules.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Emits 8 hashed, country-scoped key families:
  1. Core name tokens ($\ge 2$ chars) and token bigrams
  2. Phonetic skeleton tokens
  3. Compact (no-space) name prefix, suffix, and full string (capturing URL-style names)
  4. Normalized address words (filtering universal stopwords)
  5. Neighboring (house number, address word) ordered and unordered pairs
  6. Name token $\times$ address token cross-combinations
- **Candidate pairs generated:** **113,366,071 candidate pairs** across 9,969,589 test queries (~11.3 candidates/query).
- **Ensuring True Matches Were Not Lost:** Keys are IDF-weighted over Source 1, dropping universal tokens ($DF > 400$). Each query probes its rarest keys per family, retaining the top 10 candidates (expanded to 24 for queries lacking addresses or containing non-Latin names), achieving **98.2% candidate pair recall**.

---

## 4. Matching Model

**Features used (66 total):**
- **Stage 1 Features (41):** RapidFuzz string metrics (Levenshtein ratio, partial ratio, token sort, token set, Jaro-Winkler) across core name, full name, DBA name, compact string, and phonetic skeleton; address token overlaps and numeric Jaccard similarities; frequency priors (document frequency of names/addresses in S1); blocking score aggregates.
- **Stage 2 Features (25):** Directional unmatched token counts, minimum per-token similarity, legal form agreement/conflict matrices (`Inc`, `Ltd`, `Pvt`, `SARL`, `SAS`), house number edit distance, numeric digit permutations, range containment, and competition context (query rank, margin to next best alternative, entity competition counts).

**Model type:** Two-Stage Gradient Boosted Decision Trees (**LightGBM**, `objective=binary`, `num_leaves=127`, `learning_rate=0.08`, `min_data_in_leaf=50`).  
**Threshold selection method:** Grid search directly maximizing the official macro-$F_{0.5}$ metric on an out-of-fold validation slice, selecting the optimal threshold **$\tau = 0.68$**.

---

## 5. Results & Error Analysis

- **Macro-$F_{0.5}$ Score:** **0.98240** in out-of-sample validation (calibrated threshold $\tau = 0.68$, best iteration 2084).
- **Public Leaderboard:** **0.973** (up from 0.967 on the initial baseline). The ~0.009 gap to local validation is concentrated in France (zero training exposure); we verified it is a near-miss-distractor *discrimination* effect, not confidence miscalibration — France's median top-1 score (0.999) is indistinguishable from US/India, but France retains ~1.4 surviving candidates per query vs ~1.0.
- **Decision-layer ablations (validated on held-out data, all rejected):** expected-$F_{0.5}$ optimal top-$k$ selection (0.98100), per-country thresholds (US and India independently peak at the same $\tau$), doubling Stage-2 training data (+0.00045), and post-hoc margin/agreement filters ($\le$ +0.00003). None beat a single global threshold — the decision layer is saturated and the remaining error lies in candidate discrimination.
- **Test Inference Output:** 5,806,496 matches assigned across 1,633,886 S1 entities; 98,658 S1 singletons predicted empty.
- **Common false positives (wrong merges):** Near-duplicate sibling businesses sharing identical addresses with only a single generic descriptor swapped (e.g. "Summit Logistics" vs "Summit Storage" at the same commercial park), or generic corporate names sharing high-density building numbers.
- **Common false negatives (missed matches):** Queries exhibiting extreme name corruption combined with missing or sparse addresses ($<3.3\%$ missing address cases), where neither name nor address provides sufficient blocking overlap.

---

## 6. Conclusion

A blocking-first, feature-engineered two-stage LightGBM architecture delivers a scalable, highly accurate business entity resolution solution reaching 0.98240 validation macro-$F_{0.5}$. By prioritizing precision through metric-aligned threshold calibration and language-agnostic phonetic blocking, the system generalizes seamlessly across multilingual records and unseen countries without external dependencies.

---

## Appendix

### A. Code Artefacts
All code is located in `code/business_entity_resolution/`:
- `src/normalize.py`: Text cleaning, romanization (`anyascii`), abbreviation expansion, phonetic skeleton generation.
- `src/blocking.py`: Sliced-key memory-optimized inverted index candidate generation.
- `src/features.py` & `src/token_features.py`: Feature engineering for Stage 1 and Stage 2.
- `src/train.py`: Model training, out-of-fold evaluation, and $F_{0.5}$ threshold calibration.
- `src/predict.py`: Test candidate scoring, clustering, and TSV output generation.
- `src/validate_submission.py`: Official competition format and integrity checker.
- `run_pipeline_end_to_end.py`: Automated cross-platform runner reproducing all outputs from raw TSVs.
