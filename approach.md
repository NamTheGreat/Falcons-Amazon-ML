# Approach Notes — source material for `Documentation_template.md`

> **Living document.** This holds everything the final 2-page methodology write-up will need, in more detail than the final doc will have room for. Update the relevant section every time the pipeline, features, blocking, or validated numbers change — don't let this drift from `code/business_entity_resolution/src/`. At submission time, compress this into `student_resource/Documentation_template.md`.
>
> Cross-reference: `MASTER_REFERENCE.md` is the engineering/diagnostics log (dataset stats, noise patterns, submission tracking). This file is the outward-facing methodology draft.

**Last updated**: 2026-09-25, after adding country-generalization tooling (`feature/country-generalization` branch, not yet merged/trained).

---

## 1. Executive Summary (draft)

We resolve business entities across three noisy sources with an inverted-index blocking stage (hashed, IDF-weighted, country-scoped keys) feeding a two-stage LightGBM classifier, with a decision threshold tuned for macro F0.5. The pipeline is fully language-agnostic — no country is hard-coded — so it extends to France (absent from training) without special-casing. Current validated performance: **0.9824 macro F0.5** on a held-out slice of training data, and **0.973 on the public leaderboard** (Submissions #2/#3), up from **0.966843** on the initial submission.

---

## 2. Methodology

### 2.1 Problem Analysis (key EDA insights)

- **Key structural fact**: every S2/S3 record matches at most one S1 entity, and ~27% match nothing. This makes the problem tractable from the S2/S3 side: for each S2/S3 record ("query"), find its one best S1 entity, or none.
- **Scale**: train ~2.2M S1 / 5.0M S2 / 5.3M S3; test ~1.7M S1 / 4.9M S2 / 5.1M S3.
- **Countries**: train is US (60%) + India (40%) only. **Test adds France at ~15%, with zero training exposure** — the pipeline must generalize to it without being told anything France-specific in advance.
- **Missing data**: S1 is clean (0% missing). S2/S3 have ~3.3% missing address; name and country are never missing.
- **Noise patterns observed**:
  - Name: abbreviation variants (Pvt/Private, Ltd/Limited, Inc/Incorporated), legal-suffix drops, DBA/trade names (completely different from the S1 name), typos, transliteration to Devanagari/Tamil/Telugu/Kannada, mixed-script names, case changes, URL-style names (`kempgloba.com`).
  - Address: abbreviation variants (St/Street, Rd/Road), component reordering, state-name expansion (including romanized regional-language spellings), landmark-based references, literal `"null"` strings, missing components.
- **Ground truth shape**: 5.6% of S1 entities are singletons (no match) — F0.5 gives these full credit only for an empty prediction, and zero credit for any predicted match. Median match count is 3–4 per S1 entity, max 11.

### 2.2 Solution Strategy

**Approach type**: Blocking + two-stage classifier (hybrid), not end-to-end or embedding-only.

**Core innovation**:
1. Multi-strategy hashed-key blocking (union of 8 key families) that stays language-agnostic by scoping every key to `country` as an open string label, never a fixed set.
2. A **phonetic skeleton** per name token (`sophtveyr` → `sftvr`, same as `software`) that lets English and romanized-Indic-script spellings of the same word collide as blocking keys and features, without any transliteration dictionary.
3. Two-stage classification: a cheap stage 1 scores every candidate; a heavier stage 2 (token-alignment + competition-context features) only runs on the ~13% of pairs stage 1 doesn't confidently reject — keeping compute bounded while adding features that need per-query/per-entity context.
4. Decision rule: each query goes to its single best-scoring S1 entity, only if that score clears a threshold tuned for macro F0.5 (precision-weighted, so the tuned optimum sits high, ~0.68 in the current model).

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** (all hashed, IDF-weighted, and scoped to `country` — never hard-coded to a fixed country set):
1. Name tokens (≥2 chars)
2. Name token bigrams
3. Phonetic skeleton tokens (handles cross-script/typo'd names)
4. Compact (no-space) name prefix/suffix + full compact name (catches URL-style names)
5. Address words (excluding a stopword list of filler terms: `no`, `flat`, `plot`, `floor`, `office`, `cedex`, ...)
6. (street number, neighboring address word) pairs, both orders
7. Unordered address-word pairs
8. Name/phonetic token × address word/number cross-combinations (stays selective even when name and address are individually generic)

**Mechanics**: keys are IDF-weighted over Source 1; any key with S1 document frequency above 400 is dropped (protects against near-universal tokens). Each query probes only its rarest keys per key family (`k_name=5, k_addr=4, k_combo=6, k_apair=4`). Top **10** S1 candidates are kept per query, widened to **24** for queries with no address or a non-Latin name (their blocking signal is weakest, so we spend more candidates there rather than risk losing the true match).

**Candidate pairs generated**: ~9 candidates/query on average.

**Recall preserved**: measured against training ground truth, **98.2% pair recall** on a train sample at the current top-`m` settings (up from 97.2% before the phonetic-skeleton and cross-combo key families were added). Recall breakdown by segment (400k-query sample, earlier blocking version): 97.2% overall, ~97.6% for US, ~93.9% for India, dropping for non-Latin names without an address (~81%) — the has-address, non-Latin-name segment (bulk of transliterated Indian entries) is already ~99.1%, confirming address-based blocking is what rescues cross-script names, not name matching alone.

---

## 4. Matching Model

**Feature groups** (66 features total feeding the final classifier):

- **Stage 1 features (41)** — computed for every candidate pair:
  - *Name*: rapidfuzz ratio / token-set-ratio / token-sort-ratio / partial-ratio / Jaro-Winkler, each run on the core name, full name, DBA/alt name, compact (no-space) name, and phonetic skeleton.
  - *Address*: same similarity family on the normalized address string, plus numeric-token intersection/Jaccard overlap.
  - *Cross/context*: name-alt-to-S1-name overlap (catches DBA names), name/address length ratios, non-Latin-name flag, source flag (S2 vs S3), and label-free frequency priors (how common this exact name/address already is within S1 and within the query pool — a shared generic name or building weakens the evidence of agreeing on it).
  - *Blocking scores*: the four blocking-key-family scores (name/address/combo/address-pair) and rank carried over as features, not discarded.
- **Stage 2 additional features (18 token/legal/number features + 7 context features = 25)**, computed only on pairs stage 1 doesn't confidently reject:
  - *Token alignment*: count of unmatched name tokens in each direction, minimum per-token similarity, count of short/unmatched tokens.
  - *Legal form*: which legal-form tokens (Inc/Ltd/Pvt/SARL/SAS/...) each side has, whether they agree or actively conflict, and a specific Pvt-mismatch flag.
  - *House number*: minimum edit distance and absolute numeric difference across all number pairs, whether one side's numbers are a digit-permutation of the other's, whether the S1 number falls inside a numeric range quoted on the query side.
  - *Competition context*: the pair's rank within its query by stage-1 score, the score margin to the best alternative for that query, and the S1 entity's own strongest score and count of strong (>0.5) matches across all its candidates.

**Model type**: two-stage **LightGBM** binary classifier (`objective=binary`, `num_leaves=127`, `learning_rate=0.08`, `min_data_in_leaf=50`, feature/bagging fraction 0.8). Stage 1 is trained on ~9.5M pairs (rounds=500); stage 2 is trained on ~24M pairs held disjoint from stage 1's training set, with early stopping on a held-out validation slice (rounds up to 5000, best iteration selected automatically). Between stages, pairs with stage-1 score < 0.005 are dropped — keeps ~13% of pairs, loses <0.1% of true pairs — to bound stage-2 compute.

**Threshold selection method**: grid search (0.30–0.95, then a refined pass) for the threshold that maximizes **macro F0.5** on the held-out validation slice, computed identically to the leaderboard metric (per-S1-entity F0.5, singletons included, then averaged).

---

## 5. Results & Error Analysis

- **Local validation macro F0.5**: v3 0.98195 (threshold 0.73) → v4 **0.98240** (threshold 0.68, stage-2 trained on 8M queries instead of 4M). The v3→v4 gain (+0.00045) is within noise: the threshold sweep is flat from 0.60–0.80 (0.98215–0.98240), and feature importances are nearly identical between the two models (`q_margin` dominates both by >2x over the next feature). Doubling stage-2 training data plateaued rather than meaningfully improving discrimination.
- **Public leaderboard**: Submission #1 (v2) **0.966843** (2026-09-25) → Submission #2 (v3) **0.973** → Submission #3 (v4) **0.973** (2026-09-27). #2 and #3 display identically because their real difference (0.00045 local) is smaller than the leaderboard's 3-decimal precision — confirming the plateau on real data, not just validation data.
- **Gap analysis**: the local-validation-vs-leaderboard gap, driven by France (~15% of test, zero training exposure), has stayed roughly the same absolute size across both jumps (v2: 0.9766→0.966843, gap≈0.0098; v3/v4: ~0.982→0.973, gap≈0.009). Directly verified early on this is *not* a simple confidence-miscalibration problem: France's top-1 pick has a median score of 0.999, statistically indistinguishable from US (0.9996) and India (0.9983). The real weakness is a fatter tail of medium-scoring *near-miss distractors* for France (average 1.36 surviving candidates/query vs ~1.0 for US/India) that stage 1 doesn't cleanly reject. The v3/v4 fixes (French postal canonicalization, more training data) raised the whole model's performance but did **not** specifically shrink this France-driven gap — closing it further would need work targeted at that discrimination weakness specifically, not further generic model improvements.
- **Common false positives** (from validation error analysis): near-duplicate S1 entities with a single distinguishing word swapped (e.g. "John Summit *Integrated*" vs "John Summit *Dental*" at the same address), off-by-small-amount house numbers on an otherwise identical address, and generic/common business names colliding at nearby-but-different addresses.
- **Common false negatives**: heavy name corruption *combined with* a missing or very sparse address (the address is normally what rescues a badly mangled name; without it, recall drops sharply — see the 63.8%/47.6% true-positive rates for no-address queries above), and cases where a real match's address round-trips through multiple reordering/abbreviation differences at once.

---

## 6. Conclusion (draft)

A blocking-first, feature-rich two-stage LightGBM pipeline reaches 0.9824 macro F0.5 in local validation and 0.973 on the public leaderboard (up from an initial 0.966843), with the remaining gap to local validation attributable to France's harder candidate-discrimination profile rather than a systematic confidence problem. Two lessons stood out over the course of the challenge: first, diagnosing precision/recall problems on an unlabeled or under-labeled segment (like France) requires checking the actual decision statistic (top-1 pick per query), not aggregate candidate-pool statistics, which can look alarming for entirely benign reasons; second, once the model and features are already reasonably strong, simply scaling training data further (doubling stage-2 rows) plateaus quickly and stops being the highest-leverage lever — the France-specific gap needed a targeted fix, not a bigger version of the same model.

---

## Appendix A: Code Artefacts

Entry points, all under `code/business_entity_resolution/src/`:

| Script | Purpose |
|---|---|
| `prepare.py` | Normalize raw TSVs → parquet (`normalize.py`) |
| `run_blocking.py` | Generate candidate pairs (`blocking.py`) |
| `train.py` | Train stage-1/stage-2 LightGBM, tune threshold; `--holdout-country` for unseen-country validation |
| `predict.py` | Score test candidates, write `matching_results.tsv` / `candidate_pairs.tsv`; `--country-thresholds` for evidence-based per-country cutoffs |
| `features.py`, `token_features.py`, `pipeline.py` | Feature engineering and the shared stage-1→stage-2 scoring procedure |
| `evaluate.py` | Macro F0.5 metric and decision rule |

`run_all.sh` wraps all four pipeline stages end to end. Full reproduction steps and environment setup are in `code/business_entity_resolution/README.md`.

## Appendix B: Additional Results (fill in as they land)

- [x] Per-country threshold check: US and India independently peak at τ=0.73 on the current model — no per-country override applied (reconciled via `scratch/verify_v3_decision_rule.py`: US F0.5 0.98475, India F0.5 0.97779).
- [x] `decide_expected_f` tested against the flat threshold: underperformed (0.98100 vs flat 0.98195), not used.
- [x] `--n-stage2-queries` increase (4M → 8M): **Completed!** Local validation macro F0.5 reached **`0.98240`** (calibrated threshold $\tau = 0.68$, best iteration 2084).
- [x] Final leaderboard score history: #1 (v2) 0.966843 · #2 (v3) 0.973 · #3 (v4) 0.973 (identical to #2 at 3-decimal display precision — confirms the v3→v4 gain was noise-level on real data too). 3/15 submissions used.

---

## Change Log

| Date | Change | Where |
|---|---|---|
| 2026-09-25 | Initial pipeline: normalize → block → 2-stage LightGBM → threshold. Val F0.5 0.9766. | `main` |
| 2026-09-25 | Submission #1: 0.966843 public LB (1/15 used). | — |
| 2026-09-25 | Diagnosed France gap as candidate-discrimination, not threshold miscalibration. Added `--holdout-country` (train.py) and `--country-thresholds` (predict.py) for evidence-based generalization work; fixed missing `cedex`/`bp` French address terms. | `feature/country-generalization` (uncommitted local work / not yet merged into this doc's baseline numbers) |
| 2026-09-27 | v3 pipeline retrained end-to-end on Azure with the above fixes. Val F0.5 0.98195, threshold 0.73. Ready as Submission #2, not yet uploaded. | Azure VM (`run_pipeline_end_to_end.py`) |
| 2026-09-27 | Tested `decide_expected_f` and per-country thresholds against v3's validation set — both rejected (see MASTER_REFERENCE.md §13). | Azure VM |
| 2026-09-27 | v4 trained with `--n-stage2-queries 8000000` (76.5M Stage-2 candidate pairs). Val F0.5 **0.98240** (threshold 0.68). Passed `validate_submission.py` (once run with the correct flags — the automated watcher script used a stale/wrong invocation). Generated `matching_results.tsv` (5,806,496 matches). | Azure VM → `output/` |
| 2026-09-27 | Submissions #2 (v3) and #3 (v4) confirmed on the real leaderboard: **0.973 both**, up from 0.966843. Identical display confirms the v3→v4 gain was noise-level in the real world too, not just locally. Local-vs-real gap (~0.009) unchanged from v2 — the France-driven gap wasn't shrunk by these fixes, only the whole model's baseline was raised. 3/15 submissions used, no further experiments in progress. | Unstop leaderboard |
