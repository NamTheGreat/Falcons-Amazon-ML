"""Shared scoring procedure used identically for validation and test.

1. stage-1 LightGBM on cheap string/blocking features for *all* candidates
2. per-query / per-entity context from stage-1 scores (over all candidates)
3. keep pairs with p1 >= PREFILTER (~13% of pairs, ~0.07% of true pairs lost)
4. add token/legal/number features for kept pairs; stage-2 LightGBM on those
"""
import numpy as np
import polars as pl

from features import CTX_FEATS, PAIR_FEATS, STAGE2_FEATS, add_token_features, context_features, pair_features

PREFILTER = 0.005


def stage1(cands, q, s1, m1, chunk=2_000_000, verbose=True):
    """Returns (all_scores[qid,sid,p1], kept pairs with stage-1 features)."""
    scores, kept = [], []
    extra = [c for c in ("y",) if c in cands.columns]
    for i in range(0, cands.height, chunk):
        f = pair_features(cands.slice(i, chunk), q, s1)
        f = f.select(["qid", "sid"] + extra + [pl.col(c).cast(pl.Float32) for c in PAIR_FEATS])
        p = m1.predict(f.select(PAIR_FEATS).to_numpy()).astype(np.float32)
        f = f.with_columns(pl.Series("p1", p))
        scores.append(f.select("qid", "sid", "p1"))
        kept.append(f.filter(pl.col("p1") >= PREFILTER))
        if verbose and (i // chunk) % 10 == 0:
            print(f"    stage1 {i + f.height:,}/{cands.height:,}", flush=True)
    return pl.concat(scores), pl.concat(kept)


def stage2_frame(scores, kept, q, s1):
    ctx = context_features(scores).filter(pl.col("p1") >= PREFILTER).select(["qid", "sid"] + CTX_FEATS[1:])
    kept = kept.join(ctx, on=["qid", "sid"], how="left")
    kept = add_token_features(kept, q, s1)
    return kept.with_columns([pl.col(c).cast(pl.Float32) for c in STAGE2_FEATS])


def score_candidates(cands, q, s1, m1, m2=None, num_iteration=None):
    scores, kept = stage1(cands, q, s1, m1)
    del cands
    df = stage2_frame(scores, kept, q, s1)
    if m2 is not None:
        p2 = m2.predict(df.select(STAGE2_FEATS).to_numpy(), num_iteration=num_iteration)
        df = df.with_columns(pl.Series("p2", p2.astype(np.float32)))
    return df
