"""Train the two-stage LightGBM matcher on train candidates and tune the threshold.

Hold-out protocol: by default, 2% of Source 1 entities (by hash) form the
validation set. All queries that have any candidate in the hold-out go to
validation. The rest of the queries are split into disjoint sets A (stage-1
training) and B (stage-2 training), so stage-1 scores on B / validation are
out-of-sample, exactly as on test. The validation score is macro F0.5 over the
hold-out S1 entities, singletons included, computed like the leaderboard
metric.

Pass --holdout-country to hold out an entire country instead (e.g. `India`),
removing it from training completely. That simulates the real France
situation -- a country the model has never seen -- using labeled train data,
so you get an actual F0.5-vs-threshold curve for "unseen country" instead of
guessing a threshold from unlabeled test scores.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from evaluate import decide, macro_f05
from features import LOAD_COLS, PAIR_FEATS, STAGE2_FEATS, add_freq_columns, pair_features
from pipeline import score_candidates
from run_blocking import load_split

PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=10,
)


def fit(X, y, Xv=None, yv=None, rounds=600, params=PARAMS):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    valid = [lgb.Dataset(Xv, yv, reference=dtr)] if Xv is not None else []
    cb = [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(200)] if valid else []
    return lgb.train(params, dtr, rounds, valid_sets=valid, callbacks=cb)


def load_gt_pairs(path, s1, q):
    gt = pl.read_csv(path, separator="\t", infer_schema=False)
    gt = (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
    )
    return (
        gt.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("rid").alias("sid")), on="source1_entity_id")
        .join(q.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("rid").alias("qid")), on="matched_entity_ids")
        .select("qid", "sid")
    )


def tune_threshold(val, gt_hold, hold_sids, col="p2"):
    best = (0.0, 0.5)
    grid = list(np.arange(0.30, 0.96, 0.05))
    for thr in grid:
        sc = macro_f05(decide(val, thr, col), gt_hold, hold_sids)
        print(f"  thr {thr:.2f}: {sc}", flush=True)
        if sc["f05"] > best[0]:
            best = (sc["f05"], float(thr))
    for thr in np.arange(best[1] - 0.04, best[1] + 0.045, 0.01):
        sc = macro_f05(decide(val, thr, col), gt_hold, hold_sids)
        if sc["f05"] > best[0]:
            best = (sc["f05"], float(thr))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n-stage1-queries", type=int, default=1_200_000)
    ap.add_argument("--n-stage2-queries", type=int, default=4_000_000)
    ap.add_argument(
        "--holdout-country", default=None,
        help="Hold out ALL S1 entities of this country from both stage-1 and "
        "stage-2 training (instead of the default random 2%% S1 sample), then "
        "validate on it. Use this to estimate generalization to a country the "
        "model has never trained on -- the same situation France is in at "
        "test time -- before trusting any country-specific decision "
        "threshold. Must match the `country` column exactly, e.g. `India`.",
    )
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    s1, q = load_split(args.work_dir, "train", LOAD_COLS)
    s1, q = add_freq_columns(s1, q)
    gt = load_gt_pairs(args.gt, s1, q)
    cands = pl.read_parquet(os.path.join(args.work_dir, "train_cands.parquet"))
    cands = cands.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["qid", "sid"], how="left").with_columns(pl.col("y").fill_null(0))
    log(f"cands {cands.height:,}  positives {cands['y'].sum():,} / gt {gt.height:,}")

    if args.holdout_country:
        hold = s1.select(pl.col("rid").alias("sid"), (pl.col("country") == args.holdout_country).alias("hold"))
        n_hold = hold["hold"].sum()
        if n_hold == 0:
            raise SystemExit(f"--holdout-country {args.holdout_country!r} matches no S1 entities; check spelling/case.")
        log(f"holding out ALL of country={args.holdout_country!r}: {n_hold:,} / {s1.height:,} S1 entities (unseen-country generalization test)")
    else:
        hold = s1.select(pl.col("rid").alias("sid"), (pl.col("entity_id").hash(seed=11) % 50 == 0).alias("hold"))
    val_q = cands.join(hold.filter("hold"), on="sid", how="semi").select("qid").unique()
    rest_q = cands.select("qid").unique().join(val_q, on="qid", how="anti").sample(fraction=1.0, shuffle=True, seed=0)
    a_q = rest_q.head(args.n_stage1_queries)
    b_q = rest_q.slice(args.n_stage1_queries, args.n_stage2_queries)
    val = cands.join(val_q, on="qid", how="semi")
    A = cands.join(a_q, on="qid", how="semi")
    B = cands.join(b_q, on="qid", how="semi")
    del cands
    log(f"A pairs {A.height:,}  B pairs {B.height:,}  val pairs {val.height:,}")

    # ---- stage 1
    FA = pl.concat([pair_features(A.slice(i, 2_000_000), q, s1) for i in range(0, A.height, 2_000_000)])
    m1 = fit(FA.select(PAIR_FEATS).to_numpy().astype(np.float32), FA["y"].to_numpy(), rounds=500)
    m1.save_model(os.path.join(args.model_dir, "stage1.txt"))
    del FA, A
    log("stage1 trained")

    # ---- stage 2 frames (identical procedure to test time)
    TB = score_candidates(B, q, s1, m1)
    del B
    log(f"stage2 train frame {TB.height:,} rows, positives {TB['y'].sum():,}")
    TV = score_candidates(val, q, s1, m1)
    del val
    log(f"stage2 val frame {TV.height:,} rows")
    Xv = TV.select(STAGE2_FEATS).to_numpy()
    m2 = fit(TB.select(STAGE2_FEATS).to_numpy(), TB["y"].to_numpy(), Xv, TV["y"].to_numpy(), rounds=5000)
    m2.save_model(os.path.join(args.model_dir, "stage2.txt"))
    TV = TV.with_columns(pl.Series("p2", m2.predict(Xv, num_iteration=m2.best_iteration).astype(np.float32)))
    log(f"stage2 trained (best iter {m2.best_iteration})")

    hold_sids = hold.filter("hold")["sid"]
    gt_hold = gt.join(hold.filter("hold"), on="sid", how="semi")
    f05, thr = tune_threshold(TV, gt_hold, hold_sids)
    log(f"best thr {thr:.2f}  val macro F0.5 {f05:.5f}")
    imp = sorted(zip(STAGE2_FEATS, m2.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, int(g)) for f, g in imp[:25]])
    with open(os.path.join(args.model_dir, "config.json"), "w") as fh:
        json.dump({
            "threshold": thr, "val_f05": f05, "stage2_best_iter": m2.best_iteration,
            "holdout_country": args.holdout_country,
        }, fh, indent=1)
    TV.write_parquet(os.path.join(args.work_dir, "val_frame.parquet"))


if __name__ == "__main__":
    main()
