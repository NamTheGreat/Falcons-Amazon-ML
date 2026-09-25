"""Score test candidates with the trained models and write the submission files."""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from evaluate import decide
from features import LOAD_COLS, add_freq_columns
from pipeline import score_candidates
from run_blocking import load_split


def write_id_lists(pairs, s1, q, header, path):
    """pairs: (qid, sid). One row per S1 entity, empty list when no pairs."""
    lists = (
        pairs.join(q.select(pl.col("rid").alias("qid"), pl.col("entity_id").alias("mid")), on="qid")
        .group_by("sid").agg(pl.col("mid").unique().sort().str.join(",").alias(header[1]))
    )
    out = (
        s1.select(pl.col("rid").alias("sid"), pl.col("entity_id").alias(header[0]))
        .join(lists, on="sid", how="left")
        .with_columns(pl.col(header[1]).fill_null(""))
        .select(header)
    )
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument(
        "--country-thresholds", default=None,
        help="Optional path to a JSON file mapping country name -> decision "
        "threshold, e.g. {\"France\": 0.55}. Countries not listed fall back "
        "to --threshold / config.json's global value -- never a closed set. "
        "Only set this once you have real evidence a country needs a "
        "different cutoff (e.g. from `train.py --holdout-country`); don't "
        "guess a number from unlabeled test scores.",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    thr = args.threshold if args.threshold is not None else cfg["threshold"]
    country_thr = {}
    if args.country_thresholds:
        with open(args.country_thresholds) as fh:
            country_thr = {k: float(v) for k, v in json.load(fh).items()}
        print(f"country thresholds: {country_thr} (default {thr})", flush=True)
    m1 = lgb.Booster(model_file=os.path.join(args.model_dir, "stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(args.model_dir, "stage2.txt"))

    s1, q = load_split(args.work_dir, args.split, LOAD_COLS)
    s1, q = add_freq_columns(s1, q)
    cands = pl.read_parquet(os.path.join(args.work_dir, f"{args.split}_cands.parquet"))
    print(f"{cands.height:,} candidate pairs", flush=True)

    pairs = cands.select("qid", "sid")
    p2 = score_candidates(cands, q, s1, m1, m2, num_iteration=cfg.get("stage2_best_iter"))
    del cands
    p2.write_parquet(os.path.join(args.work_dir, f"{args.split}_scores.parquet"))
    print(f"scoring done {time.time() - t0:.0f}s", flush=True)

    if country_thr:
        thr_df = pl.DataFrame({"country": list(country_thr.keys()), "_thr": list(country_thr.values())})
        qcountry = q.select(pl.col("rid").alias("qid"), "country")
        scored = (
            p2.join(qcountry, on="qid", how="left")
            .join(thr_df, on="country", how="left")
            .with_columns(pl.col("_thr").fill_null(thr))
        )
        matches = decide(scored, "_thr")
    else:
        matches = decide(p2, thr)
    m = write_id_lists(matches, s1, q, ["source1_entity_id", "matched_entity_ids"],
                       os.path.join(args.out_dir, "matching_results.tsv"))
    write_id_lists(pairs, s1, q, ["source1_entity_id", "candidate_entity_ids"],
                   os.path.join(args.out_dir, "candidate_pairs.tsv"))
    n_empty = (m["matched_entity_ids"] == "").sum()
    thr_desc = f"default {thr:.3f} + overrides {country_thr}" if country_thr else f"{thr:.3f}"
    print(f"threshold {thr_desc}: {matches.height:,} matches, {n_empty:,}/{m.height:,} S1 entities without matches")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
