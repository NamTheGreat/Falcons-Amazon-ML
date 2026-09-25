"""Run blocking for one split: writes <work>/<split>_cands.parquet.

qid indexes rows of concat(s2, s3); sid indexes rows of s1 (both as stored in
the normalized parquet files written by prepare.py).
"""
import argparse
import os
import time

import polars as pl

from blocking import generate_candidates


BLOCK_COLS = ["country", "name_core", "name_alt", "name_compact", "name_skel", "addr", "addr_nums", "nonlatin_name"]


def load_split(work_dir, split, columns=None):
    s1 = pl.read_parquet(os.path.join(work_dir, f"{split}_s1.parquet"), columns=columns).with_row_index("rid")
    q = pl.concat([
        pl.read_parquet(os.path.join(work_dir, f"{split}_s2.parquet"), columns=columns),
        pl.read_parquet(os.path.join(work_dir, f"{split}_s3.parquet"), columns=columns),
    ]).with_row_index("rid")
    return s1, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--top-m", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()
    t = time.time()
    s1, q = load_split(args.work_dir, args.split, BLOCK_COLS)
    c = generate_candidates(s1, q, top_m=args.top_m, chunk=args.chunk)
    out = os.path.join(args.work_dir, f"{args.split}_cands.parquet")
    c.write_parquet(out)
    print(f"{out}: {c.height:,} pairs for {q.height:,} queries in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
