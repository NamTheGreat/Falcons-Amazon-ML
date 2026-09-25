"""Normalize all raw source TSVs into parquet files under the work dir."""
import argparse
import os
import time

import polars as pl

from normalize import load_source, normalize_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dataset/ folder with train/ and test/")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    for split in args.splits.split(","):
        for s in (1, 2, 3):
            out = os.path.join(args.work_dir, f"{split}_s{s}.parquet")
            if os.path.exists(out):
                print("exists", out)
                continue
            t = time.time()
            raw = load_source(os.path.join(args.data_dir, split, f"{split}_source{s}.tsv"))
            parts = [normalize_frame(raw.slice(i, 1_000_000)) for i in range(0, raw.height, 1_000_000)]
            pl.concat(parts).write_parquet(out)
            print(f"{out}: {raw.height} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
