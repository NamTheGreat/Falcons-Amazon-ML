#!/usr/bin/env bash
# End-to-end: raw TSVs -> normalized parquet -> blocking -> training -> test predictions.
# Usage: ./run_all.sh <dataset_dir> <work_dir> <output_dir>
set -euo pipefail
DATA=${1:-../../student_resource/dataset}
WORK=${2:-../../work}
OUT=${3:-../../output}
PY=${PYTHON:-python3}
cd "$(dirname "$0")/src"
$PY prepare.py --data-dir "$DATA" --work-dir "$WORK"
$PY run_blocking.py --work-dir "$WORK" --split train
$PY run_blocking.py --work-dir "$WORK" --split test
$PY train.py --work-dir "$WORK" --gt "$DATA/train/train_ground_truth.tsv" --model-dir "$WORK/models"
$PY predict.py --work-dir "$WORK" --model-dir "$WORK/models" --out-dir "$OUT"
