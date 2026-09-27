#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - 1-Click End-to-End Pipeline Runner
Runs cross-platform (Windows, Linux, macOS).
"""
import os
import sys
import time
import zipfile
import urllib.request
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
WORK_DIR = os.path.join(BASE_DIR, "work")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(WORK_DIR, "models")
DATASET_URL = "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"


def run_cmd(cmd, desc):
    print(f"\n{'='*70}\n>>> {desc}\nCMD: {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {res.returncode}: {desc}", file=sys.stderr)
        sys.exit(res.returncode)
    print(f">>> [SUCCESS] {desc} in {time.time() - t0:.1f}s", flush=True)


def download_dataset():
    if os.path.exists(os.path.join(DATA_DIR, "train", "train_source1.tsv")):
        print(">>> Dataset already present in dataset/ folder. Skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "dataset.zip")
    if not os.path.exists(zip_path):
        print(f">>> Downloading dataset from {DATASET_URL} (1.1 GB)...", flush=True)
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print(">>> Download complete.", flush=True)

    print(">>> Extracting dataset.zip...", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Flatten nested folder if student_resource/dataset exists
    nested = os.path.join(DATA_DIR, "student_resource", "dataset")
    if os.path.exists(nested):
        import shutil
        for split in ["train", "test"]:
            src_split = os.path.join(nested, split)
            dst_split = os.path.join(DATA_DIR, split)
            if os.path.exists(src_split) and not os.path.exists(dst_split):
                shutil.move(src_split, dst_split)
    print(">>> Dataset extracted successfully.")


def main():
    print(f"=== Amazon ML Challenge 2026 Pipeline ===")
    print(f"Working Directory: {BASE_DIR}")
    print(f"Python: {sys.executable}")
    
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Ensure Dataset
    download_dataset()

    # Step 2: Normalization (TSV -> Parquet)
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "prepare.py"), "--data-dir", DATA_DIR, "--work-dir", WORK_DIR],
        "Step 1: Normalizing Raw TSVs to Parquet"
    )

    # Step 3: Train Blocking (Candidate Generation)
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "run_blocking.py"), "--work-dir", WORK_DIR, "--split", "train"],
        "Step 2: Generating Train Candidates (Blocking)"
    )

    # Step 4: Model Training & Threshold Tuning
    gt_path = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "train.py"), "--work-dir", WORK_DIR, "--gt", gt_path, "--model-dir", MODELS_DIR],
        "Step 3: Training LightGBM Matchers & Tuning F0.5 Threshold"
    )

    # Step 5: Test Blocking
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "run_blocking.py"), "--work-dir", WORK_DIR, "--split", "test"],
        "Step 4: Generating Test Candidates (Blocking)"
    )

    # Step 6: Test Inference (Scoring & Clustering)
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "predict.py"), "--work-dir", WORK_DIR, "--model-dir", MODELS_DIR, "--out-dir", OUTPUT_DIR],
        "Step 5: Test Inference & Submission File Generation"
    )

    # Step 7: Submission Validation
    match_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cand_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    test_dir = os.path.join(DATA_DIR, "test")
    run_cmd(
        [sys.executable, os.path.join(SRC_DIR, "validate_submission.py"), "--matching", match_file, "--candidate", cand_file, "--test-dir", test_dir],
        "Step 6: Validating Submission TSV Files"
    )

    # Step 8: Package Results
    out_zip = os.path.join(BASE_DIR, "submission_output.zip")
    print(f"\n>>> Zipping final submission files into {out_zip}...", flush=True)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(match_file, arcname="matching_results.tsv")
        zipf.write(cand_file, arcname="candidate_pairs.tsv")

    print(f"\n{'='*70}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Generated output package: {out_zip} ({os.path.getsize(out_zip)/1024/1024:.1f} MB)")
    print(f"Send 'submission_output.zip' back to the team!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
