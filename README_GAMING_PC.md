# Amazon ML Challenge 2026 — Gaming PC Run Guide

Follow these quick steps to run the pipeline on your machine and generate the submission files.

---

### Step 1: Requirements
Make sure you have **Python 3.10, 3.11, or 3.12** installed.
Open Terminal (on Linux / macOS / Windows WSL2) or PowerShell (on Windows) and install the dependencies:

```bash
pip install polars rapidfuzz anyascii lightgbm pyarrow scikit-learn tqdm
```

---

### Step 2: Run the Pipeline (1-Click)
From inside this folder, run:

```bash
python run_pipeline_end_to_end.py
```

#### What it does automatically:
1. **Downloads the dataset** (~1.1 GB) directly from the official Unstop CDN and unzips it into `dataset/` (if not already present).
2. **Normalizes all 6 sources** into optimized Parquet files.
3. **Generates candidate pairs** using inverted-index blocking.
4. **Trains the two-stage LightGBM matchers** and automatically calibrates the optimal $F_{0.5}$ decision threshold.
5. **Generates predictions** for the test set (`matching_results.tsv` and `candidate_pairs.tsv`).
6. **Validates the submission format** against all official competition checks.
7. **Packs the results** into `submission_output.zip` (~40–60 MB).

---

### Step 3: Send Back the Output
When finished, send the generated **`submission_output.zip`** file back!
