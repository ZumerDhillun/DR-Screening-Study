# RetinaVision — RETFound vs. ConvNeXt-V2-Large under Clinical Shift

Study repo for: cross-center generalization, quality-shift robustness, and
uncertainty-based safety referral, comparing RETFound (ViT-Large) against
ConvNeXt-V2-Large.

## 1. Repo structure

```
retinavision/
├── README.md              <- you are here
├── CONTRIBUTING.md        <- git workflow for the two of you
├── requirements.txt
├── .gitignore
├── configs/
│   └── paths.yaml         <- all dataset paths, edit locally, never commit real paths with personal info
├── data/                  <- NEVER COMMITTED (see .gitignore). Populate locally.
│   ├── raw/
│   │   ├── ddr/
│   │   ├── deepdrid/
│   │   └── aptos2019/
│   └── processed/         <- output of preprocessing scripts
├── scripts/
│   ├── environment_check.py
│   ├── step1_hash_dedup.py
│   ├── step2_binarize_labels.py      (next step, stubbed)
│   ├── step3_class_weights.py        (next step, stubbed)
│   └── step4_preprocess.py           (next step, stubbed)
├── notebooks/              <- exploratory analysis only, not pipeline logic
└── outputs/
    ├── logs/
    └── reports/            <- CSV/JSON outputs from each step land here
```

## 2. Environment setup (each person runs this locally)

```bash
# Create and activate an isolated environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Confirm your setup matches your teammate's
python scripts/environment_check.py
```

Run `environment_check.py` and paste its output into your team chat before
you start Step 1 — mismatched library versions (especially `torch`,
`timm`, `Pillow`) are a common source of "it works on my machine" bugs
that silently change results.

## 3. Data placement

Data is **never committed to git** (see `.gitignore` — data files are
large, and DDR/DeepDRiD/APTOS all have their own redistribution terms).
Each teammate downloads datasets independently and places them under
`data/raw/<dataset_name>/`, matching the structure below. Update
`configs/paths.yaml` locally if your paths differ — this file is
git-ignored too, so your local paths never overwrite your teammate's.

| Dataset | Where to get it | Place at |
|---|---|---|
| DDR | https://github.com/nkicsl/DDR-dataset | `data/raw/ddr/` |
| DeepDRiD | https://github.com/deepdrdoc/DeepDRiD | `data/raw/deepdrid/` |
| APTOS 2019 | https://www.kaggle.com/competitions/aptos2019-blindness-detection/data | `data/raw/aptos2019/` |

## 4. Running Step 1 — hash de-duplication

```bash
python scripts/step1_hash_dedup.py \
    --ddr data/raw/ddr \
    --deepdrid data/raw/deepdrid \
    --aptos data/raw/aptos2019 \
    --out outputs/reports/step1_dedup_report.csv
```

This produces `outputs/reports/step1_dedup_report.csv`, listing:
- exact duplicates found (SHA-256 match)
- near-duplicates found (perceptual hash match within a small distance —
  catches recompressed/resized copies of the same image)
- which dataset pair each duplicate was found between

**Do not proceed to Step 2 until this report is empty or every flagged
pair has been manually reviewed and resolved (drop or justify keeping).**
Push the report to the repo (it's a small CSV, safe to commit) so you
both have a record of what was checked and when.

## 5. Status of pipeline steps

- [x] Step 1 — Hash de-duplication (script included, ready to run)
- [ ] Step 2 — Label harmonization & binarization
- [ ] Step 3 — Class imbalance handling
- [ ] Step 4 — Preprocessing pipeline
- [ ] Step 5 — Training & calibration (DDR only)
- [ ] Step 6 — Uncertainty & selective referral pipeline

Update the checklist above in the same commit whenever you finish a step
— see `CONTRIBUTING.md` for the exact workflow.
