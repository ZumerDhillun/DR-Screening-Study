"""
Step 2 — Label Harmonization & Binarization (v3 — index-based paths)

WHY THIS VERSION IS DIFFERENT:
Every previous version guessed a folder name + extension for each
dataset (e.g. "images/", ".png") and constructed paths from that guess.
Every guess turned out wrong in some way, because the real folder
structure only became visible after you actually unzipped things.

This version stops guessing. For each dataset, it walks the ENTIRE
data/raw/<dataset>/ tree ONCE, and builds an in-memory index mapping
"filename without extension" -> "real path on disk". Every row's image
is then found by looking up its id/image_id in that index, no matter
which subfolder it actually landed in or what extension it has.

IMPORTANT PERFORMANCE NOTE: build the index ONCE per dataset, not once
per row. Re-walking the whole folder tree for every single row (which a
naive "self-correcting" search does) is fine for a handful of files but
becomes very slow on DDR's 12,500+ rows — each miss would trigger a full
directory walk. Building the index once up front and doing O(1)
dictionary lookups per row avoids that entirely.

Run once, from the project root, with your virtual environment active:
    python scripts/step2_binarize_labels.py
"""
import os
from pathlib import Path

import pandas as pd

os.makedirs("data/processed", exist_ok=True)

REFERABLE = {2, 3, 4}
NON_REFERABLE = {0, 1}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def binarize_grade(grade):
    """Grade 0/1 -> 0 (non-referable), 2/3/4 -> 1 (referable), else None (dropped)."""
    try:
        g = int(float(grade))
    except (ValueError, TypeError):
        return None
    if g in REFERABLE:
        return 1
    elif g in NON_REFERABLE:
        return 0
    return None


def build_file_index(base_dir):
    """
    Walks base_dir ONCE and returns {filename_stem_lowercase: full_path}.
    This is what makes path resolution robust to whatever subfolder
    structure the dataset actually unzipped into, without needing you to
    tell the script the folder name in advance.
    """
    index = {}
    duplicates = 0
    base = Path(base_dir)
    if not base.exists():
        print(f"  [WARN] Base directory does not exist: {base_dir}")
        return index
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            stem = p.stem.lower()
            if stem in index:
                duplicates += 1
            else:
                index[stem] = str(p)
    if duplicates:
        print(f"  [WARN] {duplicates} duplicate filename stems found under {base_dir} — "
              f"only the first occurrence of each was kept. Worth a manual check that this "
              f"isn't hiding two genuinely different images sharing a name.")
    print(f"  Indexed {len(index)} image files under {base_dir}")
    return index


def lookup_image(index, id_code):
    """O(1) index lookup. Strips any extension already in id_code before matching."""
    stem = Path(str(id_code).strip()).stem.lower()
    return index.get(stem)  # None if genuinely not found anywhere in the tree


def find_col(df, target):
    """Case-insensitive, punctuation-insensitive column name lookup."""
    def normalize(s):
        return s.lower().replace(" ", "_").replace("-", "_")
    target_norm = normalize(target)
    for col in df.columns:
        if normalize(col) == target_norm:
            return col
    return None


def report(name, n_before, df_after):
    dropped = n_before - len(df_after)
    print(f"  {name}: {n_before} rows -> {len(df_after)} kept, {dropped} dropped (ungradable/invalid)")


def report_missing(df, label):
    missing = df["image_path"].isna().sum()
    if missing:
        print(f"  [WARN] {label}: {missing} rows had no matching image file anywhere under the "
              f"dataset's folder tree. These genuinely don't exist on disk — this is no longer a "
              f"path-guessing problem, worth checking whether the download is incomplete.")
    return missing


# ==========================================
# 1. DDR
# ==========================================
print("Processing DDR...")
ddr_csv = "data/raw/ddr/dr_grading.csv"
DDR_BASE = "data/raw/ddr"

if os.path.exists(ddr_csv):
    ddr_index = build_file_index(DDR_BASE)

    ddr_df = pd.read_csv(ddr_csv)
    print(f"  Columns found: {list(ddr_df.columns)}")

    ddr_df["dr_grade"] = ddr_df["diagnosis"]
    ddr_df["referable"] = ddr_df["dr_grade"].apply(binarize_grade)
    ddr_df["image_path"] = ddr_df["id_code"].apply(lambda x: lookup_image(ddr_index, x))

    n_before = len(ddr_df)
    ddr_clean = ddr_df.dropna(subset=["referable"]).copy()
    ddr_clean["referable"] = ddr_clean["referable"].astype(int)
    ddr_clean["dataset"] = "DDR"
    report("DDR", n_before, ddr_clean)
    report_missing(ddr_clean, "DDR")

    ddr_clean = ddr_clean[["id_code", "image_path", "dr_grade", "referable", "dataset"]]
    ddr_clean.to_csv("data/processed/ddr_clean.csv", index=False)
    print(f"  Saved data/processed/ddr_clean.csv ({len(ddr_clean)} rows)\n")
else:
    print(f"  [WARN] {ddr_csv} not found.\n")


# ==========================================
# 2. DeepDRiD
# ==========================================
print("Processing DeepDRiD...")
DEEPDRID_BASE = "data/raw/deepdrid"

REQUIRED_DEEPDRID_COLS = ["image_id", "left_eye_dr_level", "right_eye_dr_level"]
OPTIONAL_DEEPDRID_COLS = ["patient_id", "overall_quality", "clarity", "field_definition", "artifact"]

deepdrid_index = build_file_index(DEEPDRID_BASE)
deepdrid_frames = []


def get_eye_specific_grade(row, col_map):
    """Uses image_id text to pick the correct eye's grade. If no laterality
    cue is found, falls back to patient_dr_level (worse-eye) rather than
    dropping the row outright — noted as a fallback, not treated silently."""
    ref = str(row[col_map["image_id"]]).lower()
    if "_l" in ref or "left" in ref:
        return row[col_map["left_eye_dr_level"]]
    elif "_r" in ref or "right" in ref:
        return row[col_map["right_eye_dr_level"]]
    elif "patient_dr_level" in row.index and pd.notna(row.get("patient_dr_level")):
        return row["patient_dr_level"]
    return None


for split, csv_path in [
    ("train", f"{DEEPDRID_BASE}/regular-fundus-training/regular-fundus-training.csv"),
    ("validation", f"{DEEPDRID_BASE}/regular-fundus-validation/regular-fundus-validation.csv"),
]:
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found.")
        continue

    df = pd.read_csv(csv_path)
    print(f"  Columns found in {csv_path}: {list(df.columns)}")

    col_map = {name: find_col(df, name) for name in REQUIRED_DEEPDRID_COLS}
    missing_required = [name for name, actual in col_map.items() if actual is None]
    if missing_required:
        print(f"  [ERROR] {split}: could not find required column(s) {missing_required}. "
              f"Actual columns: {list(df.columns)}. Skipping this file.")
        continue

    opt_map = {name: find_col(df, name) for name in OPTIONAL_DEEPDRID_COLS}
    found_optional = {k: v for k, v in opt_map.items() if v is not None}

    # normalize to plain column names used by get_eye_specific_grade
    df["patient_dr_level"] = df[opt_map["patient_id"]] if False else df.get(find_col(df, "patient_dr_level"))

    n_before = len(df)
    df["dr_grade"] = df.apply(lambda row: get_eye_specific_grade(row, col_map), axis=1)

    has_some_grade = df[[col_map["left_eye_dr_level"], col_map["right_eye_dr_level"]]].notna().any(axis=1)
    undetermined = df["dr_grade"].isna() & has_some_grade
    if undetermined.sum():
        print(f"  [WARN] {split}: {undetermined.sum()} rows had eye-level grades available "
              f"but laterality couldn't be parsed and patient_dr_level fallback also failed.")

    df["referable"] = df["dr_grade"].apply(binarize_grade)
    df["split"] = split
    df["image_id"] = df[col_map["image_id"]]
    df["image_path"] = df["image_id"].apply(lambda x: lookup_image(deepdrid_index, x))

    for name in ["patient_id", "overall_quality", "clarity", "field_definition", "artifact"]:
        if name in found_optional:
            df[name] = df[found_optional[name]]

    keep_cols = ["patient_id", "image_id", "image_path", "overall_quality",
                 "clarity", "field_definition", "artifact", "dr_grade", "referable", "split"]
    df = df[[c for c in keep_cols if c in df.columns]]
    deepdrid_frames.append(df)
    report(f"DeepDRiD/{split}", n_before, df.dropna(subset=["referable"]))

# Online-Challenge evaluation folder
challenge1_path = f"{DEEPDRID_BASE}/Online-Challenge1&2-Evaluation/challenge1_labels.xlsx"
challenge2_path = f"{DEEPDRID_BASE}/Online-Challenge1&2-Evaluation/challenge2_labels.xlsx"

if os.path.exists(challenge1_path) and os.path.exists(challenge2_path):
    c1 = pd.read_excel(challenge1_path)
    c2 = pd.read_excel(challenge2_path)
    print(f"  Columns in challenge1_labels.xlsx: {list(c1.columns)}")
    print(f"  Columns in challenge2_labels.xlsx: {list(c2.columns)}")

    col_img1 = find_col(c1, "image_id")
    col_img2 = find_col(c2, "image_id")
    col_dr = find_col(c1, "dr_levels")

    if not all([col_img1, col_img2, col_dr]):
        print(f"  [ERROR] Could not find image_id/dr_levels columns in the evaluation files. Skipping.")
    else:
        overlap = set(c1[col_img1]) & set(c2[col_img2])
        print(f"  challenge1/challenge2 image_id overlap: {len(overlap)} of {len(c1)} (c1) / {len(c2)} (c2)")

        eval_df = c1.merge(c2, left_on=col_img1, right_on=col_img2, how="inner", suffixes=("", "_c2"))
        eval_df["dr_grade"] = eval_df[col_dr]
        eval_df["referable"] = eval_df["dr_grade"].apply(binarize_grade)
        eval_df["split"] = "evaluation"
        eval_df["image_id"] = eval_df[col_img1]
        eval_df["image_path"] = eval_df["image_id"].apply(lambda x: lookup_image(deepdrid_index, x))

        for name in ["overall_quality", "clarity", "field_definition", "artifact"]:
            actual = find_col(eval_df, name)
            if actual:
                eval_df[name] = eval_df[actual]

        keep_cols = ["image_id", "image_path", "overall_quality", "clarity",
                     "field_definition", "artifact", "dr_grade", "referable", "split"]
        eval_df = eval_df[[c for c in keep_cols if c in eval_df.columns]]
        deepdrid_frames.append(eval_df)
        report("DeepDRiD/evaluation", len(c1), eval_df.dropna(subset=["referable"]))
elif os.path.exists(challenge1_path) or os.path.exists(challenge2_path):
    print("  [WARN] Only one of challenge1_labels.xlsx / challenge2_labels.xlsx found — skipping.")
else:
    print("  [INFO] Online-Challenge evaluation files not found — skipping.")

if deepdrid_frames:
    deepdrid_all = pd.concat(deepdrid_frames, ignore_index=True)
    deepdrid_clean = deepdrid_all.dropna(subset=["referable"]).copy()
    deepdrid_clean["referable"] = deepdrid_clean["referable"].astype(int)
    deepdrid_clean["dataset"] = "DeepDRiD"

    report_missing(deepdrid_clean, "DeepDRiD")

    deepdrid_clean.to_csv("data/processed/deepdrid_clean.csv", index=False)
    print(f"  Saved data/processed/deepdrid_clean.csv ({len(deepdrid_clean)} rows)\n")
else:
    print("  [WARN] No DeepDRiD data processed.\n")


# ==========================================
# 3. APTOS 2019
# ==========================================
print("Processing APTOS 2019...")
APTOS_BASE = "data/raw/aptos2019"
aptos_index = build_file_index(APTOS_BASE)

aptos_files = {
    "train": f"{APTOS_BASE}/train_1.csv",
    "valid": f"{APTOS_BASE}/valid.csv",
    "test": f"{APTOS_BASE}/test.csv",
}

aptos_frames = []
for split, path in aptos_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Columns found in {path}: {list(df.columns)}")
        df["dr_grade"] = df["diagnosis"]
        df["referable"] = df["dr_grade"].apply(binarize_grade)
        df["image_path"] = df["id_code"].apply(lambda x: lookup_image(aptos_index, x))
        df["split"] = split
        aptos_frames.append(df)
    else:
        print(f"  [WARN] {path} not found.")

if aptos_frames:
    aptos_all = pd.concat(aptos_frames, ignore_index=True)
    n_before = len(aptos_all)
    aptos_clean = aptos_all.dropna(subset=["referable"]).copy()
    aptos_clean["referable"] = aptos_clean["referable"].astype(int)
    aptos_clean["dataset"] = "APTOS2019"
    report("APTOS2019", n_before, aptos_clean)
    report_missing(aptos_clean, "APTOS2019")

    aptos_clean = aptos_clean[["id_code", "image_path", "dr_grade", "referable", "split", "dataset"]]
    aptos_clean.to_csv("data/processed/aptos_clean.csv", index=False)
    print(f"  Saved data/processed/aptos_clean.csv ({len(aptos_clean)} rows)\n")
else:
    print("  [WARN] No APTOS data processed.\n")


# ==========================================
# Final summary
# ==========================================
print("=" * 60)
print("STEP 2 COMPLETE — check class balance below before Step 3")
print("=" * 60)
for f in ["data/processed/ddr_clean.csv", "data/processed/deepdrid_clean.csv", "data/processed/aptos_clean.csv"]:
    if os.path.exists(f):
        d = pd.read_csv(f)
        print(f"\n{f}: {len(d)} rows")
        print(d["referable"].value_counts())