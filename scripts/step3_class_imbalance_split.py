"""
Step 3 — Class Imbalance & Stratified Train/Val Split (corrected)

Fixes applied vs. the original draft:
  1. pos_weight is now computed from the TRAINING split only, after the
     split, not from the full train+val pool beforehand. Computing it on
     data that includes your validation set is a (small but real) form
     of information leakage into a training decision.
  2. Stratification uses the native 0-4 dr_grade, not just the binary
     referable label. Binary-only stratification protects the overall
     referable/non-referable ratio but says nothing about whether your
     rarest classes (severe, proliferative) end up reasonably
     represented in both splits — with small counts to begin with, pure
     chance could dump most of your proliferative cases into one split.
     Stratifying on grade automatically preserves the binary ratio too.
  3. Post-split class balance is now printed for both train and val, at
     both grade level and binary level, so you can actually see the
     result rather than trusting stratify= blindly.
  4. class_weights.txt replaced with class_weights.json — the training
     script in Step 5 will need to parse this reliably, and free-form
     text is more brittle to read back than JSON.

KNOWN, DOCUMENTED LIMITATION (not a bug, cannot be fixed with the data
available): DDR's public release does not include patient identifiers,
so this split is at the IMAGE level, not the patient level. DDR has
13,673 images from 9,598 patients, so some patients do have more than
one image, and a small amount of validation leakage from repeat
patients is possible. Other published work using DDR has hit this same
wall and stated it as a limitation rather than solved it — state this
explicitly in your methods section too.

Run once, from the project root, with your virtual environment active:
    python scripts/step3_class_imbalance_split.py
"""
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("RUNNING STEP 3: DATASET SPLITTING & LOSS WEIGHT CALCULATION")
print("=" * 60 + "\n")

# 1. Load Clean DDR Metadata
ddr_path = "data/processed/ddr_clean.csv"
if not os.path.exists(ddr_path):
    raise FileNotFoundError(f"Could not find {ddr_path}. Please run Step 2 first.")

df = pd.read_csv(ddr_path)
print(f"Loaded DDR cleaned metadata: {len(df)} initial rows")

# 2. Filter out rows whose image files don't exist OR aren't valid, openable
# images. NOTE: existence alone is not enough -- a truncated/interrupted
# download or a partial zip extraction can leave a file that EXISTS on disk
# but isn't real image data. That kind of file passes an os.path.exists()
# check, sails through Step 2 and this filter's old version, and then kills
# the entire training run the first time the DataLoader tries to open it
# (this is exactly the failure that showed up in Step 4). Checking real
# openability once here, up front, costs a few minutes on 12,000+ images --
# far cheaper than discovering it mid-training.
from PIL import Image, UnidentifiedImageError


def is_openable_image(path):
    path = str(path)
    if not os.path.exists(path):
        return False
    try:
        with Image.open(path) as img:
            img.convert("RGB")  # mirrors exactly what FundusDataset does in Step 4
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


print("Checking that every image file actually opens (this may take a few minutes "
      "on the full dataset)...")
df["file_valid"] = df["image_path"].apply(is_openable_image)
df_valid = df[df["file_valid"]].copy().drop(columns=["file_valid"])
dropped_count = len(df) - len(df_valid)
print(f"Filtered out {dropped_count} missing/corrupt images from DDR. Valid images on disk: {len(df_valid)}")
print("(This count may be larger than Step 2's '[WARN] ... no matching image file' count -- "
      "that warning only catches files that don't EXIST at all. This check also catches files "
      "that exist but are corrupted/unreadable, which Step 2 has no way to detect.)\n")

# 3. Stratified Train / Validation split — FIX #2: stratify on native grade
print("--- Native grade distribution (full valid set) ---")
print(df_valid["dr_grade"].value_counts().sort_index())
print()

train_df, val_df = train_test_split(
    df_valid, test_size=0.20, random_state=42, stratify=df_valid["dr_grade"]
)
train_df = train_df.copy()
val_df = val_df.copy()
train_df["split"] = "train"
val_df["split"] = "val"

# FIX #3 — verify the split actually preserved balance, don't just trust stratify=
print("--- Post-split grade distribution ---")
compare = pd.DataFrame({
    "train_n": train_df["dr_grade"].value_counts().sort_index(),
    "val_n": val_df["dr_grade"].value_counts().sort_index(),
})
compare["train_pct"] = (compare["train_n"] / compare["train_n"].sum() * 100).round(2)
compare["val_pct"] = (compare["val_n"] / compare["val_n"].sum() * 100).round(2)
print(compare)
print()

print("--- Post-split binary (referable) distribution ---")
print("Train:")
print(train_df["referable"].value_counts())
print("Val:")
print(val_df["referable"].value_counts())
print()

# 4. FIX #1 — compute pos_weight from the TRAINING split only, after splitting
train_counts = train_df["referable"].value_counts()
n_neg = train_counts.get(0, 0)
n_pos = train_counts.get(1, 0)
pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
print(f"Calculated PyTorch pos_weight for BCE Loss (from TRAIN split only): {pos_weight:.4f}\n")

# 5. Save final split metadata
combined_df = pd.concat([train_df, val_df], ignore_index=True)
out_split_path = "data/processed/ddr_split.csv"
combined_df.to_csv(out_split_path, index=False)

# FIX #4 — JSON instead of free-form text, easier for Step 5 to parse reliably
class_weights = {
    "pos_weight": round(float(pos_weight), 6),
    "train_samples": int(len(train_df)),
    "val_samples": int(len(val_df)),
    "train_class_counts": {str(k): int(v) for k, v in train_counts.items()},
    "computed_from": "train split only (post-split), see Step 3 script for why",
}
with open("data/processed/class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)

print("Step 3 Complete!")
print(f"  Training Set:   {len(train_df)} samples")
print(f"  Validation Set: {len(val_df)} samples")
print(f"  Saved master split metadata to: {out_split_path}")
print(f"  Saved training weights to:       data/processed/class_weights.json")
print()
print("REMINDER: DDR's public release has no patient IDs, so this split is at the image")
print("level, not patient level. State this explicitly as a limitation in your methods.")
