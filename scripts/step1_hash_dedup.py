"""
Step 1 — Data Audit & Hash De-duplication

Checks for:
  1. Exact duplicate images (SHA-256 match) — byte-identical files.
  2. Near-duplicate images (perceptual hash match) — the same underlying
     photo saved twice with different compression/resizing, which SHA-256
     will NOT catch but which still represents data leakage if it happens
     across your train/test split.

Runs both WITHIN each dataset and ACROSS every pair of datasets
(DDR vs DeepDRiD, DDR vs APTOS, DeepDRiD vs APTOS). Cross-dataset matches
are the ones that matter most — if DDR (your training set) contains a
near-duplicate of an image in DeepDRiD or APTOS (your held-out test
sets), that test image is no longer a valid measure of generalization.

Usage:
    python scripts/step1_hash_dedup.py \
        --ddr data/raw/ddr \
        --deepdrid data/raw/deepdrid \
        --aptos data/raw/aptos2019 \
        --out outputs/reports/step1_dedup_report.csv \
        --phash-threshold 5

Output:
    A CSV report at --out listing every flagged pair, with columns:
    match_type, dataset_a, file_a, dataset_b, file_b, hamming_distance

    Also prints a summary to stdout. An empty report (0 rows) means no
    duplicates were found — you can proceed to Step 2. Any non-empty
    report must be manually reviewed before proceeding; do not silently
    auto-delete flagged images without a human looking at the pairs.
"""
import argparse
import hashlib
from pathlib import Path
from itertools import combinations

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    raise SystemExit(
        "Missing dependency: imagehash. Install with `pip install imagehash` "
        "(or `pip install -r requirements.txt`)."
    )

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(folder: Path):
    """Recursively find image files under a folder."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def compute_hashes(image_paths, dataset_name):
    """
    Returns a list of dicts: {dataset, path, sha256, phash}
    Skips unreadable files with a printed warning rather than crashing —
    corrupt files are common in large medical image dumps and you want
    the audit to finish and tell you about them, not die halfway through.
    """
    records = []
    for path in tqdm(image_paths, desc=f"Hashing {dataset_name}"):
        try:
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as img:
                img = img.convert("RGB")
                phash = imagehash.phash(img)
            records.append({
                "dataset": dataset_name,
                "path": str(path),
                "sha256": sha256,
                "phash": phash,
            })
        except Exception as e:
            print(f"  [WARN] Could not process {path}: {e}")
    return records


def find_exact_duplicates(records):
    """Group by sha256 hash; any group with >1 member is a set of exact duplicates."""
    by_hash = {}
    for r in records:
        by_hash.setdefault(r["sha256"], []).append(r)

    rows = []
    for sha, group in by_hash.items():
        if len(group) > 1:
            for a, b in combinations(group, 2):
                rows.append({
                    "match_type": "exact",
                    "dataset_a": a["dataset"], "file_a": a["path"],
                    "dataset_b": b["dataset"], "file_b": b["path"],
                    "hamming_distance": 0,
                })
    return rows


def find_near_duplicates(records, threshold):
    """
    Brute-force pairwise phash comparison. Fine for a few thousand images;
    for very large datasets (tens of thousands+), consider bucketing by
    phash prefix first to cut down comparisons — flagged here as a TODO
    rather than prematurely optimized.
    """
    rows = []
    n = len(records)
    for i in tqdm(range(n), desc="Near-duplicate scan"):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if a["sha256"] == b["sha256"]:
                continue  # already caught as an exact duplicate
            dist = a["phash"] - b["phash"]  # Hamming distance
            if dist <= threshold:
                rows.append({
                    "match_type": "near",
                    "dataset_a": a["dataset"], "file_a": a["path"],
                    "dataset_b": b["dataset"], "file_b": b["path"],
                    "hamming_distance": dist,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Step 1: hash-based de-duplication audit")
    parser.add_argument("--ddr", required=True, help="Path to DDR image folder")
    parser.add_argument("--deepdrid", required=True, help="Path to DeepDRiD image folder")
    parser.add_argument("--aptos", required=True, help="Path to APTOS 2019 image folder")
    parser.add_argument("--out", required=True, help="Output CSV path for the report")
    parser.add_argument(
        "--phash-threshold", type=int, default=5,
        help="Max Hamming distance to flag as a near-duplicate (default 5; "
             "lower = stricter/fewer matches, higher = more permissive/more matches)",
    )
    args = parser.parse_args()

    datasets = {
        "DDR": Path(args.ddr),
        "DeepDRiD": Path(args.deepdrid),
        "APTOS2019": Path(args.aptos),
    }

    all_records = []
    for name, folder in datasets.items():
        images = find_images(folder)
        print(f"{name}: found {len(images)} image files under {folder}")
        if not images:
            print(f"  [WARN] No images found for {name} — check the path in configs/paths.yaml")
            continue
        all_records.extend(compute_hashes(images, name))

    if not all_records:
        raise SystemExit("No images were successfully hashed across any dataset. Check your paths.")

    print(f"\nTotal images hashed: {len(all_records)}")

    exact_rows = find_exact_duplicates(all_records)
    near_rows = find_near_duplicates(all_records, args.phash_threshold)

    report = pd.DataFrame(exact_rows + near_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("STEP 1 SUMMARY")
    print("=" * 60)
    print(f"Exact duplicates found : {len(exact_rows)}")
    print(f"Near-duplicates found  : {len(near_rows)} (threshold={args.phash_threshold})")

    if not report.empty:
        cross_dataset = report[report["dataset_a"] != report["dataset_b"]]
        print(f"  of which CROSS-DATASET : {len(cross_dataset)}  <-- review these first")
        print(f"  of which WITHIN-DATASET: {len(report) - len(cross_dataset)}")

    print(f"\nFull report written to: {out_path}")
    if report.empty:
        print("No duplicates found. You may proceed to Step 2.")
    else:
        print("Duplicates found. Manually review every row before proceeding to Step 2 —")
        print("do not auto-delete without a human check, especially near-duplicates,")
        print("which can occasionally be false positives (e.g. two genuinely similar")
        print("but distinct fundus images from the same patient's follow-up visit).")
    print("=" * 60)


if __name__ == "__main__":
    main()
