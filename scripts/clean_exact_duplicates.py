import os
import pandas as pd

# Load report
report_path = "outputs/reports/step1_dedup_report.csv"
df = pd.read_csv(report_path)

# Filter only EXACT matches within the same dataset
exact_matches = df[
    (df["match_type"] == "exact") & (df["dataset_a"] == df["dataset_b"])
]

# Collect unique duplicate files to remove (file_b)
files_to_remove = set(exact_matches["file_b"])

print(f"Found {len(files_to_remove)} exact duplicate files to delete...")

# Delete files directly from disk
deleted_count = 0
for file_path in files_to_remove:
    # Convert Windows backslashes if needed
    clean_path = os.path.normpath(file_path)

    if os.path.exists(clean_path):
        os.remove(clean_path)
        deleted_count += 1
        print(f"Deleted: {clean_path}")
    else:
        print(f"File not found (already deleted or wrong path): {clean_path}")

print(
    f"\n Done! Successfully deleted {deleted_count} duplicate images directly from your data folder."
)