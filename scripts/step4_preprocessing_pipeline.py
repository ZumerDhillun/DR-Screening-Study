"""
Step 4 — Preprocessing & Normalization Pipeline (corrected)

Fixes applied vs. the original draft:
  1. Square-padding before resize. Cropping the black border leaves a
     rectangular (not necessarily square) image. Resizing that directly
     to 224x224 stretches it non-uniformly, distorting the actual shape
     of the retina. This version pads the crop to a square canvas first,
     THEN resizes — preserving the original proportions.
  2. Model-specific normalization structure. RETFound and ConvNeXt-V2-
     Large are not guaranteed to use the same normalization stats.
     Plain ImageNet stats are used here as a DEFAULT, clearly marked as
     unverified — see the VERIFY_RETFOUND_NORMALIZATION note below for
     how to check the real values before you trust this for RETFound.
  3. Renamed "Graham Processing" to what it actually is: global mean-std
     normalization. Real Graham preprocessing — a different operation, not implemented here.
     If your methods section currently says "Graham processing," fix
     that wording or implement the real technique.
  4. FundusDataset's label column is now configurable (defaults to
     "referable", can be switched to "dr_grade" later for the 5-class
     deployment model) instead of hardcoded.
  5. Fixed random seed for reproducibility across your two machines.

Run once, from the project root, with your virtual environment active:
    python scripts/step4_preprocessing_pipeline.py
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SAVE_DIR = "data/processed/preprocessed_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

# FIX #2 — model-specific normalization, NOT one shared assumption.
# Both currently default to standard ImageNet stats — UNVERIFIED for
# RETFound specifically. See the module docstring above for how to check.
MODEL_NORM_STATS = {
    "convnext_v2_large": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "retfound": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # VERIFY ME
}


# ==========================================
# 1. Black Border Cropping Function
# ==========================================
def crop_black_border(img, tol=7):
    """Crops out dead black margins around the fundus retina circle."""
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    if img_np.ndim == 2:
        mask = img_np > tol
    elif img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
    else:
        return Image.fromarray(img_np)

    check_col = mask.any(axis=0)
    check_row = mask.any(axis=1)
    if not check_col.any() or not check_row.any():
        return Image.fromarray(img_np)

    ymin, ymax = np.where(check_row)[0][[0, -1]]
    xmin, xmax = np.where(check_col)[0][[0, -1]]
    cropped = img_np[ymin: ymax + 1, xmin: xmax + 1]
    return Image.fromarray(cropped)


def pad_to_square(img, fill=0):
    """
    FIX #1 — pads the (likely rectangular, post-crop) image to a square
    canvas by adding borders on the shorter side, centering the original
    content. This must happen BEFORE Resize, or Resize will stretch the
    image non-uniformly and distort the retina's true proportions.
    """
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    pad_w = side - w
    pad_h = side - h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return ImageOps.expand(img, padding, fill=fill)


# ==========================================
# 2. PyTorch Transforms Pipeline
# ==========================================
def get_transforms(img_size=224, is_training=True, model_name="convnext_v2_large"):
    """
    Builds the PyTorch transform pipeline, using the NORMALIZATION STATS
    SPECIFIC TO model_name (see MODEL_NORM_STATS above) rather than one
    shared assumption for every model in the study.
    """
    if model_name not in MODEL_NORM_STATS:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Add its normalization stats to "
            f"MODEL_NORM_STATS first — don't fall back to a guessed default."
        )
    stats = MODEL_NORM_STATS[model_name]

    transform_list = [
        transforms.Lambda(lambda img: crop_black_border(img, tol=7)),
        transforms.Lambda(lambda img: pad_to_square(img, fill=0)),
        transforms.Resize((img_size, img_size)),
    ]

    if is_training:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])

    return transforms.Compose(transform_list)


# ==========================================
# 3. Custom Fundus PyTorch Dataset
# ==========================================
class FundusDataset(Dataset):
    """
    FIX #4 — label_col is now configurable. Defaults to "referable" (the
    binary research target) but can be set to "dr_grade" to reuse this
    same class for the native 0-4 deployment model later, without
    duplicating this whole file.
    """

    def __init__(self, df, transform=None, label_col="referable"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")

        label = int(row[self.label_col])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_path


# ==========================================
# 4. Pipeline Verification & Visualization
# ==========================================
def visualize_preprocessed_samples(dataset, num_samples=4):
    """Un-normalizes and saves visual comparisons of raw vs preprocessed images."""
    stats = MODEL_NORM_STATS["convnext_v2_large"]  # for display only
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])

    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])

    for i in range(num_samples):
        row = dataset.df.iloc[i]
        raw_img = Image.open(row["image_path"]).convert("RGB")

        processed_tensor, label, _ = dataset[i]
        proc_np = processed_tensor.numpy().transpose((1, 2, 0))
        proc_np = np.clip(std * proc_np + mean, 0, 1)

        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"Raw Fundus\nPath: {os.path.basename(row['image_path'])}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(proc_np)
        axes[i, 1].set_title(f"Preprocessed (224x224)\nLabel: {int(label)}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "preprocessing_verification.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved visual inspection grid to: {save_path}")
    print("MANUALLY OPEN THIS FILE and confirm: (a) black borders are gone, "
          "(b) the retina looks circular, not stretched into an oval.")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STEP 4: PREPROCESSING & NORMALIZATION PIPELINE")
    print("=" * 60 + "\n")

    split_csv = "data/processed/ddr_split.csv"
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Could not find {split_csv}. Please run Step 3 first.")

    df_all = pd.read_csv(split_csv)
    train_df = df_all[df_all["split"] == "train"].copy()
    val_df = df_all[df_all["split"] == "val"].copy()

    print(f"Loaded train split: {len(train_df)} rows")
    print(f"Loaded val split:   {len(val_df)} rows\n")

    IMG_SIZE = 224
    MODEL_NAME = "convnext_v2_large"  # change to "retfound" for that arm — see VERIFY note above

    train_transform = get_transforms(img_size=IMG_SIZE, is_training=True, model_name=MODEL_NAME)
    val_transform = get_transforms(img_size=IMG_SIZE, is_training=False, model_name=MODEL_NAME)

    train_dataset = FundusDataset(train_df, transform=train_transform, label_col="referable")
    val_dataset = FundusDataset(val_df, transform=val_transform, label_col="referable")

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    images, labels, _ = next(iter(train_loader))
    print("DataLoader Test Batch Successfully Fetched!")
    print(f"   Batch Images Shape: {images.shape} (Batch, Channels, Height, Width)")
    print(f"   Batch Labels Shape: {labels.shape}")
    print(f"   Image Tensor Min/Max Values: [{images.min():.2f}, {images.max():.2f}] (Normalized)")

    print("\nGenerating image crop & resize verification grid...")
    visualize_preprocessed_samples(val_dataset, num_samples=4)

    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE — Preprocessing transforms & DataLoaders ready!")
    print("REMINDER: verify RETFound's real normalization stats (see docstring)")
    print("before running this same pipeline for the RETFound arm.")
    print("=" * 60 + "\n")