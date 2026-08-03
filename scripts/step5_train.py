"""
Step 5 — Training & Freezing Protocol (corrected)

Fixes applied vs. the original draft:
  1. Correct model identifiers. "convnext_base" was NOT ConvNeXt-V2-Large
     (it's a smaller, different model family). The correct timm tag is
     "convnextv2_large.fcmae_ft_in22k_in1k" (~198M params, matched scale
     to RETFound's ~304M). RETFound is loaded via a SEPARATE path (see
     load_retfound_backbone below) since it isn't a standard timm
     pretrained tag.
  2. Preprocessing is IMPORTED from step4_preprocessing_pipeline.py, not
     reimplemented — guarantees the square-padding fix (and any future
     fix) is actually used, instead of silently regressing.
  3. pos_weight is loaded from data/processed/class_weights.json (Step
     3's actual output), not hardcoded.
  4. Temperature scaling is now a genuine POST-HOC step: the backbone is
     fully trained and FROZEN first, then ONLY the temperature scalar is
     fit afterward, on validation logits alone, via a separate small
     optimization. This is what temperature scaling (Guo et al. 2017)
     actually means — fitting it jointly with the backbone during main
     training (as the original draft did) does not calibrate anything.
  5. Checkpoint-per-epoch + resume-from-checkpoint, given Colab sessions
     can disconnect mid-run. Re-running this script picks up where it
     left off instead of restarting from scratch.
  6. Fixed random seed for reproducibility across your two Colab
     accounts running in parallel.
  7. A quick existence check on resolved image paths before training
     starts — catches the Windows-CSV-on-Colab-Linux path problem
     immediately instead of crashing deep into an epoch.

USAGE:
    python scripts/step5_train.py --model convnext_v2_large --epochs 15
    python scripts/step5_train.py --model retfound --epochs 15 --batch_size 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step4_preprocessing_pipeline import FundusDataset, get_transforms, MODEL_NORM_STATS  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


MODEL_REGISTRY = {
    "convnext_v2_large": {
        "timm_name": "convnextv2_large.fcmae_ft_in22k_in1k",
        "norm_key": "convnext_v2_large",
    },
    # RETFound is not a standard timm pretrained tag -- see load_retfound_backbone().
    "retfound": {
        "timm_name": None,
        "norm_key": "retfound",
    },
}


def load_convnext_backbone(timm_name):
    import timm
    return timm.create_model(timm_name, pretrained=True, num_classes=1)


def load_retfound_backbone(checkpoint_path):
    """
    RETFound is a ViT-Large pretrained via MAE — not a standard timm
    pretrained tag, so it can't be loaded with timm.create_model(pretrained=True).

    UNVERIFIED BY ME: I do not have access to the gated RETFound checkpoint,
    so I cannot test this loading path end-to-end myself. Confirm against
    RETFound's official repo (https://github.com/rmaphoh/RETFound_MAE)
    before trusting this in a real run -- the exact state_dict key names
    and any required prefix-stripping can change between checkpoint
    releases, and a silent partial load (missing keys) is worse than a
    loud failure here.
    """
    import timm
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=1)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"RETFound checkpoint not found at {checkpoint_path}. Download it from the "
            f"official gated HuggingFace/GitHub release first (requires requesting access)."
        )
    # weights_only=False is safe here: this is the official RETFound checkpoint
    # downloaded directly from the gated release, not an untrusted source.
    # It contains a pickled argparse.Namespace (the original authors' training
    # args) which torch's default safe-unpickler blocks.
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  [RETFound load] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if len(missing) > 5 or len(unexpected) > 5:
        print("  [WARN] A large number of mismatched keys suggests this checkpoint format "
              "doesn't match this loading code exactly -- verify against the official repo "
              "before trusting any results from this model.")
    return model


def load_class_weights():
    path = "data/processed/class_weights.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run Step 3 first.")
    with open(path) as f:
        weights = json.load(f)
    return weights["pos_weight"]


def check_paths_exist(df, label, sample_size=50):
    """
    Catches the cross-machine path problem (CSV built on Windows, running
    on Colab's Linux filesystem) immediately, rather than failing deep
    into the first epoch.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=SEED)
    missing = sample[~sample["image_path"].apply(lambda p: os.path.exists(str(p)))]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"{label}: {len(missing)}/{len(sample)} sampled image paths don't exist on this "
            f"filesystem. Example: {missing.iloc[0]['image_path']}. This usually means the "
            f"CSV was generated on a different machine (e.g. Windows paths on Colab's Linux "
            f"filesystem). Re-run scripts/step2_binarize_labels.py and "
            f"step3_class_imbalance_split.py IN THIS ENVIRONMENT after extracting the raw "
            f"data here, rather than trusting an imported CSV's baked-in paths."
        )
    print(f"  {label}: path check passed ({len(sample)} sampled paths all exist)")


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_auc):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_auc": best_val_auc,
    }, path)


def load_checkpoint_if_exists(path, model, optimizer, scheduler):
    if not os.path.exists(path):
        return 0, 0.0
    # weights_only=False is safe here: this checkpoint is written by our own
    # save_checkpoint() above, not downloaded from an untrusted source.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"  Resumed from checkpoint: epoch {ckpt['epoch']}, best_val_auc so far {ckpt['best_val_auc']:.4f}")
    return ckpt["epoch"], ckpt["best_val_auc"]


def fit_temperature(model, val_loader, device):
    """
    FIX #4 — genuine post-hoc temperature scaling. The backbone is
    already fully trained and is frozen here; ONLY this scalar is fit,
    on validation logits, via a short separate optimization (LBFGS is
    standard for this since it's a 1-parameter, well-behaved problem).
    """
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            targets_list.append(labels)
    logits = torch.cat(logits_list).squeeze()
    targets = torch.cat(targets_list).squeeze()

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = bce(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_temp = float(temperature.detach().item())
    print(f"  Fitted temperature (post-hoc, validation-only): {fitted_temp:.4f}")
    return fitted_temp


def train_model(model_key, epochs, batch_size, lr, out_dir, retfound_checkpoint=None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_key} on {device} for up to {epochs} epochs...")

    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(MODEL_REGISTRY)}.")
    norm_key = MODEL_REGISTRY[model_key]["norm_key"]

    df = pd.read_csv("data/processed/ddr_split.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    check_paths_exist(train_df, "train split")
    check_paths_exist(val_df, "val split")

    train_transform = get_transforms(img_size=224, is_training=True, model_name=norm_key)
    val_transform = get_transforms(img_size=224, is_training=False, model_name=norm_key)

    train_loader = DataLoader(
        FundusDataset(train_df, train_transform, label_col="referable"),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        FundusDataset(val_df, val_transform, label_col="referable"),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    if model_key == "convnext_v2_large":
        model = load_convnext_backbone(MODEL_REGISTRY[model_key]["timm_name"])
    elif model_key == "retfound":
        if retfound_checkpoint is None:
            raise ValueError("--retfound_checkpoint path is required when --model retfound")
        model = load_retfound_backbone(retfound_checkpoint)
    model = model.to(device)

    pos_weight_value = load_class_weights()
    print(f"  Using pos_weight={pos_weight_value:.4f} (loaded from data/processed/class_weights.json)")
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_path = os.path.join(out_dir, f"{model_key}_checkpoint.pth")
    best_path = os.path.join(out_dir, f"{model_key}_best.pth")

    start_epoch, best_val_auc = load_checkpoint_if_exists(ckpt_path, model, optimizer, scheduler)

    # Quick timing benchmark on the first few batches before committing to a
    # full run -- do NOT trust a guessed "this will take N minutes" estimate.
    if start_epoch == 0:
        print("\n--- Benchmarking a few batches before full training ---")
        t0 = time.time()
        n_bench = 0
        for images, labels, _ in train_loader:
            n_bench += 1
            if n_bench >= 5:
                break
        elapsed = time.time() - t0
        per_batch = elapsed / n_bench
        batches_per_epoch = len(train_loader)
        est_epoch_time = per_batch * batches_per_epoch
        print(f"  ~{per_batch:.2f}s/batch -> estimated ~{est_epoch_time/60:.1f} min/epoch, "
              f"~{est_epoch_time*epochs/60:.1f} min for all {epochs} epochs. "
              f"Treat this as a rough guide, not a promise.\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_df)

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                logits = model(images)
                val_logits.extend(logits.cpu().numpy().flatten())
                val_targets.extend(labels.numpy())

        val_probs = 1 / (1 + np.exp(-np.array(val_logits)))
        val_auc = roc_auc_score(val_targets, val_probs)
        print(f"  Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val AUC {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved (Val AUC {best_val_auc:.4f})")

        # Save a resumable checkpoint EVERY epoch, not just the best one --
        # this is what survives a Colab disconnect mid-run.
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch + 1, best_val_auc)

    # ==========================================
    # Post-hoc calibration and threshold selection -- backbone frozen from here on
    # ==========================================
    print("\n--- Post-hoc temperature scaling & decision threshold selection ---")
    model.load_state_dict(torch.load(best_path, weights_only=False))
    for p in model.parameters():
        p.requires_grad = False

    fitted_temp = fit_temperature(model, val_loader, device)

    model.eval()
    val_logits, val_targets = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images) / fitted_temp
            val_logits.extend(logits.cpu().numpy().flatten())
            val_targets.extend(labels.numpy())

    val_targets = np.array(val_targets)
    val_probs = 1 / (1 + np.exp(-np.array(val_logits)))

    precisions, recalls, thresholds = precision_recall_curve(val_targets, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    config = {
        "model_key": model_key,
        "best_val_auc": float(best_val_auc),
        "T_decision": best_threshold,
        "T_temp": fitted_temp,
        "pos_weight": pos_weight_value,
        "frozen": True,
    }
    config_path = os.path.join(out_dir, f"{model_key}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print("\nTraining complete.")
    print(f"  Best Val AUC:       {best_val_auc:.4f}")
    print(f"  Fitted temperature: {fitted_temp:.4f}")
    print(f"  Selected threshold: {best_threshold:.4f}")
    print(f"  Saved weights:      {best_path}")
    print(f"  Saved config:       {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="/content/drive/My Drive/retinavision_models")
    parser.add_argument("--retfound_checkpoint", default=None,
                         help="Path to the RETFound .pth checkpoint (required if --model retfound)")
    args = parser.parse_args()

    train_model(
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        retfound_checkpoint=args.retfound_checkpoint,
    )
