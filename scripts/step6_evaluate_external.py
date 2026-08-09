"""
Step 6 — Frozen Evaluation on DeepDRiD & APTOS 2019
USAGE:
    python scripts/step6_evaluate_external.py --model convnext_v2_large \
        --weights "/content/drive/My Drive/retinavision_models/convnext_v2_large_best.pth" \
        --config  "/content/drive/My Drive/retinavision_models/convnext_v2_large_config.json"
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from step4_preprocessing_pipeline import FundusDataset, get_transforms  # noqa: E402


def load_model(model_key, weights_path, device, retfound_raw_checkpoint=False):
    import timm

    if model_key == "convnext_v2_large":
        model = timm.create_model(
            "convnextv2_large.fcmae_ft_in22k_in1k", pretrained=False, num_classes=1,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    elif model_key == "retfound":
        model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=1)
        if retfound_raw_checkpoint:
            import argparse as _argparse
            torch.serialization.add_safe_globals([_argparse.Namespace])
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            state_dict = (
                checkpoint["model"]
                if isinstance(checkpoint, dict) and "model" in checkpoint
                else checkpoint
            )
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = in_bin | (y_prob == 0)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def sanitize_dataframe_paths(df, label):
    """Filters out non-existent paths in memory rather than crashing with an error."""
    initial_count = len(df)
    valid_mask = df["image_path"].apply(lambda p: os.path.exists(str(p)))
    clean_df = df[valid_mask].copy()
    dropped_count = initial_count - len(clean_df)

    if dropped_count > 0:
        print(
            f"  [WARN] {label}: {dropped_count}/{initial_count} image paths don't exist on disk. "
            f"Filtered out in-memory ({len(clean_df)} valid images retained)."
        )
    else:
        print(f"  {label}: path check passed ({initial_count}/{initial_count} paths all exist)")

    return clean_df


def run_inference(model, dataloader, device):
    raw_logits, targets, paths = [], [], []
    with torch.no_grad():
        for images, labels, img_paths in tqdm(dataloader, desc="Running Inference"):
            images = images.to(device)
            logits = model(images).squeeze(-1)
            raw_logits.extend(logits.cpu().numpy().flatten())
            targets.extend(labels.numpy())
            paths.extend(img_paths)
    return np.array(raw_logits), np.array(targets), np.array(paths)


def compute_dataset_metrics(
    raw_logits, targets, t_temp, t_decision, save_path=None, paths=None, extra_cols=None
):
    if len(targets) == 0:
        return None
    if len(np.unique(targets)) < 2:
        print(
            "  [WARN] Only one class present in this subset -- AUC/PR-AUC are undefined here, "
            "reporting what's computable and marking the rest None."
        )

    calibrated_logits = raw_logits / t_temp
    uncal_probs = 1 / (1 + np.exp(-raw_logits))
    cal_probs = 1 / (1 + np.exp(-calibrated_logits))

    if save_path is not None:
        pred_df = pd.DataFrame({
            "image_path": paths if paths is not None else np.arange(len(targets)),
            "target": targets,
            "raw_logit": raw_logits,
            "cal_prob": cal_probs,
        })
        if extra_cols:
            for col_name, col_values in extra_cols.items():
                pred_df[col_name] = col_values
        pred_df.to_csv(save_path, index=False)
        print(f"  Saved per-sample predictions to: {save_path}")

    auc = roc_auc_score(targets, cal_probs) if len(np.unique(targets)) > 1 else None
    pr_auc = average_precision_score(targets, cal_probs) if len(np.unique(targets)) > 1 else None

    preds = (cal_probs >= t_decision).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(targets, preds, zero_division=0))
    f1 = float(f1_score(targets, preds, zero_division=0))
    acc = float(accuracy_score(targets, preds))

    ece_uncal = calculate_ece(targets, uncal_probs)
    ece_cal = calculate_ece(targets, cal_probs)
    brier_score = float(np.mean((cal_probs - targets) ** 2))

    return {
        "n_samples": int(len(targets)),
        "auc_roc": float(auc) if auc is not None else None,
        "pr_auc": float(pr_auc) if pr_auc is not None else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "accuracy": acc,
        "ece_uncalibrated": ece_uncal,
        "ece_calibrated": ece_cal,
        "brier_score": brier_score,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_external(
    model_key, weights_path, config_path, out_dir, retfound_raw_checkpoint=False
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = json.load(f)
    t_temp = config.get("T_temp", 1.0)
    t_decision = config.get("T_decision", 0.5)

    print(f"Loading {model_key} from {weights_path}...")
    model = load_model(model_key, weights_path, device, retfound_raw_checkpoint)

    transform = get_transforms(img_size=224, is_training=False, model_name=model_key)
    results = {"model_key": model_key, "frozen_config": config, "datasets": {}}

    # 1. APTOS 2019
    aptos_csv = "data/processed/aptos_clean.csv"
    if os.path.exists(aptos_csv):
        print("\n--- Evaluating APTOS 2019 ---")
        df_aptos = pd.read_csv(aptos_csv)
        df_aptos_clean = sanitize_dataframe_paths(df_aptos, "APTOS2019")

        if len(df_aptos_clean) > 0:
            loader_aptos = DataLoader(
                FundusDataset(df_aptos_clean, transform, label_col="referable"),
                batch_size=16,
                shuffle=False,
                num_workers=2,
            )
            logits, targets, paths = run_inference(model, loader_aptos, device)
            pred_path = os.path.join(out_dir, f"{model_key}_aptos2019_predictions.csv")
            results["datasets"]["aptos2019"] = compute_dataset_metrics(
                logits, targets, t_temp, t_decision, save_path=pred_path, paths=paths
            )
    else:
        print(f"  [WARN] {aptos_csv} not found -- skipping APTOS evaluation.")

    # 2. DeepDRiD
    deepdrid_csv = "data/processed/deepdrid_clean.csv"
    if os.path.exists(deepdrid_csv):
        print("\n--- Evaluating DeepDRiD ---")
        df_deepdrid = pd.read_csv(deepdrid_csv)
        df_deepdrid_clean = sanitize_dataframe_paths(df_deepdrid, "DeepDRiD")

        if len(df_deepdrid_clean) > 0:
            loader_deepdrid = DataLoader(
                FundusDataset(df_deepdrid_clean, transform, label_col="referable"),
                batch_size=16,
                shuffle=False,
                num_workers=2,
            )
            logits, targets, paths = run_inference(model, loader_deepdrid, device)
            pred_path = os.path.join(out_dir, f"{model_key}_deepdrid_predictions.csv")

            quality_extra = (
                {"overall_quality": df_deepdrid_clean["overall_quality"].values}
                if "overall_quality" in df_deepdrid_clean.columns
                else None
            )

            results["datasets"]["deepdrid_overall"] = compute_dataset_metrics(
                logits,
                targets,
                t_temp,
                t_decision,
                save_path=pred_path,
                paths=paths,
                extra_cols=quality_extra,
            )

            if "overall_quality" in df_deepdrid_clean.columns:
                print("\n--- DeepDRiD Quality Stratification (overall_quality) ---")
                results["datasets"]["deepdrid_by_quality"] = {}
                for q_val in sorted(df_deepdrid_clean["overall_quality"].dropna().unique()):
                    q_mask = (df_deepdrid_clean["overall_quality"] == q_val).values
                    n_in_bucket = int(np.sum(q_mask))
                    print(f"  overall_quality={q_val}: n={n_in_bucket}")
                    if n_in_bucket > 0:
                        metrics = compute_dataset_metrics(
                            logits[q_mask], targets[q_mask], t_temp, t_decision
                        )
                        results["datasets"]["deepdrid_by_quality"][str(q_val)] = metrics
            else:
                print(
                    "  [WARN] 'overall_quality' column not found in deepdrid_clean.csv -- "
                    "quality-stratified evaluation skipped."
                )
    else:
        print(f"  [WARN] {deepdrid_csv} not found -- skipping DeepDRiD evaluation.")

    out_file = os.path.join(out_dir, f"{model_key}_eval.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nStep 6 evaluation complete. Saved report to: {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["convnext_v2_large", "retfound"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="outputs/reports")
    parser.add_argument(
        "--retfound_raw_checkpoint",
        action="store_true",
        help="Set this if --weights points to the raw RETFound release checkpoint.",
    )
    args = parser.parse_args()

    evaluate_external(
        args.model, args.weights, args.config, args.out_dir, args.retfound_raw_checkpoint
    )