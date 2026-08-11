"""
Complete Visualization Suite for Publication 

Generates:
  1. ROC Curves (APTOS 2019 & DeepDRiD)
  2. Precision-Recall (PR) Curves
  3. Confusion Matrices (Raw Counts & Percentages)
  4. Multi-Metric Comparison Bar Charts (Accuracy, Precision, Recall/Sensitivity, Specificity, F1)
  5. Training vs Validation Curves (Loss & Accuracy over Epochs)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Set global academic plot formatting
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "ConvNeXt-V2-Large": "convnext_v2_large",
    "RETFound": "retfound",
}
COLORS = {"ConvNeXt-V2-Large": "#1f77b4", "RETFound": "#d62728"}


# ----------------------------------------------------------------------
# 1. Plot ROC Curves & Precision-Recall Curves
# ----------------------------------------------------------------------
def plot_curves():
    datasets = ["aptos2019", "deepdrid"]

    for ds in datasets:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

        for model_label, model_key in MODELS.items():
            csv_path = os.path.join(REPORTS_DIR, f"{model_key}_{ds}_predictions.csv")
            if not os.path.exists(csv_path):
                print(f"Skipping {csv_path} (file not found)")
                continue

            df = pd.read_csv(csv_path)
            y_true = df["target"].values
            y_prob = df["cal_prob"].values

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = pd.read_csv(csv_path)  # recalculated dynamically
            from sklearn.metrics import average_precision_score, roc_auc_score

            auc_score = roc_auc_score(y_true, y_prob)
            pr_auc_score = average_precision_score(y_true, y_prob)

            ax_roc.plot(
                fpr,
                tpr,
                label=f"{model_label} (AUC = {auc_score:.4f})",
                color=COLORS[model_label],
                linewidth=2.5,
            )

            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax_pr.plot(
                recall,
                precision,
                label=f"{model_label} (PR-AUC = {pr_auc_score:.4f})",
                color=COLORS[model_label],
                linewidth=2.5,
            )

        # Format ROC Plot
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (0.50)")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
        ax_roc.set_title(f"ROC Curves — {ds.upper()}", fontweight="bold")
        ax_roc.legend(loc="lower right", frameon=True)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])

        # Format PR Plot
        ax_pr.set_xlabel("Recall (Sensitivity)", fontweight="bold")
        ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
        ax_pr.set_title(f"Precision-Recall Curves — {ds.upper()}", fontweight="bold")
        ax_pr.legend(loc="lower left", frameon=True)
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"roc_pr_curves_{ds}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved ROC/PR curves: {out_path}")


# ----------------------------------------------------------------------
# 2. Plot Confusion Matrices (Grid Layout)
# ----------------------------------------------------------------------
def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    datasets = ["aptos2019", "deepdrid"]

    for row_idx, ds in enumerate(datasets):
        for col_idx, (model_label, model_key) in enumerate(MODELS.items()):
            ax = axes[row_idx, col_idx]
            csv_path = os.path.join(REPORTS_DIR, f"{model_key}_{ds}_predictions.csv")

            if not os.path.exists(csv_path):
                ax.text(0.5, 0.5, "Data Missing", ha="center", va="center")
                continue

            df = pd.read_csv(csv_path)
            y_true = df["target"].values
            y_pred = (df["cal_prob"] >= 0.5).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Custom text annotations with raw counts + percentages
            labels = np.array(
                [
                    [
                        f"TN\n{cm[0,0]}\n({cm_norm[0,0]*100:.1f}%)",
                        f"FP\n{cm[0,1]}\n({cm_norm[0,1]*100:.1f}%)",
                    ],
                    [
                        f"FN\n{cm[1,0]}\n({cm_norm[1,0]*100:.1f}%)",
                        f"TP\n{cm[1,1]}\n({cm_norm[1,1]*100:.1f}%)",
                    ],
                ]
            )

            sns.heatmap(
                cm,
                annot=labels,
                fmt="",
                cmap="Blues" if model_label == "ConvNeXt-V2-Large" else "Reds",
                cbar=False,
                ax=ax,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=["Non-Referable", "Referable"],
                yticklabels=["Non-Referable", "Referable"],
            )

            ax.set_title(f"{model_label} — {ds.upper()}", fontweight="bold")
            ax.set_xlabel("Predicted Label", fontweight="bold")
            ax.set_ylabel("True Ground Truth Label", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confusion_matrices_grid.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices grid: {out_path}")


# ----------------------------------------------------------------------
# 3. Multi-Metric Comparison Bar Charts (Accuracy, Precision, Recall, Specificity, F1)
# ----------------------------------------------------------------------
def plot_multi_metric_comparison():
    metrics_list = ["accuracy", "precision", "sensitivity", "specificity", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall (Sens.)", "Specificity", "F1-Score"]

    datasets = ["aptos2019", "deepdrid_overall"]

    for ds in datasets:
        data_to_plot = []

        for model_label, model_key in MODELS.items():
            json_path = os.path.join(REPORTS_DIR, f"{model_key}_eval.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path) as f:
                res = json.load(f)

            if ds in res["datasets"] and res["datasets"][ds] is not None:
                m = res["datasets"][ds]
                for metric_key, metric_name in zip(metrics_list, metric_labels):
                    data_to_plot.append(
                        {
                            "Model": model_label,
                            "Metric": metric_name,
                            "Score": m.get(metric_key, 0.0) * 100,  # convert to %
                        }
                    )

        if not data_to_plot:
            continue

        df_plot = pd.DataFrame(data_to_plot)

        plt.figure(figsize=(9, 5.5))
        ax = sns.barplot(
            data=df_plot,
            x="Metric",
            y="Score",
            hue="Model",
            palette={"ConvNeXt-V2-Large": "#1f77b4", "RETFound": "#d62728"},
        )

        # Annotate bars with exact percentages
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontweight="bold",
                )

        clean_ds_title = "APTOS 2019" if "aptos" in ds else "DeepDRiD"
        plt.title(f"Comprehensive Metric Comparison — {clean_ds_title}", fontweight="bold")
        plt.ylabel("Score (%)", fontweight="bold")
        plt.xlabel("Evaluation Metric", fontweight="bold")
        plt.ylim(50, 105)
        plt.legend(title="Model", frameon=True)
        plt.tight_layout()

        out_path = os.path.join(FIGURES_DIR, f"full_metrics_comparison_{ds}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved full metrics bar chart: {out_path}")


# ----------------------------------------------------------------------
# 4. Training vs Validation Curves (Loss & Accuracy over Epochs)
# ----------------------------------------------------------------------
def plot_training_curves():
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
    found_any = False

    for model_label, model_key in MODELS.items():
        history_path = os.path.join(REPORTS_DIR, f"{model_key}_history.json")
        if not os.path.exists(history_path):
            continue

        found_any = True
        with open(history_path) as f:
            hist = json.load(f)

        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss Plot
        ax_loss.plot(
            epochs,
            hist["train_loss"],
            "--",
            label=f"{model_label} (Train)",
            color=COLORS[model_label],
            alpha=0.7,
        )
        ax_loss.plot(
            epochs,
            hist["val_loss"],
            "-",
            label=f"{model_label} (Val)",
            color=COLORS[model_label],
            linewidth=2,
        )

        # Accuracy Plot
        ax_acc.plot(
            epochs,
            hist["train_acc"],
            "--",
            label=f"{model_label} (Train)",
            color=COLORS[model_label],
            alpha=0.7,
        )
        ax_acc.plot(
            epochs,
            hist["val_acc"],
            "-",
            label=f"{model_label} (Val)",
            color=COLORS[model_label],
            linewidth=2,
        )

    if found_any:
        ax_loss.set_title("Training & Validation Loss", fontweight="bold")
        ax_loss.set_xlabel("Epoch", fontweight="bold")
        ax_loss.set_ylabel("Loss", fontweight="bold")
        ax_loss.legend(frameon=True)

        ax_acc.set_title("Training & Validation Accuracy", fontweight="bold")
        ax_acc.set_xlabel("Epoch", fontweight="bold")
        ax_acc.set_ylabel("Accuracy (%)", fontweight="bold")
        ax_acc.legend(frameon=True)

        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, "training_validation_curves.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved training/validation curves: {out_path}")
    else:
        plt.close()
        print("Note: No training history files (`*_history.json`) found in outputs/reports/. Skipping training curves plot.")


if __name__ == "__main__":
    print("=======================================================")
    print("   GENERATING PUBLICATION & THESIS FIGURES")
    print("=======================================================\n")
    plot_curves()
    plot_confusion_matrices()
    plot_multi_metric_comparison()
    plot_training_curves()
    print("\nAll publication plots successfully generated in `outputs/reports/figures/`!")