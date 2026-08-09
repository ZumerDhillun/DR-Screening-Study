import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def generate_plots(eval_json_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(eval_json_path) as f:
        data = json.load(f)

    model_key = data["model_key"]
    datasets = data["datasets"]

    # 1. Quality Stratification Bar Plot (DeepDRiD)
    if "deepdrid_by_quality" in datasets:
        q_data = datasets["deepdrid_by_quality"]
        qualities = list(q_data.keys())
        aucs = [
            q_data[q]["auc_roc"]
            for q in qualities
            if q_data[q]["auc_roc"] is not None
        ]
        f1s = [
            q_data[q]["f1_score"]
            for q in qualities
            if q_data[q]["f1_score"] is not None
        ]

        plt.figure(figsize=(8, 5))
        x = np.arange(len(qualities))
        width = 0.35

        plt.bar(x - width / 2, aucs, width, label="ROC-AUC", color="#1f77b4")
        plt.bar(x + width / 2, f1s, width, label="F1-Score", color="#ff7f0e")

        plt.xlabel("DeepDRiD Image Quality Level")
        plt.ylabel("Score")
        plt.title(f"{model_key} Performance Across Image Quality Levels")
        plt.xticks(x, [f"Quality {q}" for q in qualities])
        plt.ylim(0.5, 1.0)
        plt.legend()
        plt.tight_layout()

        quality_fig_path = os.path.join(
            out_dir, f"{model_key}_quality_breakdown.png"
        )
        plt.savefig(quality_fig_path, dpi=300)
        plt.close()
        print(f" Saved quality plot: {quality_fig_path}")

    # 2. Performance Comparison Table / Bar Chart
    ds_names, auc_scores, f1_scores = [], [], []
    for name, metrics in datasets.items():
        if name != "deepdrid_by_quality" and metrics is not None:
            ds_names.append(name)
            auc_scores.append(metrics["auc_roc"])
            f1_scores.append(metrics["f1_score"])

    plt.figure(figsize=(8, 5))
    x = np.arange(len(ds_names))
    plt.bar(x - 0.17, auc_scores, 0.35, label="ROC-AUC", color="#2ca02c")
    plt.bar(x + 0.17, f1_scores, 0.35, label="F1-Score", color="#d62728")

    plt.xlabel("External Test Dataset")
    plt.ylabel("Score")
    plt.title(f"{model_key} External Generalization Scores")
    plt.xticks(x, ds_names)
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.tight_layout()

    summary_fig_path = os.path.join(
        out_dir, f"{model_key}_external_summary.png"
    )
    plt.savefig(summary_fig_path, dpi=300)
    plt.close()
    print(f" Saved external summary plot: {summary_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_json", default="outputs/reports/convnext_v2_large_eval.json"
    )
    parser.add_argument("--out_dir", default="outputs/reports/figures")
    args = parser.parse_args()

    generate_plots(args.eval_json, args.out_dir)