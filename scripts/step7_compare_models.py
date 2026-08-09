"""
Step 7 — Head-to-Head Model Comparison (corrected)

Fixes applied vs. the original draft:
  1. Bootstrap 95% confidence intervals and a paired bootstrap significance
     test for the AUC difference between models, on every dataset. Point
     estimates alone (what the original script printed) cannot tell you
     whether one model actually outperforms the other or whether the gap
     is just sampling noise -- see the demonstration in chat before this
     script for a concrete example of two identical-skill models producing
     a 2.6-point AUC "difference" from noise alone.
  2. DeepDRiD's quality-stratified comparison (good vs. poor image quality)
     is now included. The original script only compared aptos2019 and
     deepdrid_overall -- it never touched the quality breakdown, which is
     the entire reason DeepDRiD was chosen as the primary test set.
  3. Sample-size consistency check: warns explicitly if the two models
     were evaluated on different numbers of images for the same dataset
     (can happen if each teammate's raw-data extraction had different
     missing/corrupt files) -- a silent mismatch here would make the
     comparison itself questionable.
  4. No more silent plot-skipping -- if data for a chart is missing, the
     script says so explicitly instead of just not producing the file.
  5. Dynamic y-axis limits instead of a hardcoded (0.7, 1.0) that would
     clip any subgroup scoring below 0.7 AUC off the chart.
  6. seaborn added to requirements.txt (was missing, would have crashed
     on a fresh install).

REQUIRES: this script reads the PER-SAMPLE prediction CSVs that Step 6
now saves (outputs/reports/{model}_aptos2019_predictions.csv and
{model}_deepdrid_predictions.csv), not just the summary JSON files --
the JSON alone has no way to support confidence intervals since the
per-sample data can't be reconstructed from an aggregate.

Run once, from the project root, with your virtual environment active:
    python scripts/step7_compare_models.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 2000
SEED = 42
np.random.seed(SEED)

MODEL_DISPLAY_NAMES = {
    "convnext_v2_large": "ConvNeXt-V2-Large",
    "retfound": "RETFound",
}
MODEL_COLORS = {
    "convnext_v2_large": "#2b5c8f",
    "retfound": "#d95f02",
}


def load_predictions(out_dir, model_key, dataset_name):
    path = os.path.join(out_dir, f"{model_key}_{dataset_name}_predictions.csv")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found -- was Step 6 run with the version that saves "
              f"per-sample predictions? Re-run Step 6 for {model_key} if this file is missing.")
        return None
    return pd.read_csv(path)


def bootstrap_auc_ci(targets, probs, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Returns (point_estimate, ci_lower, ci_upper) via case resampling."""
    rng = np.random.RandomState(seed)
    n = len(targets)
    point = roc_auc_score(targets, probs)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(targets[idx])) < 2:
            continue  # skip degenerate resamples with only one class present
        boot_aucs.append(roc_auc_score(targets[idx], probs[idx]))
    lower, upper = np.percentile(boot_aucs, [2.5, 97.5])
    return point, lower, upper


def paired_bootstrap_auc_diff(df_a, df_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Paired bootstrap test for whether model A's AUC differs from model B's,
    on the SAME underlying images. Aligns the two models' predictions by
    image_path (NOT by row order, which is not safe to assume -- if the two
    evaluation runs excluded different missing/corrupt files, row i in one
    file will not correspond to row i in the other, and a row-order pairing
    would silently compare the wrong images against each other).
    Returns (observed_diff, ci_lower, ci_upper, two_sided_p_approx, n_aligned).
    """
    merged = df_a.merge(df_b, on="image_path", suffixes=("_a", "_b"), how="inner")
    n_a, n_b, n_aligned = len(df_a), len(df_b), len(merged)
    if n_aligned < n_a or n_aligned < n_b:
        print(f"  [WARN] Only {n_aligned} of {n_a}/{n_b} images matched by image_path between "
              f"the two models -- the paired test uses only this aligned overlap. This usually "
              f"means the two evaluation runs excluded different missing/corrupt files.")

    mismatched_labels = merged["target_a"] != merged["target_b"]
    if mismatched_labels.any():
        raise ValueError(
            f"{mismatched_labels.sum()} images have DIFFERENT target labels between the two "
            f"models' prediction files, despite matching image_path. This means the two runs "
            f"are not evaluating the same ground truth for the same image -- stop and "
            f"investigate before trusting any comparison here."
        )

    targets = merged["target_a"].values
    probs_a = merged["cal_prob_a"].values
    probs_b = merged["cal_prob_b"].values

    rng = np.random.RandomState(seed)
    n = len(targets)
    observed_diff = roc_auc_score(targets, probs_a) - roc_auc_score(targets, probs_b)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        t = targets[idx]
        if len(np.unique(t)) < 2:
            continue
        auc_a = roc_auc_score(t, probs_a[idx])
        auc_b = roc_auc_score(t, probs_b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.array(diffs)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    p_approx = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return observed_diff, lower, upper, min(p_approx, 1.0), n_aligned


def compare_on_dataset(out_dir, dataset_name, models_present):
    """Returns a results dict for one dataset (aptos2019 or deepdrid), across whichever models are present."""
    preds = {}
    for model_key in models_present:
        df = load_predictions(out_dir, model_key, dataset_name)
        if df is not None:
            preds[model_key] = df

    if len(preds) == 0:
        return None

    # FIX #3 -- sample size consistency check
    sizes = {k: len(v) for k, v in preds.items()}
    if len(set(sizes.values())) > 1:
        print(f"  [WARN] {dataset_name}: models were evaluated on DIFFERENT sample counts: {sizes}. "
              f"This likely means each teammate's raw-data extraction had different missing/corrupt "
              f"files. A head-to-head comparison on mismatched test pools is weaker evidence than "
              f"one on an identical pool -- worth noting explicitly if you report this result.")

    rows = []
    for model_key, df in preds.items():
        targets = df["target"].values
        probs = df["cal_prob"].values
        point, lo, hi = bootstrap_auc_ci(targets, probs)
        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
            "Dataset": dataset_name,
            "n_samples": len(df),
            "auc_roc": point,
            "auc_ci_lower": lo,
            "auc_ci_upper": hi,
        })

    # Paired significance test, aligned by image_path -- only if both models present
    if len(preds) == 2:
        keys = list(preds.keys())
        df_a, df_b = preds[keys[0]], preds[keys[1]]
        try:
            diff, dlo, dhi, p, n_aligned = paired_bootstrap_auc_diff(df_a, df_b)
            significant = not (dlo <= 0 <= dhi)
            print(f"  {dataset_name}: AUC({MODEL_DISPLAY_NAMES.get(keys[0],keys[0])}) - "
                  f"AUC({MODEL_DISPLAY_NAMES.get(keys[1],keys[1])}) = {diff:+.4f} (n={n_aligned} aligned images), "
                  f"95% CI [{dlo:+.4f}, {dhi:+.4f}], approx p={p:.4f} -- "
                  f"{'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant (CI crosses zero)'}")
        except ValueError as e:
            print(f"  [ERROR] {dataset_name}: {e}")

    return rows


def compare_quality_stratified(out_dir, models_present):
    """DeepDRiD quality-stratified comparison -- the part the original script omitted entirely."""
    rows = []
    quality_data = {}
    for model_key in models_present:
        df = load_predictions(out_dir, model_key, "deepdrid")
        if df is None or "overall_quality" not in df.columns:
            continue
        quality_data[model_key] = df

    if not quality_data:
        print("  [INFO] No DeepDRiD predictions with an 'overall_quality' column found -- "
              "skipping quality-stratified comparison.")
        return rows

    all_quality_values = sorted(set().union(*[set(df["overall_quality"].dropna().unique()) for df in quality_data.values()]))

    for q_val in all_quality_values:
        for model_key, df in quality_data.items():
            subset = df[df["overall_quality"] == q_val]
            if len(subset) < 5:
                print(f"  [WARN] {MODEL_DISPLAY_NAMES.get(model_key,model_key)}, quality={q_val}: "
                      f"only {len(subset)} samples -- confidence interval will be wide, interpret cautiously.")
            if len(subset) == 0 or len(subset["target"].unique()) < 2:
                continue
            point, lo, hi = bootstrap_auc_ci(subset["target"].values, subset["cal_prob"].values)
            rows.append({
                "Model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
                "overall_quality": q_val,
                "n_samples": len(subset),
                "auc_roc": point,
                "auc_ci_lower": lo,
                "auc_ci_upper": hi,
            })
    return rows


def plot_auc_comparison(df_comp, out_dir, group_col="Dataset", title="Head-to-Head External Generalization (ROC-AUC)", filename="head_to_head_roc_auc.png"):
    groups = sorted(df_comp[group_col].unique())
    models = sorted(df_comp["Model"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.35 if len(models) == 2 else 0.8 / max(len(models), 1)

    plotted_any = False
    for i, model in enumerate(models):
        sub = df_comp[df_comp["Model"] == model].set_index(group_col).reindex(groups)
        if sub["auc_roc"].isna().all():
            print(f"  [WARN] No data for model '{model}' across groups {groups} -- skipping its bars.")
            continue
        offset = (i - (len(models) - 1) / 2) * width
        yerr = np.array([
            sub["auc_roc"] - sub["auc_ci_lower"],
            sub["auc_ci_upper"] - sub["auc_roc"],
        ])
        ax.bar(x + offset, sub["auc_roc"], width, yerr=yerr, capsize=4,
               label=model, color=MODEL_COLORS.get(model.lower().replace("-", "_").replace(" ", "_"), None))
        plotted_any = True

    if not plotted_any:
        print(f"  [WARN] Nothing to plot for '{title}' -- no model had data across any group.")
        plt.close(fig)
        return

    ax.set_xlabel(group_col)
    ax.set_ylabel("ROC-AUC (with 95% bootstrap CI)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    # FIX #5 -- dynamic y-limits instead of a hardcoded (0.7, 1.0) that clips low-scoring subgroups
    y_min = max(0.0, df_comp["auc_ci_lower"].min() - 0.05)
    y_max = min(1.0, df_comp["auc_ci_upper"].max() + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.legend()
    plt.tight_layout()

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, filename)
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"  Saved plot to: {fig_path}")


def compare_models(out_dir="outputs/reports"):
    models_present = [m for m in MODEL_DISPLAY_NAMES if os.path.exists(
        os.path.join(out_dir, f"{m}_aptos2019_predictions.csv")
    ) or os.path.exists(os.path.join(out_dir, f"{m}_deepdrid_predictions.csv"))]

    if len(models_present) == 0:
        print("[ERROR] No per-sample prediction CSVs found in outputs/reports/. "
              "Run the current version of step6_evaluate_external.py for each model first "
              "-- it now saves these alongside the summary JSON.")
        return

    print(f"Found predictions for: {[MODEL_DISPLAY_NAMES.get(m,m) for m in models_present]}\n")

    print("=" * 60)
    print("APTOS 2019 comparison")
    print("=" * 60)
    aptos_rows = compare_on_dataset(out_dir, "aptos2019", models_present)

    print("\n" + "=" * 60)
    print("DeepDRiD (overall) comparison")
    print("=" * 60)
    deepdrid_rows = compare_on_dataset(out_dir, "deepdrid", models_present)

    print("\n" + "=" * 60)
    print("DeepDRiD quality-stratified comparison")
    print("=" * 60)
    quality_rows = compare_quality_stratified(out_dir, models_present)

    all_rows = (aptos_rows or []) + (deepdrid_rows or [])
    df_comp = pd.DataFrame(all_rows)
    df_quality = pd.DataFrame(quality_rows)

    if not df_comp.empty:
        csv_path = os.path.join(out_dir, "model_comparison_summary.csv")
        df_comp.to_csv(csv_path, index=False)
        print(f"\nSaved overall comparison table to: {csv_path}")
        print(df_comp.to_string(index=False))
        plot_auc_comparison(df_comp, out_dir, group_col="Dataset")

    if not df_quality.empty:
        q_csv_path = os.path.join(out_dir, "model_comparison_by_quality.csv")
        df_quality.to_csv(q_csv_path, index=False)
        print(f"\nSaved quality-stratified comparison table to: {q_csv_path}")
        print(df_quality.to_string(index=False))
        plot_auc_comparison(
            df_quality, out_dir, group_col="overall_quality",
            title="AUC by DeepDRiD Image Quality Bucket",
            filename="quality_stratified_roc_auc.png",
        )

    print("\nStep 7 comparison complete.")


if __name__ == "__main__":
    compare_models()
