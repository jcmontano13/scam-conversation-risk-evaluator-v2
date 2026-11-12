"""
generate_five_model_figures.py

Description:
    Generates the figures for the COMP903 final report and presentation from
    the saved result files of stories 13 to 18. No model is trained and no
    data is re-split here: every figure is drawn from an existing output file,
    so the figures always match the reported numbers.

    Each figure is saved as a PNG with a short summary text file, following
    the COMP902 convention. Curve figures also save their plotted points as
    a CSV.

    Figures:
    1. model_comparison_significance.png  (RQ1 and RQ2, repeated CV)
    2. test_confusion_matrices.png        (held-out test)
    3. roc_curves_test_set.png            (held-out test)
    4. precision_recall_curves_test_set.png (held-out test)
    5. reliability_curves_test_set.png    (held-out test, uncalibrated probabilities)
    6. calibration_brier_comparison.png   (RQ2, before and after calibration)
    7. ablation_f1_drop_heatmap.png       (RQ3)
    8. generalisation_recall.png          (extension, if its results exist)

Inputs:
    - outputs/reports/significance_tests.csv
    - outputs/reports/test_results_five_models.csv
    - outputs/reports/test_predictions_five_models.csv
    - outputs/reports/calibration_results_five_models.csv
    - outputs/reports/ablation_results_five_models.csv
    - outputs/reports/leave_one_type_out_results.csv (optional)

Outputs:
    - outputs/figures/ (PNG, summary TXT and points CSV files)

Run:
    - python src/reporting/generate_five_model_figures.py
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import (
    ABLATION_RESULTS_FIVE_MODELS,
    CALIBRATION_RESULTS_FIVE_MODELS,
    FIGURES_DIR,
    LOTO_RESULTS,
    MODEL_NAMES,
    NEW_MODELS,
    REFERENCE_MODEL,
    SIGNIFICANCE_TESTS,
    TARGET_COLUMN,
    TEST_PREDICTIONS_FIVE_MODELS,
    TEST_RESULTS_FIVE_MODELS,
)

# Colours: one accent for the model being shown, a quiet grey for references.
ACCENT = "#2a78d6"
REFERENCE_GREY = "#a3a29c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#f4f8fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281"]
)

SHORT_NAMES = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "Gaussian Naive Bayes": "Naive Bayes",
    "Random Forest": "Random Forest",
    "k-Nearest Neighbours": "k-NN",
}

DPI = 200


def apply_style():
    """
    Description:
        Sets a clean, consistent style for every figure: light grid, no top
        or right borders, and text in neutral colours.
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_csv(file_path, description):
    """
    Description:
        Loads a result CSV, with a clear message naming the script that
        creates it if it is missing.

    Input:
        file_path: Path to the CSV file.
        description: Name of the script that creates the file.

    Output:
        Returns a pandas DataFrame.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{Path(file_path).name} not found. Run {description} first.")
    return pd.read_csv(file_path)


def save_figure(fig, name, summary_lines):
    """
    Description:
        Saves a figure as PNG and its summary as a text file.

    Input:
        fig: Matplotlib figure.
        name: File name without extension.
        summary_lines: List of text lines describing the figure.

    Output:
        Creates <name>.png and <name>_summary.txt in outputs/figures/.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    text = "\n".join(summary_lines + ["", f"Generated at: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"])
    (FIGURES_DIR / f"{name}_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(f"Saved {name}.png")


def figure_model_comparison(significance_df):
    """
    Description:
        Figure 1 (RQ1 and RQ2). Shows each new model's difference from
        Logistic Regression in F1 and Brier score, with the 95% confidence
        interval from the corrected resampled t-test. An interval that
        crosses zero means the difference could be due to chance.
    """
    panels = [
        ("f1", "F1-score difference", "Higher is better"),
        ("brier_score", "Brier score difference (before calibration)", "Lower is better"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    summary = [
        "Figure: New models compared with Logistic Regression (repeated cross-validation, 50 paired folds)",
        "",
        "Each point is the mean difference from Logistic Regression; the line is the 95% confidence",
        "interval from the corrected resampled t-test. Holm-corrected significance is shown at the right.",
        "",
    ]

    for ax, (metric, title, direction) in zip(axes, panels):
        data = significance_df[significance_df["metric"] == metric].set_index("model").loc[NEW_MODELS]
        positions = np.arange(len(NEW_MODELS))[::-1]

        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        for y_pos, (model_name, row) in zip(positions, data.iterrows()):
            ax.plot([row["ci_low"], row["ci_high"]], [y_pos, y_pos], color=ACCENT, linewidth=2, solid_capstyle="round")
            ax.plot(row["mean_difference"], y_pos, "o", color=ACCENT, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
            label = "significant" if row["significant"] else "not significant"
            ax.annotate(
                label, xy=(1.0, y_pos), xycoords=("axes fraction", "data"), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8.5,
                color=TEXT_PRIMARY if row["significant"] else TEXT_SECONDARY,
                fontweight="bold" if row["significant"] else "normal",
            )
            summary.append(
                f"{title} | {model_name}: {row['mean_difference']:+.4f} "
                f"(95% CI {row['ci_low']:+.4f} to {row['ci_high']:+.4f}), Holm p = {row['p_value_holm']:.4f}, {label}"
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([SHORT_NAMES[m] for m in NEW_MODELS])
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Difference from {REFERENCE_MODEL}  ({direction.lower()})")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(NEW_MODELS) - 0.4)

    fig.suptitle("No new model classifies significantly better than Logistic Regression",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "model_comparison_significance", summary)


def figure_confusion_matrices(results_df):
    """
    Description:
        Figure 2. Confusion matrices for all five models on the held-out
        test set, with counts written in each cell.
    """
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0))
    vmax = results_df[["true_negative", "false_positive", "false_negative", "true_positive"]].to_numpy().max()
    summary = ["Figure: Confusion matrices on the held-out test set (320 conversations)", ""]

    for ax, model_name in zip(axes, MODEL_NAMES):
        row = results_df.set_index("model").loc[model_name]
        matrix = np.array(
            [[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]]
        ).astype(int)

        ax.imshow(matrix, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(j, i, f"{value}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if value > vmax * 0.55 else TEXT_PRIMARY)

        ax.set_title(SHORT_NAMES[model_name], fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-scam", "Scam"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-scam", "Scam"] if model_name == MODEL_NAMES[0] else ["", ""])
        ax.set_xlabel("Predicted")
        if model_name == MODEL_NAMES[0]:
            ax.set_ylabel("Actual")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        summary.append(
            f"{model_name}: TN {row['true_negative']}, FP (false alarms) {row['false_positive']}, "
            f"FN (missed scams) {row['false_negative']}, TP {row['true_positive']}"
        )

    fig.suptitle("Logistic Regression misses the fewest scams; k-NN raises the fewest false alarms",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "test_confusion_matrices", summary)


def small_multiple_curves(predictions_df, curve_type):
    """
    Description:
        Figures 3 and 4. One panel per model on the held-out test set. The
        model's curve is drawn in the accent colour; Logistic Regression is
        drawn in grey in every other panel as a reference.

    Input:
        predictions_df: Test predictions for every model.
        curve_type: "roc" or "pr".
    """
    y_true = predictions_df[TARGET_COLUMN].to_numpy()
    curves, points, scores = {}, [], {}

    for model_name in MODEL_NAMES:
        y_prob = predictions_df[f"{model_name} | scam_probability"].to_numpy()
        if curve_type == "roc":
            x, y, _ = roc_curve(y_true, y_prob)
            scores[model_name] = auc(x, y)
        else:
            y, x, _ = precision_recall_curve(y_true, y_prob)
            scores[model_name] = average_precision_score(y_true, y_prob)
        curves[model_name] = (x, y)
        points += [{"model": model_name, "x": a, "y": b} for a, b in zip(x, y)]

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    reference_x, reference_y = curves[REFERENCE_MODEL]

    for ax, model_name in zip(axes, MODEL_NAMES):
        x, y = curves[model_name]
        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], linestyle=":", color=GRID, linewidth=1.2)
        if model_name != REFERENCE_MODEL:
            ax.plot(reference_x, reference_y, color=REFERENCE_GREY, linewidth=1.5, label=REFERENCE_MODEL)
        ax.plot(x, y, color=ACCENT, linewidth=2, label=SHORT_NAMES[model_name])

        score_name = "AUC" if curve_type == "roc" else "AP"
        ax.set_title(f"{SHORT_NAMES[model_name]}\n{score_name} {scores[model_name]:.3f}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate" if curve_type == "roc" else "Recall")
        if model_name == MODEL_NAMES[0]:
            ax.set_ylabel("True positive rate" if curve_type == "roc" else "Precision")
        if model_name == MODEL_NAMES[1]:
            ax.legend(loc="lower right", fontsize=7.5, frameon=False)

    if curve_type == "roc":
        name, label = "roc_curves_test_set", "ROC"
        headline = "All five models separate scams from non-scams almost equally well (AUC 0.94 to 0.96)"
    else:
        name, label = "precision_recall_curves_test_set", "Precision-recall"
        headline = "Precision-recall curves overlap closely across the five models"

    fig.suptitle(headline, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    pd.DataFrame(points).to_csv(FIGURES_DIR / f"{name}_points.csv", index=False)
    summary = [f"Figure: {label} curves on the held-out test set (320 conversations)", "",
               f"Grey line in each panel = {REFERENCE_MODEL}, for reference.", ""]
    summary += [f"{m}: {'AUC' if curve_type == 'roc' else 'average precision'} {scores[m]:.4f}" for m in MODEL_NAMES]
    save_figure(fig, name, summary)


def figure_reliability_curves(predictions_df):
    """
    Description:
        Figure 5. Reliability curves on the held-out test set for the tuned
        (uncalibrated) models. A perfectly calibrated model follows the
        diagonal: when it says 70%, 70% of those conversations are scams.
    """
    y_true = predictions_df[TARGET_COLUMN].to_numpy()
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    points = []
    summary = [
        "Figure: Reliability curves on the held-out test set (tuned models, before calibration)",
        "",
        "Conversations are grouped into 5 bins of equal size by predicted probability.",
        "Points on the diagonal mean the predicted probability matches the actual scam rate.",
        "",
    ]

    for ax, model_name in zip(axes, MODEL_NAMES):
        y_prob = predictions_df[f"{model_name} | scam_probability"].to_numpy()
        actual, predicted = calibration_curve(y_true, y_prob, n_bins=5, strategy="quantile")

        ax.plot([0, 1], [0, 1], linestyle=":", color=REFERENCE_GREY, linewidth=1.2)
        ax.plot(predicted, actual, "-o", color=ACCENT, linewidth=2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(SHORT_NAMES[model_name], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Predicted scam probability")
        if model_name == MODEL_NAMES[0]:
            ax.set_ylabel("Actual share of scams")

        points += [{"model": model_name, "predicted": p, "actual": a} for p, a in zip(predicted, actual)]
        summary.append(f"{model_name}: largest gap from the diagonal {np.max(np.abs(predicted - actual)):.3f}")

    fig.suptitle("Naive Bayes probabilities stray furthest from reality on the test set; Random Forest stays closest",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pd.DataFrame(points).to_csv(FIGURES_DIR / "reliability_curves_test_set_points.csv", index=False)
    save_figure(fig, "reliability_curves_test_set", summary)


def figure_calibration_brier(calibration_df):
    """
    Description:
        Figure 6 (RQ2). For each model, the Brier score before calibration
        (grey) and after its selected calibration method (accent), from
        repeated cross-validation on the training set. Lower is better.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    positions = np.arange(len(MODEL_NAMES))[::-1]
    summary = ["Figure: Brier score before and after calibration (repeated cross-validation, training set)",
               "", "Lower is better. Grey = uncalibrated; blue = selected calibration method.", ""]

    for y_pos, model_name in zip(positions, MODEL_NAMES):
        rows = calibration_df[calibration_df["model"] == model_name]
        before = rows[rows["method"] == "uncalibrated"]["brier_score_mean"].iloc[0]
        selected = rows[rows["selected"]].iloc[0]
        after = selected["brier_score_mean"]

        ax.plot([before, after], [y_pos, y_pos], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.plot(before, y_pos, "o", color=REFERENCE_GREY, markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        ax.plot(after, y_pos, "o", color=ACCENT, markersize=9, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{after:.4f} ({selected['method']})", xy=(min(before, after), y_pos), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.5, color=TEXT_SECONDARY)
        summary.append(f"{model_name}: uncalibrated {before:.4f} -> {selected['method']} {after:.4f} (change {after - before:+.4f})")

    ax.plot([], [], "o", color=REFERENCE_GREY, label="Before calibration")
    ax.plot([], [], "o", color=ACCENT, label="After calibration")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    ax.set_yticks(positions)
    ax.set_yticklabels([SHORT_NAMES[m] for m in MODEL_NAMES])
    ax.set_xlabel("Brier score (lower is better)")
    left = calibration_df["brier_score_mean"].min()
    ax.set_xlim(left - 0.009, calibration_df["brier_score_mean"].max() + 0.002)
    ax.grid(axis="y", visible=False)
    ax.set_title("Calibration helps Naive Bayes most; Random Forest and k-NN barely change", loc="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "calibration_brier_comparison", summary)


def figure_ablation(ablation_df):
    """
    Description:
        Figure 7 (RQ3). Heatmap of the F1 drop when each feature is removed,
        for every model. Darker = the model relies more on that feature.
        An asterisk marks a significant drop.
    """
    feature_order = (
        ablation_df.groupby("removed_feature")["f1_drop"].mean().sort_values(ascending=False).index.tolist()
    )
    matrix = (
        ablation_df.pivot(index="model", columns="removed_feature", values="f1_drop")
        .loc[MODEL_NAMES, feature_order]
    )
    significant = (
        ablation_df.pivot(index="model", columns="removed_feature", values="significant")
        .loc[MODEL_NAMES, feature_order]
    )

    fig, ax = plt.subplots(figsize=(8, 3.6))
    vmax = matrix.to_numpy().max()
    ax.imshow(matrix.clip(lower=0), cmap=SEQUENTIAL_BLUE, vmin=0, vmax=vmax, aspect="auto")

    for i, model_name in enumerate(MODEL_NAMES):
        for j, feature in enumerate(feature_order):
            value = matrix.loc[model_name, feature]
            value = 0.0 if abs(value) < 0.0005 else value  # avoid printing "-0.000"
            marker = "*" if significant.loc[model_name, feature] else ""
            ax.text(j, i, f"{value:+.3f}{marker}" if value else "0.000", ha="center", va="center", fontsize=9.5,
                    color="white" if value > vmax * 0.55 else TEXT_PRIMARY)

    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels(feature_order, fontsize=9)
    ax.set_yticks(range(len(MODEL_NAMES)))
    ax.set_yticklabels([SHORT_NAMES[m] for m in MODEL_NAMES])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Feature removed")
    ax.set_title("Every model relies most on turn_count, then aggregate_money_terms", loc="left", fontsize=12)
    fig.text(0.01, -0.04, "Value = drop in F1 when the feature is removed (repeated CV, training set). "
             "* = significant after Holm correction.", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()

    summary = ["Figure: Drop in F1 when each feature is removed (ablation, repeated cross-validation)", ""]
    for model_name in MODEL_NAMES:
        ranked = matrix.loc[model_name].sort_values(ascending=False)
        summary.append(f"{model_name}: " + ", ".join(f"{f} {v:+.4f}" for f, v in ranked.items()))
    save_figure(fig, "ablation_f1_drop_heatmap", summary)


def figure_generalisation(loto_df):
    """
    Description:
        Figure 8 (extension). One panel per scam type. For each model, recall
        when the scam type was seen in training (grey) and when it was never
        seen (accent). The line between them is the generalisation gap.
    """
    scam_types = list(dict.fromkeys(loto_df["scam_type"]))
    fig, axes = plt.subplots(1, len(scam_types), figsize=(13, 3.4), sharey=True, sharex=True)
    positions = np.arange(len(MODEL_NAMES))[::-1]
    summary = ["Figure: Recall on each scam type when seen in training (control) and when never seen", "",
               "Grey = seen (control); blue = never seen. Training split only.", ""]

    for ax, scam_type in zip(axes, scam_types):
        data = loto_df[loto_df["scam_type"] == scam_type].set_index("model").loc[MODEL_NAMES]
        for y_pos, (model_name, row) in zip(positions, data.iterrows()):
            ax.plot([row["recall_seen"], row["recall_unseen"]], [y_pos, y_pos], color=GRID, linewidth=3,
                    solid_capstyle="round", zorder=1)
            ax.plot(row["recall_seen"], y_pos, "o", color=REFERENCE_GREY, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.3, zorder=2)
            ax.plot(row["recall_unseen"], y_pos, "o", color=ACCENT, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.3, zorder=3)
            summary.append(
                f"{scam_type} | {model_name}: seen {row['recall_seen']:.3f}, unseen {row['recall_unseen']:.3f}, "
                f"gap {row['generalisation_gap']:+.3f}"
            )

        ax.set_title(f"{scam_type}  (avg unseen {data['recall_unseen'].mean():.2f})", fontsize=10)
        ax.set_yticks(positions)
        ax.set_yticklabels([SHORT_NAMES[m] for m in MODEL_NAMES])
        ax.set_xlim(0.4, 1.02)
        ax.set_xlabel("Recall")
        ax.grid(axis="y", visible=False)

    axes[-1].plot([], [], "o", color=REFERENCE_GREY, label="Seen (control)")
    axes[-1].plot([], [], "o", color=ACCENT, label="Never seen")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("Tech support scams are hardest to detect, and hardest of all when never seen",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "generalisation_recall", summary)


def main():
    """
    Description:
        Loads the saved result files and generates every figure.
    """
    apply_style()

    significance_df = load_csv(SIGNIFICANCE_TESTS, "src/evaluation/significance_tests.py")
    results_df = load_csv(TEST_RESULTS_FIVE_MODELS, "src/evaluation/evaluate_test_set_five_models.py")
    predictions_df = load_csv(TEST_PREDICTIONS_FIVE_MODELS, "src/evaluation/evaluate_test_set_five_models.py")
    calibration_df = load_csv(CALIBRATION_RESULTS_FIVE_MODELS, "src/evaluation/calibration_five_models.py")
    ablation_df = load_csv(ABLATION_RESULTS_FIVE_MODELS, "src/evaluation/ablation_five_models.py")

    figure_model_comparison(significance_df)
    figure_confusion_matrices(results_df)
    small_multiple_curves(predictions_df, "roc")
    small_multiple_curves(predictions_df, "pr")
    figure_reliability_curves(predictions_df)
    figure_calibration_brier(calibration_df)
    figure_ablation(ablation_df)

    if Path(LOTO_RESULTS).exists():
        figure_generalisation(load_csv(LOTO_RESULTS, "src/evaluation/leave_one_type_out.py"))
    else:
        print("Skipped generalisation_recall.png (leave_one_type_out_results.csv not found)")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()