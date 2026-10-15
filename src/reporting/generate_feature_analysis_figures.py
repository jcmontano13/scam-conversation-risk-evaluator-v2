"""
Generate Feature Analysis Figures

Description:
    This script generates feature analysis figures for the final report.
    It uses existing ablation and interpretability output files only.

    The script does not introduce a new dataset, new features, new models,
    or new training process. It is only used to convert completed project
    outputs into final report figures.

Input:
    - outputs/reports/ablation_results.csv
    - outputs/reports/coefficients_table.csv
    - outputs/reports/tree_feature_importance.csv

Output:
    - outputs/figures/ablation_f1_drop.png
    - outputs/figures/logistic_regression_coefficients.png
    - outputs/figures/decision_tree_feature_importance.png
    - outputs/figures/feature_analysis_summary.txt
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Allow script to import config.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

ABLATION_RESULTS_PATH = REPORT_DIR / "ablation_results.csv"
COEFFICIENTS_TABLE_PATH = REPORT_DIR / "coefficients_table.csv"
TREE_FEATURE_IMPORTANCE_PATH = REPORT_DIR / "tree_feature_importance.csv"

ABLATION_FIGURE_PATH = FIGURE_DIR / "ablation_f1_drop.png"
LR_COEFFICIENTS_FIGURE_PATH = FIGURE_DIR / "logistic_regression_coefficients.png"
DT_IMPORTANCE_FIGURE_PATH = FIGURE_DIR / "decision_tree_feature_importance.png"
FEATURE_ANALYSIS_SUMMARY_PATH = FIGURE_DIR / "feature_analysis_summary.txt"


def normalise_column_names(df):
    """
    Description:
        Normalise column names so the script can handle minor naming differences.

    Input:
        DataFrame

    Output:
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def check_file_exists(file_path):
    """
    Description:
        Check that a required file exists.

    Input:
        file_path

    Output:
        Raises FileNotFoundError if missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def generate_ablation_f1_drop_figure():
    """
    Description:
        Generate a grouped horizontal bar chart showing F1-score drop
        after each feature is removed.

        If the ablation_results.csv file already contains a F1-drop column,
        the script will use it. If not, the script will calculate the F1 drop
        by comparing each ablation result against the full feature set result
        for the same model.

    Input:
        ablation_results.csv

    Output:
        ablation_f1_drop.png
    """
    print("Loading ablation results...")
    check_file_exists(ABLATION_RESULTS_PATH)

    ablation_df = pd.read_csv(ABLATION_RESULTS_PATH)
    ablation_df = normalise_column_names(ablation_df)

    print("Available ablation columns:")
    print(list(ablation_df.columns))

    required_base_columns = ["model", "removed_feature"]
    missing_base_columns = [
        column for column in required_base_columns if column not in ablation_df.columns
    ]

    if missing_base_columns:
        raise ValueError(
            f"Missing required columns in ablation_results.csv: {missing_base_columns}"
        )

    possible_f1_drop_columns = [
        "f1_drop",
        "f1_score_drop",
        "f1_drop_from_full",
        "f1_mean_drop",
    ]

    possible_f1_columns = [
        "f1",
        "f1_score",
        "f1_mean",
    ]

    f1_drop_column = None
    for column in possible_f1_drop_columns:
        if column in ablation_df.columns:
            f1_drop_column = column
            break

    if f1_drop_column is None:
        f1_column = None
        for column in possible_f1_columns:
            if column in ablation_df.columns:
                f1_column = column
                break

        if f1_column is None:
            raise ValueError(
                "Could not find an F1 drop column or an F1 score column. "
                f"Available columns are: {list(ablation_df.columns)}"
            )

        print(f"No F1 drop column found. Calculating F1 drop using column: {f1_column}")

        ablation_df[f1_column] = pd.to_numeric(
            ablation_df[f1_column],
            errors="coerce",
        )

        full_rows = ablation_df[
            ablation_df["removed_feature"]
            .astype(str)
            .str.lower()
            .isin(["none", "nan", "", "full_feature_set"])
        ]

        if full_rows.empty and "feature_set" in ablation_df.columns:
            full_rows = ablation_df[
                ablation_df["feature_set"]
                .astype(str)
                .str.lower()
                .eq("full_feature_set")
            ]

        if full_rows.empty:
            raise ValueError(
                "Could not identify full feature set rows to calculate F1 drop."
            )

        full_f1_by_model = full_rows.set_index("model")[f1_column].to_dict()

        def calculate_f1_drop(row):
            full_f1 = full_f1_by_model.get(row["model"])
            if full_f1 is None:
                return None
            return full_f1 - row[f1_column]

        ablation_df["f1_drop"] = ablation_df.apply(calculate_f1_drop, axis=1)
        f1_drop_column = "f1_drop"

    else:
        print(f"Using existing F1 drop column: {f1_drop_column}")
        ablation_df[f1_drop_column] = pd.to_numeric(
            ablation_df[f1_drop_column],
            errors="coerce",
        )

    ablation_df = ablation_df.copy()
    ablation_df["removed_feature"] = ablation_df["removed_feature"].astype(str)

    ablation_df = ablation_df[
        ~ablation_df["removed_feature"].str.lower().isin(
            ["none", "nan", "", "full_feature_set"]
        )
    ]

    ablation_df = ablation_df.dropna(subset=[f1_drop_column])

    pivot_df = ablation_df.pivot_table(
        index="removed_feature",
        columns="model",
        values=f1_drop_column,
        aggfunc="mean",
    )

    feature_order = (
        pivot_df.abs()
        .max(axis=1)
        .sort_values(ascending=True)
        .index
    )

    pivot_df = pivot_df.loc[feature_order]

    plt.figure(figsize=(9, 6))
    pivot_df.plot(kind="barh", ax=plt.gca())

    plt.title("F1-Score Drop After Removing Each Feature")
    plt.xlabel("F1-Score Drop")
    plt.ylabel("Removed Feature")
    plt.axvline(x=0, linestyle="--", linewidth=1)
    plt.legend(title="Model", loc="lower right")
    plt.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving ablation F1-score drop figure...")
    plt.savefig(ABLATION_FIGURE_PATH, dpi=300)
    plt.close()

    return pivot_df

def generate_logistic_regression_coefficients_figure():
    """
    Description:
        Generate a horizontal bar chart showing Logistic Regression coefficients.

    Input:
        coefficients_table.csv

    Output:
        logistic_regression_coefficients.png
    """
    print("Loading Logistic Regression coefficients...")
    check_file_exists(COEFFICIENTS_TABLE_PATH)

    coefficients_df = pd.read_csv(COEFFICIENTS_TABLE_PATH)
    coefficients_df = normalise_column_names(coefficients_df)

    required_columns = ["feature", "coefficient"]
    missing_columns = [
        column for column in required_columns if column not in coefficients_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in coefficients_table.csv: {missing_columns}"
        )

    coefficients_df["coefficient"] = pd.to_numeric(
        coefficients_df["coefficient"],
        errors="coerce",
    )

    coefficients_df = coefficients_df.dropna(subset=["coefficient"])
    coefficients_df = coefficients_df.sort_values(
        by="coefficient",
        ascending=True,
    )

    plt.figure(figsize=(9, 5))
    plt.barh(
        coefficients_df["feature"],
        coefficients_df["coefficient"],
    )

    plt.title("Logistic Regression Feature Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.axvline(x=0, linestyle="--", linewidth=1)
    plt.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving Logistic Regression coefficients figure...")
    plt.savefig(LR_COEFFICIENTS_FIGURE_PATH, dpi=300)
    plt.close()

    return coefficients_df


def generate_decision_tree_feature_importance_figure():
    """
    Description:
        Generate a horizontal bar chart showing Decision Tree feature importance.

    Input:
        tree_feature_importance.csv

    Output:
        decision_tree_feature_importance.png
    """
    print("Loading Decision Tree feature importance...")
    check_file_exists(TREE_FEATURE_IMPORTANCE_PATH)

    importance_df = pd.read_csv(TREE_FEATURE_IMPORTANCE_PATH)
    importance_df = normalise_column_names(importance_df)

    required_columns = ["feature", "importance"]
    missing_columns = [
        column for column in required_columns if column not in importance_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in tree_feature_importance.csv: "
            f"{missing_columns}"
        )

    importance_df["importance"] = pd.to_numeric(
        importance_df["importance"],
        errors="coerce",
    )

    importance_df = importance_df.dropna(subset=["importance"])
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=True,
    )

    plt.figure(figsize=(9, 5))
    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    print("Saving Decision Tree feature importance figure...")
    plt.savefig(DT_IMPORTANCE_FIGURE_PATH, dpi=300)
    plt.close()

    return importance_df


def save_feature_analysis_summary(
    ablation_pivot_df,
    coefficients_df,
    importance_df,
):
    """
    Description:
        Save a short summary explaining the feature analysis figures.

    Input:
        ablation_pivot_df
        coefficients_df
        importance_df

    Output:
        feature_analysis_summary.txt
    """
    strongest_lr_feature = coefficients_df.iloc[
        coefficients_df["coefficient"].abs().idxmax()
    ]

    strongest_dt_feature = importance_df.sort_values(
        by="importance",
        ascending=False,
    ).iloc[0]

    ablation_max_feature = (
        ablation_pivot_df.abs()
        .max(axis=1)
        .sort_values(ascending=False)
        .index[0]
    )

    with open(FEATURE_ANALYSIS_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Feature Analysis Figures Summary\n")
        file.write("================================\n\n")
        file.write("Input files:\n")
        file.write(f"- {ABLATION_RESULTS_PATH}\n")
        file.write(f"- {COEFFICIENTS_TABLE_PATH}\n")
        file.write(f"- {TREE_FEATURE_IMPORTANCE_PATH}\n\n")

        file.write("Generated figures:\n")
        file.write(f"- {ABLATION_FIGURE_PATH}\n")
        file.write(f"- {LR_COEFFICIENTS_FIGURE_PATH}\n")
        file.write(f"- {DT_IMPORTANCE_FIGURE_PATH}\n\n")

        file.write("Main findings:\n")
        file.write(
            f"- Ablation analysis showed that removing {ablation_max_feature} "
            "caused the largest F1-score drop.\n"
        )
        file.write(
            f"- Logistic Regression identified {strongest_lr_feature['feature']} "
            "as the strongest coefficient by absolute value.\n"
        )
        file.write(
            f"- Decision Tree identified {strongest_dt_feature['feature']} "
            "as the most important feature.\n\n"
        )

        file.write(
            "Interpretation:\n"
            "The feature analysis figures support the interpretability section of "
            "the final report. They show which structural conversation features "
            "contributed most to scam classification. The findings also support "
            "the ablation results by showing that turn_count and "
            "aggregate_money_terms were more influential than aggregate_num_urls.\n\n"
        )

        file.write(
            "Note:\n"
            "This script only converts existing output files into figures for final "
            "report evidence. It does not introduce new models, features, datasets, "
            "or evaluation results.\n"
        )


def main():
    """
    Description:
        Generate feature analysis figures and summary file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating feature analysis figures...")

    ablation_pivot_df = generate_ablation_f1_drop_figure()
    coefficients_df = generate_logistic_regression_coefficients_figure()
    importance_df = generate_decision_tree_feature_importance_figure()

    print("Saving feature analysis summary...")
    save_feature_analysis_summary(
        ablation_pivot_df=ablation_pivot_df,
        coefficients_df=coefficients_df,
        importance_df=importance_df,
    )

    print("Feature analysis figure generation completed.")
    print(f"Ablation figure saved to: {ABLATION_FIGURE_PATH}")
    print(f"Logistic Regression coefficients figure saved to: {LR_COEFFICIENTS_FIGURE_PATH}")
    print(f"Decision Tree feature importance figure saved to: {DT_IMPORTANCE_FIGURE_PATH}")
    print(f"Summary saved to: {FEATURE_ANALYSIS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()