"""
Generate Decision Tree Visualisation and Rule Diagram

Description:
    This script generates Decision Tree visual evidence for the final report.
    It recreates the existing Decision Tree model using the processed feature
    dataset, saved train/test split, and selected hyperparameters where available.

    The script outputs:
        - a full Decision Tree visualisation
        - a simplified Decision Tree rule diagram
        - a rule node summary CSV for review and Excel recreation
        - a text summary for the final report

    This script does not introduce a new dataset, new features, or a new model.
    It only recreates the existing Decision Tree setup for reporting evidence.

Input:
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - outputs/reports/selected_hyperparams.json, if available

Output:
    - outputs/figures/decision_tree_visualisation.png
    - outputs/figures/decision_tree_rule_diagram_depth3.png
    - outputs/figures/decision_tree_rule_nodes.csv
    - outputs/figures/decision_tree_rule_diagram_summary.txt
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# Allow script to import config.py from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import PROCESSED_DATA, SEED, SPLIT_INDICES  # noqa: E402


FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]

LABEL_COLUMN = "thread_label"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SELECTED_HYPERPARAMS_PATH = REPORT_DIR / "selected_hyperparams.json"

FULL_TREE_FIGURE_PATH = FIGURE_DIR / "decision_tree_visualisation.png"
RULE_DIAGRAM_FIGURE_PATH = FIGURE_DIR / "decision_tree_rule_diagram_depth3.png"
RULE_NODES_PATH = FIGURE_DIR / "decision_tree_rule_nodes.csv"
RULE_SUMMARY_PATH = FIGURE_DIR / "decision_tree_rule_diagram_summary.txt"


def load_split_indices():
    """
    Description:
        Load the saved train/test split indices.

    Input:
        split_indices.json

    Output:
        train_row_indices and test_row_indices lists
    """
    with open(SPLIT_INDICES, "r", encoding="utf-8") as file:
        split_payload = json.load(file)

    train_indices = split_payload.get("train_row_indices")
    test_indices = split_payload.get("test_row_indices")

    if train_indices is None or test_indices is None:
        raise KeyError(
            "Expected split_indices.json to contain "
            "'train_row_indices' and 'test_row_indices'."
        )

    return train_indices, test_indices


def load_selected_hyperparams():
    """
    Description:
        Load selected hyperparameters if the file exists.

    Output:
        Dictionary containing selected hyperparameters, or empty dictionary.
    """
    if not SELECTED_HYPERPARAMS_PATH.exists():
        print("selected_hyperparams.json not found. Using default tuned Decision Tree settings.")
        return {}

    with open(SELECTED_HYPERPARAMS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_decision_tree_params(payload):
    """
    Description:
        Extract Decision Tree parameters from selected_hyperparams.json
        in a flexible way.

    Input:
        payload: selected hyperparameter JSON payload

    Output:
        Dictionary of Decision Tree selected parameters
    """
    model_payload = payload.get("Decision Tree", {})

    if isinstance(model_payload, dict):
        return (
            model_payload.get("best_params")
            or model_payload.get("selected_params")
            or model_payload.get("params")
            or {}
        )

    return {}


def build_decision_tree_model():
    """
    Description:
        Build the existing Decision Tree model using selected hyperparameters
        where available.

    Output:
        DecisionTreeClassifier
    """
    selected_payload = load_selected_hyperparams()
    selected_params = get_decision_tree_params(selected_payload)

    # Fallback values use the selected tuned setup from the project.
    dt_params = {
        "criterion": "entropy",
        "max_depth": 5,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": None,
        "random_state": SEED,
    }

    allowed_params = {
        "criterion",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "class_weight",
    }

    for key, value in selected_params.items():
        cleaned_key = key.replace("model__", "")
        if cleaned_key in allowed_params:
            dt_params[cleaned_key] = value

    return DecisionTreeClassifier(**dt_params), dt_params


def validate_required_columns(feature_df):
    """
    Description:
        Validate that required feature and label columns exist.

    Input:
        feature_df: Processed feature DataFrame

    Output:
        Raises ValueError if required columns are missing
    """
    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        column for column in required_columns if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def train_decision_tree():
    """
    Description:
        Train the Decision Tree model on the existing training split.

    Output:
        trained_model, model_params, train_df, test_indices
    """
    print("Loading processed feature data...")
    feature_df = pd.read_csv(PROCESSED_DATA)

    validate_required_columns(feature_df)

    print("Loading saved train/test split...")
    train_indices, test_indices = load_split_indices()

    train_df = feature_df.iloc[train_indices].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN]

    model, model_params = build_decision_tree_model()

    print("Training Decision Tree model using training rows only...")
    model.fit(X_train, y_train)

    return model, model_params, train_df, test_indices


def generate_full_tree_visualisation(model):
    """
    Description:
        Generate full Decision Tree visualisation.

    Input:
        model: trained DecisionTreeClassifier

    Output:
        decision_tree_visualisation.png
    """
    print("Generating full Decision Tree visualisation...")

    plt.figure(figsize=(24, 14))

    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=["Non-scam", "Scam"],
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=False,
    )

    plt.title("Decision Tree Visualisation")
    plt.tight_layout()

    plt.savefig(FULL_TREE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def generate_simplified_rule_diagram(model):
    """
    Description:
        Generate simplified Decision Tree rule diagram using only the top
        levels of the tree.

    Input:
        model: trained DecisionTreeClassifier

    Output:
        decision_tree_rule_diagram_depth3.png
    """
    print("Generating simplified Decision Tree rule diagram...")

    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=["Non-scam", "Scam"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,
        proportion=False,
    )

    plt.title("Simplified Decision Tree Rule Diagram: Top-Level Rules")
    plt.tight_layout()

    plt.savefig(RULE_DIAGRAM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def extract_rule_nodes(model):
    """
    Description:
        Extract Decision Tree node details into a table for review and
        Excel recreation.

    Input:
        model: trained DecisionTreeClassifier

    Output:
        DataFrame of rule nodes
    """
    print("Extracting Decision Tree rule nodes...")

    tree = model.tree_
    records = []

    def walk_tree(node_id, depth, path_rule):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        is_leaf = left_child == right_child

        values = tree.value[node_id][0]
        total_value = values.sum()

        if total_value > 0:
            non_scam_probability = values[0] / total_value
            scam_probability = values[1] / total_value
        else:
            non_scam_probability = 0
            scam_probability = 0

        predicted_class = "Scam" if scam_probability >= non_scam_probability else "Non-scam"

        if is_leaf:
            feature_name = "Leaf"
            threshold = None
            rule_text = path_rule
        else:
            feature_index = tree.feature[node_id]
            feature_name = FEATURE_COLUMNS[feature_index]
            threshold = tree.threshold[node_id]
            rule_text = f"{feature_name} <= {threshold:.4f}"

        records.append(
            {
                "node_id": node_id,
                "depth": depth,
                "is_leaf": is_leaf,
                "feature": feature_name,
                "threshold": threshold,
                "samples": int(tree.n_node_samples[node_id]),
                "non_scam_probability": round(non_scam_probability, 6),
                "scam_probability": round(scam_probability, 6),
                "predicted_class": predicted_class,
                "path_rule": path_rule,
                "node_rule": rule_text,
                "left_child": None if is_leaf else int(left_child),
                "right_child": None if is_leaf else int(right_child),
            }
        )

        if not is_leaf:
            left_rule = f"{path_rule} AND {feature_name} <= {threshold:.4f}"
            right_rule = f"{path_rule} AND {feature_name} > {threshold:.4f}"

            if path_rule == "ROOT":
                left_rule = f"{feature_name} <= {threshold:.4f}"
                right_rule = f"{feature_name} > {threshold:.4f}"

            walk_tree(left_child, depth + 1, left_rule)
            walk_tree(right_child, depth + 1, right_rule)

    walk_tree(node_id=0, depth=0, path_rule="ROOT")

    rule_nodes_df = pd.DataFrame(records)
    rule_nodes_df.to_csv(RULE_NODES_PATH, index=False)

    return rule_nodes_df


def save_rule_diagram_summary(model, model_params, train_df, test_indices, rule_nodes_df):
    """
    Description:
        Save a summary explaining the Decision Tree rule diagram.

    Input:
        model: trained DecisionTreeClassifier
        model_params: Decision Tree parameters used
        train_df: Training DataFrame
        test_indices: Held-out test indices
        rule_nodes_df: Extracted rule node DataFrame

    Output:
        decision_tree_rule_diagram_summary.txt
    """
    text_rules = export_text(
        model,
        feature_names=FEATURE_COLUMNS,
        decimals=4,
    )

    root_node = rule_nodes_df[rule_nodes_df["node_id"] == 0].iloc[0]

    with open(RULE_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write("Decision Tree Visualisation and Rule Diagram Summary\n")
        file.write("====================================================\n\n")

        file.write("Generated files:\n")
        file.write(f"- {FULL_TREE_FIGURE_PATH}\n")
        file.write(f"- {RULE_DIAGRAM_FIGURE_PATH}\n")
        file.write(f"- {RULE_NODES_PATH}\n\n")

        file.write("Dataset and split:\n")
        file.write("- Dataset used: processed thread-level feature dataset\n")
        file.write(f"- Training rows used: {len(train_df)}\n")
        file.write(f"- Held-out test rows used for this figure: 0 of {len(test_indices)}\n\n")

        file.write("Features used:\n")
        for feature in FEATURE_COLUMNS:
            file.write(f"- {feature}\n")

        file.write("\nDecision Tree parameters:\n")
        for key, value in model_params.items():
            file.write(f"- {key}: {value}\n")

        file.write("\nTree structure:\n")
        file.write(f"- Total nodes: {model.tree_.node_count}\n")
        file.write(f"- Maximum depth: {model.tree_.max_depth}\n")
        file.write(
            f"- Root split: {root_node['feature']} <= "
            f"{root_node['threshold']:.4f}\n\n"
        )

        file.write("Text version of tree rules:\n")
        file.write(text_rules)
        file.write("\n\n")

        file.write("Interpretation:\n")
        file.write(
            "The Decision Tree rule diagram provides a visual explanation of how "
            "the tree model classified conversation threads. Each internal node "
            "shows a feature threshold used to split the data. Each leaf node shows "
            "the final predicted class. This supports the interpretability goal of "
            "the project because the decision process can be inspected as a set of "
            "explicit rules rather than as a black-box prediction.\n\n"
        )

        file.write("Note:\n")
        file.write(
            "This script is used for final report evidence only. It recreates the "
            "existing Decision Tree model using the saved training split and selected "
            "hyperparameters where available. It does not introduce a new dataset, "
            "new features, or a new model scope.\n"
        )


def main():
    """
    Description:
        Generate Decision Tree visualisation, simplified rule diagram,
        rule node CSV, and summary file.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Decision Tree visualisation and rule diagram...")

    model, model_params, train_df, test_indices = train_decision_tree()

    generate_full_tree_visualisation(model)
    generate_simplified_rule_diagram(model)
    rule_nodes_df = extract_rule_nodes(model)

    print("Saving Decision Tree rule diagram summary...")
    save_rule_diagram_summary(
        model=model,
        model_params=model_params,
        train_df=train_df,
        test_indices=test_indices,
        rule_nodes_df=rule_nodes_df,
    )

    print("Decision Tree visualisation generation completed.")
    print(f"Full tree figure saved to: {FULL_TREE_FIGURE_PATH}")
    print(f"Simplified rule diagram saved to: {RULE_DIAGRAM_FIGURE_PATH}")
    print(f"Rule nodes saved to: {RULE_NODES_PATH}")
    print(f"Summary saved to: {RULE_SUMMARY_PATH}")


if __name__ == "__main__":
    main()