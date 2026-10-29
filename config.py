from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "scam-dialogue_all.csv"

SEED = 42

PARSED_DATA = PROJECT_ROOT / "data" / "processed" / "scam_dialogue_parsed.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "scam_dialogue_thread_features.csv"

VALIDATION_REPORT = PROJECT_ROOT / "outputs" / "reports" / "feature_validation_report.txt"

SPLIT_INDICES = PROJECT_ROOT / "data" / "processed" / "split_indices.json"

FEATURE_METADATA = PROJECT_ROOT / "data" / "processed" / "feature_extraction_metadata.json"

CV_RESULTS = PROJECT_ROOT / "outputs" / "reports" / "cv_results.csv"
CV_SUMMARY_REPORT = PROJECT_ROOT / "outputs" / "reports" / "cv_summary_report.md"

TEST_RESULTS = PROJECT_ROOT / "outputs" / "reports" / "test_results.csv"
TEST_EVALUATION_REPORT = PROJECT_ROOT / "outputs" / "reports" / "test_evaluation_report.md"

GRID_SEARCH_RESULTS = PROJECT_ROOT / "outputs" / "reports" / "grid_search_results.csv"
SELECTED_HYPERPARAMS = PROJECT_ROOT / "outputs" / "reports" / "selected_hyperparams.json"
CV_VARIANCE_REPORT = PROJECT_ROOT / "outputs" / "reports" / "cv_variance_report.md"

CALIBRATION_RESULTS = PROJECT_ROOT / "outputs" / "reports" / "calibration_results.csv"
CALIBRATION_REPORT = PROJECT_ROOT / "outputs" / "reports" / "calibration_report.md"
PROPOSED_RISK_THRESHOLDS = PROJECT_ROOT / "outputs" / "reports" / "proposed_risk_thresholds.csv"

ABLATION_RESULTS = PROJECT_ROOT / "outputs" / "reports" / "ablation_results.csv"
ABLATION_REPORT = PROJECT_ROOT / "outputs" / "reports" / "ablation_report.md"

COEFFICIENTS_TABLE = PROJECT_ROOT / "outputs" / "reports" / "coefficients_table.csv"
TREE_FEATURE_IMPORTANCE = PROJECT_ROOT / "outputs" / "reports" / "tree_feature_importance.csv"
TREE_RULES = PROJECT_ROOT / "outputs" / "reports" / "decision_tree_rules.txt"
INTERPRETABILITY_REPORT = PROJECT_ROOT / "outputs" / "reports" / "interpretability_report.md"

# COMP903: baseline verification against the frozen COMP902 pipeline
BASELINE_VERIFICATION_REPORT = PROJECT_ROOT / "outputs" / "reports" / "comp902_baseline_verification.md"



# =============================================================================
# COMP903 settings
# Everything below is new for COMP903. The COMP902 settings above are kept
# unchanged so the original scripts still run.
# =============================================================================

# Shared feature and target columns
FEATURE_COLUMNS = [
    "turn_count",
    "aggregate_num_urls",
    "aggregate_money_terms",
    "ratio_request_messages",
]
TARGET_COLUMN = "thread_label"

# Models compared in COMP903 (baselines first, then the new models)
BASELINE_MODELS = ["Logistic Regression", "Decision Tree"]
NEW_MODELS = ["Gaussian Naive Bayes", "Random Forest", "k-Nearest Neighbours"]
MODEL_NAMES = BASELINE_MODELS + NEW_MODELS
REFERENCE_MODEL = "Logistic Regression"  # every new model is compared with this one

# Evaluation settings
CV_FOLDS = 5
CV_REPEATS = 10              # repeated cross-validation: 10 repeats x 5 folds
PRIMARY_METRIC = "f1"        # used for tuning and for RQ1
SIGNIFICANCE_ALPHA = 0.05

# Output folders
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Five-model baseline cross-validation
CV_RESULTS_FIVE_MODELS = REPORTS_DIR / "cv_results_five_models.csv"
CV_SUMMARY_REPORT_FIVE_MODELS = REPORTS_DIR / "cv_summary_report_five_models.md"
UNIT_TEST_LOG_MODELS = REPORTS_DIR / "unit_test_log_models.txt"

# Tuning and repeated cross-validation
GRID_SEARCH_RESULTS_FIVE_MODELS = REPORTS_DIR / "grid_search_results_five_models.csv"
SELECTED_HYPERPARAMS_FIVE_MODELS = REPORTS_DIR / "selected_hyperparams_five_models.json"
REPEATED_CV_RESULTS = REPORTS_DIR / "repeated_cv_results.csv"

# Significance testing
SIGNIFICANCE_TESTS = REPORTS_DIR / "significance_tests.csv"
SIGNIFICANCE_REPORT = REPORTS_DIR / "significance_report.md"

# Held-out test evaluation
TEST_RESULTS_FIVE_MODELS = REPORTS_DIR / "test_results_five_models.csv"
MCNEMAR_RESULTS = REPORTS_DIR / "mcnemar_results.csv"
TEST_EVALUATION_REPORT_FIVE_MODELS = REPORTS_DIR / "test_evaluation_report_five_models.md"

# Calibration
CALIBRATION_RESULTS_FIVE_MODELS = REPORTS_DIR / "calibration_results_five_models.csv"
CALIBRATION_REPORT_FIVE_MODELS = REPORTS_DIR / "calibration_report_five_models.md"

# Ablation
ABLATION_RESULTS_FIVE_MODELS = REPORTS_DIR / "ablation_results_five_models.csv"
ABLATION_REPORT_FIVE_MODELS = REPORTS_DIR / "ablation_report_five_models.md"

# Interpretability for the new models
RF_FEATURE_IMPORTANCE = REPORTS_DIR / "rf_feature_importance.csv"
NB_CLASS_STATISTICS = REPORTS_DIR / "nb_class_statistics.csv"
KNN_NEIGHBOUR_EXAMPLES = REPORTS_DIR / "knn_neighbour_examples.csv"
INTERPRETABILITY_REPORT_FIVE_MODELS = REPORTS_DIR / "interpretability_report_five_models.md"

# Project documentation
CHALLENGES_LOG = DOCS_DIR / "challenges_log.md"


# Repeated cross-validation uses its own seed so its folds are different from
# the folds used for grid search (with SEED, its first repeat would reuse them).
REPEATED_CV_SEED = SEED + 1



# Held-out test predictions (one row per test conversation, all five models)
TEST_PREDICTIONS_FIVE_MODELS = REPORTS_DIR / "test_predictions_five_models.csv"