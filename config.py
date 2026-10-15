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