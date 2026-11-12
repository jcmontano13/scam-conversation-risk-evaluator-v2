# Comparative Evaluation of Additional Machine Learning Models for Scam Conversation Classification

## Project Overview

This repository is the COMP903 Information Technology Applied Research Project. It continues the COMP902 project [scam-conversation-risk-evaluator](https://github.com/jcmontano13/scam-conversation-risk-evaluator).

COMP902 built a reproducible and interpretable machine learning pipeline. It classifies scam conversations from the BothBosu scam-dialogue dataset using four compact structural features. It compared Logistic Regression and Decision Tree, and found Logistic Regression to be the preferred model.

COMP903 keeps the same dataset, features, train/test split and evaluation process, and adds three models from different learning families. The aim is to find out whether the COMP902 results hold across more models, or whether they depend on the two models first chosen.

## Research Questions

| # | Research question | Main measure |
|---|---|---|
| RQ1 | Do Gaussian Naive Bayes, Random Forest or k-Nearest Neighbours classify scam conversations better than Logistic Regression using the four structural features? | F1-score |
| RQ2 | Do the additional models produce better-calibrated scam probabilities than Logistic Regression? | Brier score |
| RQ3 | Is the COMP902 feature ranking (turn_count first, aggregate_money_terms second) consistent across all five models? | Drop in F1 in ablation |
| Extension | Can each model detect a scam type it never saw in training? | Recall on the unseen type |

Each research question is tested with a null hypothesis and a significance test (alpha = 0.05). The answers are in the [Results Summary](#results-summary).

## Models

| Model | Role | Learning family |
|---|---|---|
| Logistic Regression | COMP902 baseline and reference model | Linear |
| Decision Tree | COMP902 baseline | Rule-based |
| Gaussian Naive Bayes | New in COMP903 | Probabilistic |
| Random Forest | New in COMP903 | Ensemble |
| k-Nearest Neighbours | New in COMP903 | Instance-based |

All five models are built in one place, `src/models/model_registry.py`. The model list and evaluation settings are stored in `config.py`.

## Project Origin

The dataset, feature extraction, validation and train/test split were copied from COMP902 commit `a282ecf` and regenerated in this repository. The regenerated outputs were verified as identical to COMP902 (step 6 of the execution order).

## Project Status

Progress was tracked as user stories on the [COMP903 project board](https://github.com/users/jcmontano13/projects/26).

| Sprint | Dates | Focus | Status |
|---|---|---|---|
| Sprint 0 | 13 Oct to 17 Oct 2026 | Repository setup, config, README, research proposal | Complete |
| Sprint 1 | 18 Oct to 23 Oct 2026 | Five-model baseline cross-validation and unit tests | Complete |
| Sprint 2 | 24 Oct to 30 Oct 2026 | Grid search, repeated cross-validation, Progress Report 1 | Complete |
| Sprint 3 | 31 Oct to 6 Nov 2026 | Significance tests, held-out test evaluation, calibration | Complete |
| Sprint 4 | 7 Nov to 13 Nov 2026 | Ablation, interpretability, leave-one-type-out extension, Progress Report 2 | Complete |
| Sprint 5 | 14 Nov to 20 Nov 2026 | Figures, internal freeze checklist, README sync | Complete |
| Sprint 6 | 21 Nov to 27 Nov 2026 | Final report and submission | Pending |

Problems, resolutions and design decisions are recorded in the [Challenges and Decisions Log](docs/challenges_log.md).

## Feature Set

The four engineered features from COMP902 are used unchanged:

- `turn_count`
- `aggregate_num_urls`
- `aggregate_money_terms`
- `ratio_request_messages`

The target label is:

- `thread_label` (0 = non-scam, 1 = scam)

Feature definitions are in `docs/feature_dictionary.md`.

## Excluded From Scope

The project intentionally excludes:

- New features, TF-IDF features or deep learning embeddings
- Deep learning, transformer models or LLM-based classification
- Support Vector Machines and Gradient Boosting
- New data collection or user studies
- Real-time detection or deployment

These are excluded to keep every model simple to rerun and interpret, and to keep results comparable with COMP902.

---

## Quick Start

To rerun the whole project from a fresh clone (Windows PowerShell):

```text
git clone https://github.com/jcmontano13/<repository>.git
git clone https://github.com/jcmontano13/scam-conversation-risk-evaluator.git
cd <repository>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src/feature_engineering/parse_and_features.py
python src/validation/check_features.py
python src/preprocessing/create_split_indices.py
python src/feature_engineering/generate_feature_metadata.py
python src/validation/verify_comp902_baseline.py --reference-dir "../scam-conversation-risk-evaluator"
python src/models/train_five_models.py
python src/evaluation/generate_cv_summary_report_five_models.py
python src/models/tune_five_models.py
python src/models/repeated_cv_five_models.py
python src/evaluation/significance_tests.py
python src/evaluation/evaluate_test_set_five_models.py
python src/evaluation/calibration_five_models.py
python src/evaluation/ablation_five_models.py
python src/evaluation/interpretability_five_models.py
python src/evaluation/leave_one_type_out.py
python src/reporting/generate_five_model_figures.py
python -m pytest tests -v
python src/validation/internal_freeze_check.py
python src/validation/check_readme.py
```

The second clone (COMP902) is needed only for step 6, which checks that this repository matches COMP902. On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The order matters, because each step reads files written by earlier steps. The run is successful when `internal_freeze_check.py` reports **READY FOR FREEZE** and `check_readme.py` reports **README is complete**.

## Setup

Requirements: Python 3.11 or later (check with `python --version`) and Git.

Create and activate a virtual environment:

```text
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```text
pip install -r requirements.txt
```

`requirements.txt` pins exact package versions, so the results are reproduced with the same library behaviour.

### Dataset

The raw dataset file is stored in:

```text
data/raw/scam-dialogue_all.csv
```

This file is the original BothBosu dataset and should not be edited. It contains 1,600 conversations: 800 scam and 800 non-scam. The scams are of four types: refund, reward, ssn and support.

## Reproducibility Settings

All settings are in `config.py`. Change nothing there to reproduce the reported results.

| Setting | Value | Purpose |
|---|---|---|
| `SEED` | 42 | Train/test split, model training and grid search folds (same as COMP902) |
| `REPEATED_CV_SEED` | 43 | Repeated cross-validation, so it never reuses the grid search folds (challenges log C1) |
| `CV_FOLDS` x `CV_REPEATS` | 5 x 10 | 50 paired folds for the significance tests |
| `PRIMARY_METRIC` | f1 | Grid search and model selection |
| `SIGNIFICANCE_ALPHA` | 0.05 | Significance level after Holm correction |
| `LOTO_CONTROL_REPEATS` | 10 | Random removals for the leave-one-type-out control |
| `COMP902_RISK_THRESHOLDS` | 0.26 / 0.54 | Medium / high risk bands carried over from COMP902 |

Rules that keep the results valid:

- The train/test split is fixed in `data/processed/split_indices.json`: 1,280 training and 320 test conversations.
- The held-out test set is used **once only**, in step 14. Calibration, ablation, interpretability and leave-one-type-out use the training set only.
- All five models are tuned with grids of similar size (20 to 24 combinations), so no model gets an unfair tuning advantage.

---

## Execution Order

Steps 1 to 5 are the COMP902 data pipeline. Steps 6 and 7 confirm that this repository matches COMP902 and that the COMP903 settings are valid. Steps 8 to 18 are the COMP903 experiments, and steps 19 to 21 produce the figures and final checks.

### 1. Feature Extraction

```text
python src/feature_engineering/parse_and_features.py
```

Expected outputs:

```text
data/processed/scam_dialogue_parsed.csv
data/processed/scam_dialogue_thread_features.csv
```

### 2. Feature Validation

```text
python src/validation/check_features.py
```

Expected outputs:

```text
outputs/reports/feature_validation_report.txt
```

### 3. Train / Test Split

```text
python src/preprocessing/create_split_indices.py
```

Expected outputs:

```text
data/processed/split_indices.json
```

### 4. Feature Metadata

```text
python src/feature_engineering/generate_feature_metadata.py
```

Expected outputs:

```text
data/processed/feature_extraction_metadata.json
```

### 5. Unit Checks

```text
pytest tests/test_feature_extraction.py
```

Expected outputs:

```text
Test results printed in the terminal
```

### 6. COMP902 Baseline Verification

Compares the regenerated parsed turns, features and split with the COMP902 repository:

```text
python src/validation/verify_comp902_baseline.py --reference-dir "<path to COMP902 repository>"
```

Expected outputs:

```text
outputs/reports/comp902_baseline_verification.md
```

### 7. Configuration Checks

Checks the COMP903 model list, settings and output paths in `config.py`:

```text
pytest tests/test_config.py
```

Expected outputs:

```text
Test results printed in the terminal
```

### 8. Five-Model Cross-Validation

Runs stratified 5-fold cross-validation for all five models on the training set. The models come from `src/models/model_registry.py`.

```text
python src/models/train_five_models.py
```

Expected outputs:

```text
outputs/reports/cv_results_five_models.csv
```

### 9. Five-Model Cross-Validation Summary Report

Generates a readable summary of the five-model cross-validation results:

```text
python src/evaluation/generate_cv_summary_report_five_models.py
```

Expected outputs:

```text
outputs/reports/cv_summary_report_five_models.md
```

### 10. Model Registry Unit Tests

Checks the five model definitions, scaling, seeds and settings in `src/models/model_registry.py`:

```text
pytest tests/test_model_registry.py -v
```

Expected outputs:

```text
outputs/reports/unit_test_log_models.txt
```

### 11. Grid Search for Five Models

Tunes each model on the training set using grids of similar size (20 to 24 combinations), selected by F1:

```text
python src/models/tune_five_models.py
```

Expected outputs:

```text
outputs/reports/grid_search_results_five_models.csv
outputs/reports/selected_hyperparams_five_models.json
```

### 12. Repeated Cross-Validation

Scores the five tuned models on the same 50 folds (10 repeats x 5 folds), using a separate seed from grid search:

```text
python src/models/repeated_cv_five_models.py
```

Expected outputs:

```text
outputs/reports/repeated_cv_results.csv
```

### 13. Significance Tests

Compares each new model with Logistic Regression on F1 and Brier score using the corrected resampled t-test and Holm correction:

```text
pytest tests/test_significance.py -v
python src/evaluation/significance_tests.py
```

Expected outputs:

```text
outputs/reports/significance_tests.csv
outputs/reports/significance_report.md
```

### 14. Held-Out Test Evaluation

Trains each tuned model on the full training set, then evaluates it once on the held-out test set, with McNemar's test against Logistic Regression:

```text
python src/evaluation/evaluate_test_set_five_models.py
```

Expected outputs:

```text
outputs/reports/test_results_five_models.csv
outputs/reports/mcnemar_results.csv
outputs/reports/test_predictions_five_models.csv
outputs/reports/test_evaluation_report_five_models.md
```

### 15. Calibration

Compares uncalibrated, sigmoid and isotonic probabilities for all five models on the 50 repeated cross-validation folds. It then tests the calibrated models against Logistic Regression and re-checks the COMP902 risk bands:

```text
python src/evaluation/calibration_five_models.py
```

Expected outputs:

```text
outputs/reports/calibration_fold_results_five_models.csv
outputs/reports/calibration_results_five_models.csv
outputs/reports/risk_band_check_five_models.csv
outputs/reports/calibration_report_five_models.md
```

### 16. Ablation for Five Models

Removes one feature at a time from each tuned model on the 50 repeated cross-validation folds, and checks whether the COMP902 feature ranking holds (RQ3):

```text
python src/evaluation/ablation_five_models.py
```

Expected outputs:

```text
outputs/reports/ablation_fold_results_five_models.csv
outputs/reports/ablation_results_five_models.csv
outputs/reports/ablation_report_five_models.md
```

### 17. Interpretability for the New Models

Produces Random Forest feature importance, Gaussian Naive Bayes class statistics and k-Nearest Neighbours worked examples:

```text
python src/evaluation/interpretability_five_models.py
```

Expected outputs:

```text
outputs/reports/rf_feature_importance.csv
outputs/reports/nb_class_statistics.csv
outputs/reports/knn_neighbour_examples.csv
outputs/reports/interpretability_report_five_models.md
```

### 18. Leave-One-Scam-Type-Out (Extension)

Tests whether each model can detect a scam type it never saw in training. A control removes the same number of scams at random from all types, so the type is still seen:

```text
python src/evaluation/leave_one_type_out.py
```

Expected outputs:

```text
outputs/reports/leave_one_type_out_results.csv
outputs/reports/leave_one_type_out_report.md
```

### 19. Figures for Five Models

Creates the figures for the final report and presentation from the saved results of steps 13 to 18. No model is retrained, so the figures always match the reported numbers. Each figure has a summary text file, and the curve figures also have a points CSV, following the COMP902 convention.

```text
python src/reporting/generate_five_model_figures.py
```

Expected outputs in `outputs/figures/`:

| Figure | Supports |
|---|---|
| model_comparison_significance.png | RQ1 and RQ2: difference from Logistic Regression with 95% CI |
| test_confusion_matrices.png | Held-out test: missed scams and false alarms |
| roc_curves_test_set.png / precision_recall_curves_test_set.png | Held-out test: ranking quality |
| reliability_curves_test_set.png | RQ2: probability reliability on unseen data |
| calibration_brier_comparison.png | RQ2: Brier score before and after calibration |
| ablation_f1_drop_heatmap.png | RQ3: feature reliance per model |
| generalisation_recall.png | Extension: recall on seen vs never-seen scam types |

### 20. Internal Freeze Checklist

Runs an automated freeze checklist before final report writing. Unlike the hand-written COMP902 checklist, every "complete" status is backed by a check:

- Completeness: every expected output file exists and is not empty
- Consistency:
  - test metrics recompute from the saved predictions
  - significance results match the repeated cross-validation folds
  - the Holm correction, calibration selection, ablation and leave-one-type-out results are complete and correct
- Integrity:
  - test rows never appear in training
  - the repeated cross-validation seed differs from the grid search seed
  - figures are newer than their inputs
  - unit tests pass

```text
python -m pytest tests -v
python src/validation/internal_freeze_check.py
```

Expected outputs:

```text
outputs/reports/internal_freeze_checklist_five_models.md
```

The checklist must report **READY FOR FREEZE**. It also records the frozen headline numbers for the final report and the manual checks. Freeze tag: `freeze-v1`.

### 21. README Check

Checks that this README still matches the repository:
- every command points to a real script
- every script and test file is documented
- every output file in `config.py` is listed
- the Quick Start runs in the same order as this section

```text
python src/validation/check_readme.py
```

Expected outputs:

```text
outputs/reports/readme_check.txt
```

---

## Results Summary

These are the numbers frozen at `freeze-v1`. The full tables are in `outputs/reports/internal_freeze_checklist_five_models.md`.

| Model | Repeated-CV F1 | Test F1 | Missed scams (of 160) | False alarms (of 160) | Calibrated Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.8889 | 0.8918 | 24 | 9 | 0.0827 |
| Decision Tree | 0.8880 | 0.8822 | 29 | 6 | 0.0802 |
| Gaussian Naive Bayes | 0.8809 | 0.8629 | 31 | 10 | 0.0861 |
| Random Forest | 0.8929 | 0.8792 | 29 | 7 | 0.0787 |
| k-Nearest Neighbours | 0.8907 | 0.8851 | 29 | 5 | 0.0774 |

- **RQ1:** No new model classifies significantly better than Logistic Regression. The F1 Holm p-value is 1.00 for all three, and the McNemar tests on the test set agree (Holm p 0.40 to 1.00).
- **RQ2:** Before calibration, Random Forest and k-NN have significantly lower Brier scores than Logistic Regression. After isotonic calibration, no difference is significant; the closest is k-NN, with Holm p = 0.075.
- **RQ3:** All five models rank the features as COMP902 did: turn_count, then aggregate_money_terms, then ratio_request_messages, then aggregate_num_urls.
- **Extension:** Tech-support scams are the hardest to detect when unseen (average recall 0.54). Naive Bayes loses the least recall on unseen types.

---

## Project Structure

```text
<repository>/
├── data/
│   ├── raw/                  scam-dialogue_all.csv
│   └── processed/            parsed turns, features, split indices, metadata
├── docs/                     feature dictionary, challenges and decisions log
├── outputs/
│   ├── reports/              CSV, JSON, TXT and MD results
│   └── figures/              PNG figures with summaries
├── src/
│   ├── evaluation/
│   ├── feature_engineering/
│   ├── models/
│   ├── preprocessing/
│   ├── reporting/
│   └── validation/
├── tests/
├── config.py
├── requirements.txt
└── README.md
```

## COMP902 Scripts

The original COMP902 modelling scripts are kept unchanged in `src/`, so the COMP902 results can still be reproduced. COMP903 does not need them, because the steps listed below replace them. Their run commands are documented in the [COMP902 README](https://github.com/jcmontano13/scam-conversation-risk-evaluator).

| COMP902 script | Purpose | Replaced in COMP903 by |
|---|---|---|
| `src/models/train_baseline_models.py` | Two-model cross-validation | Step 8 |
| `src/evaluation/generate_cv_summary_report.py` | Two-model cross-validation summary | Step 9 |
| `src/models/tune_baseline_models.py` | Two-model grid search | Step 11 |
| `src/evaluation/evaluate_test_set.py` | Two-model test evaluation (untuned models, decision D13) | Step 14 |
| `src/evaluation/calibration_experiments.py` | Logistic Regression calibration | Step 15 |
| `src/evaluation/ablation_test.py` | Two-model ablation | Step 16 |
| `src/evaluation/interpretability_artifacts.py` | Logistic Regression and Decision Tree interpretability | Step 17 |
| `src/reporting/generate_calibration_curve.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_confusion_matrix_heatmap.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_decision_tree_rule_diagram.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_feature_analysis_figures.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_precision_recall_curve.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_roc_curve.py` | COMP902 figure | Step 19 |
| `src/reporting/generate_threshold_tradeoff_curve.py` | COMP902 figure | Step 19 |

## Generated Outputs

### Processed Data

```text
data/processed/scam_dialogue_parsed.csv
data/processed/scam_dialogue_thread_features.csv
data/processed/split_indices.json
data/processed/feature_extraction_metadata.json
```

### COMP903 Outputs

Steps 6 to 21 above list each COMP903 output next to the command that creates it. They are all saved in `outputs/reports/` and `outputs/figures/`.

### COMP902 Outputs Kept for Reference

These are COMP902 results, written by the COMP902 scripts above. COMP903 does not read them.

```text
outputs/reports/cv_results.csv
outputs/reports/cv_summary_report.md
outputs/reports/cv_variance_report.md
outputs/reports/grid_search_results.csv
outputs/reports/selected_hyperparams.json
outputs/reports/test_results.csv
outputs/reports/test_evaluation_report.md
outputs/reports/calibration_results.csv
outputs/reports/calibration_report.md
outputs/reports/proposed_risk_thresholds.csv
outputs/reports/ablation_results.csv
outputs/reports/ablation_report.md
outputs/reports/coefficients_table.csv
outputs/reports/tree_feature_importance.csv
outputs/reports/decision_tree_rules.txt
outputs/reports/interpretability_report.md
```

### Documentation

```text
README.md
docs/feature_dictionary.md
docs/challenges_log.md
```