# Comparative Evaluation of Additional Machine Learning Models for Scam Conversation Classification

## Project Overview

This repository is the COMP903 Information Technology Applied Research Project. It continues the COMP902 project [scam-conversation-risk-evaluator](https://github.com/jcmontano13/scam-conversation-risk-evaluator).

COMP902 built a reproducible and interpretable machine learning pipeline that classifies scam conversations from the BothBosu scam-dialogue dataset using four compact structural features. It compared Logistic Regression and Decision Tree, and found Logistic Regression to be the preferred model.

COMP903 keeps the same dataset, features, train/test split and evaluation process, and adds three machine learning models from different learning families. The aim is to find out whether the COMP902 results hold across more models, or whether they depend on the two models first chosen.

## Research Questions

| # | Research question | Main measure |
|---|---|---|
| RQ1 | Do Gaussian Naive Bayes, Random Forest or k-Nearest Neighbours classify scam conversations better than Logistic Regression using the four structural features? | F1-score |
| RQ2 | Do the additional models produce better-calibrated scam probabilities than Logistic Regression? | Brier score |
| RQ3 | Is the COMP902 feature ranking (turn_count first, aggregate_money_terms second) consistent across all five models? | Drop in F1 in ablation |

Each research question is tested with a null hypothesis and a significance test (alpha = 0.05).

## Models

| Model | Role | Learning family |
|---|---|---|
| Logistic Regression | COMP902 baseline and reference model | Linear |
| Decision Tree | COMP902 baseline | Rule-based |
| Gaussian Naive Bayes | New in COMP903 | Probabilistic |
| Random Forest | New in COMP903 | Ensemble |
| k-Nearest Neighbours | New in COMP903 | Instance-based |

The model list and evaluation settings are stored in `config.py`.

## Project Origin

The dataset, feature extraction, validation and train/test split were copied from COMP902 commit `a282ecf` and regenerated in this repository. The regenerated outputs were verified as identical to COMP902 (see step 6 of the execution order).

## Current Project Status

Progress is tracked as user stories on the [COMP903 project board](https://github.com/users/jcmontano13/projects/26).

| Sprint | Dates | Focus | Status |
|---|---|---|---|
| Sprint 0 | 13 Oct to 17 Oct 2026 | Repository setup, config, README, research proposal | Complete |
| Sprint 1 | 18 Oct to 23 Oct 2026 | Five-model baseline cross-validation and unit tests | In progress |
| Sprint 2 | 24 Oct to 30 Oct 2026 | Grid search, repeated cross-validation, Progress Report 1 | Pending |
| Sprint 3 | 31 Oct to 6 Nov 2026 | Significance tests, held-out test evaluation, calibration | Pending |
| Sprint 4 | 7 Nov to 13 Nov 2026 | Ablation, interpretability, Progress Report 2 | Pending |
| Sprint 5 | 14 Nov to 20 Nov 2026 | Figures, internal freeze checklist, README sync | Pending |
| Sprint 6 | 21 Nov to 27 Nov 2026 | Final report and submission | Pending |

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

## Project Structure

```text
<repository>/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── outputs/
│   └── reports/
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

## Setup

Create and activate a virtual environment:

```text
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```text
pip install -r requirements.txt
```

## Dataset Location

The raw dataset file is stored in:

```text
data/raw/scam-dialogue_all.csv
```

This file is the original BothBosu dataset and should not be edited.

## Execution Order

Steps 1 to 5 are the COMP902 data pipeline. Steps 6 and 7 confirm that this repository matches COMP902 and that the COMP903 settings are valid.

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

Runs stratified 5-fold cross-validation for all five models on the training set:

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

### Planned Steps

The following steps are added as their user stories are completed:

| Step | Script | User story sprint |
|---|---|---|
| Grid search for five models | `src/models/tune_five_models.py` | Sprint 2 |
| Repeated cross-validation | `src/models/repeated_cv_five_models.py` | Sprint 2 |
| Significance tests | `src/evaluation/significance_tests.py` | Sprint 3 |
| Held-out test evaluation | `src/evaluation/evaluate_test_set_five_models.py` | Sprint 3 |
| Calibration | `src/evaluation/calibration_five_models.py` | Sprint 3 |
| Ablation | `src/evaluation/ablation_five_models.py` | Sprint 4 |
| Interpretability | `src/evaluation/interpretability_five_models.py` | Sprint 4 |

## COMP902 Scripts

The original COMP902 modelling scripts (baseline cross-validation, grid search, held-out test evaluation, calibration, ablation, interpretability and figures) are kept unchanged in `src/` so the COMP902 results can still be reproduced. Their run commands are documented in the [COMP902 README](https://github.com/jcmontano13/scam-conversation-risk-evaluator).

## Generated Outputs

### Processed Data

```text
data/processed/scam_dialogue_parsed.csv
data/processed/scam_dialogue_thread_features.csv
data/processed/split_indices.json
data/processed/feature_extraction_metadata.json
```

### Validation Outputs

```text
outputs/reports/feature_validation_report.txt
outputs/reports/comp902_baseline_verification.md
```

### Documentation

```text
docs/feature_dictionary.md
README.md
```