# Five-Model Cross-Validation Summary Report

## Description

This report summarises the baseline cross-validation results for the COMP903 comparison of five machine learning models for scam conversation classification.

The models compared are:

- Logistic Regression (reference model)
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- k-Nearest Neighbours

The models use only the four structural thread-level features:

- `turn_count`
- `aggregate_num_urls`
- `aggregate_money_terms`
- `ratio_request_messages`

## Evaluation Method

Stratified 5-fold cross-validation was performed on the training split only, using the same folds for every model. All models used their default settings from `model_registry.py`.

The held-out test set was not used in this stage. It remains reserved for final model evaluation.

Scores are shown as the mean across folds with the standard deviation in brackets.

## Results (ranked by F1)

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.8875 (+/- 0.0155) | 0.9097 (+/- 0.0153) | 0.8609 (+/- 0.0347) | 0.8841 (+/- 0.0177) | 0.9478 (+/- 0.0140) |
| 2 | Random Forest | 0.8867 (+/- 0.0142) | 0.9107 (+/- 0.0183) | 0.8578 (+/- 0.0159) | 0.8834 (+/- 0.0145) | 0.9473 (+/- 0.0182) |
| 3 | Gaussian Naive Bayes | 0.8867 (+/- 0.0243) | 0.9457 (+/- 0.0185) | 0.8203 (+/- 0.0360) | 0.8783 (+/- 0.0272) | 0.9444 (+/- 0.0100) |
| 4 | Decision Tree | 0.8703 (+/- 0.0114) | 0.9102 (+/- 0.0147) | 0.8219 (+/- 0.0167) | 0.8637 (+/- 0.0124) | 0.9055 (+/- 0.0178) |
| 5 | k-Nearest Neighbours | 0.8500 (+/- 0.0269) | 0.8277 (+/- 0.0334) | 0.8859 (+/- 0.0330) | 0.8553 (+/- 0.0256) | 0.9323 (+/- 0.0163) |

## Best Model by Metric

- Best model by Accuracy: `Logistic Regression` (0.8875)
- Best model by Precision: `Gaussian Naive Bayes` (0.9457)
- Best model by Recall: `k-Nearest Neighbours` (0.8859)
- Best model by F1: `Logistic Regression` (0.8841)
- Best model by ROC-AUC: `Logistic Regression` (0.9478)

## Comparison with Logistic Regression

Differences are each model's mean score minus the Logistic Regression mean score. A positive value means the model scored higher.

| Model | F1 difference | Precision difference | Recall difference | F1 gap compared with Logistic Regression spread |
|---|---:|---:|---:|---|
| Decision Tree | -0.0205 | +0.0005 | -0.0391 | larger than one standard deviation |
| Gaussian Naive Bayes | -0.0058 | +0.0360 | -0.0406 | within one standard deviation |
| Random Forest | -0.0008 | +0.0010 | -0.0031 | within one standard deviation |
| k-Nearest Neighbours | -0.0289 | -0.0820 | +0.0250 | larger than one standard deviation |

## Interpretation

The highest mean F1-score was achieved by Logistic Regression (0.8841). Across all five models, mean F1 ranged by 0.0289, compared with a fold-to-fold standard deviation of 0.0177 for Logistic Regression.

The best of the new models by F1 was Random Forest (0.8834).

The models made different trade-offs between precision and recall. Gaussian Naive Bayes had the highest precision (0.9457), meaning its scam predictions were most often correct, while k-Nearest Neighbours had the highest recall (0.8859), meaning it missed the fewest scam conversations.

These results use default model settings and a single 5-fold cross-validation. They describe the baseline comparison only; tuning, repeated cross-validation and significance tests are needed before concluding that one model is better than another.

## Reproducibility Note

These results were generated from `cv_results_five_models.csv`, which was created by `train_five_models.py`.

Generated at: 2026-10-20T02:23:51
