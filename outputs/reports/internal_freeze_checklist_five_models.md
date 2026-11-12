# Internal Freeze Checklist (Five Models)

Generated: 2026-11-13 02:53:44

## Freeze Decision

**READY FOR FREEZE**

Automated checks passed: 44 of 44.

## Automated Checks

| Status | Section | Check | Evidence |
|---|---|---|---|
| PASS | Files: Data | PROCESSED_DATA | data/processed/scam_dialogue_thread_features.csv |
| PASS | Files: Data | SPLIT_INDICES | data/processed/split_indices.json |
| PASS | Files: Baseline and training (stories 2-6) | BASELINE_VERIFICATION_REPORT | outputs/reports/comp902_baseline_verification.md |
| PASS | Files: Baseline and training (stories 2-6) | UNIT_TEST_LOG_MODELS | outputs/reports/unit_test_log_models.txt |
| PASS | Files: Baseline and training (stories 2-6) | CV_RESULTS_FIVE_MODELS | outputs/reports/cv_results_five_models.csv |
| PASS | Files: Baseline and training (stories 2-6) | CV_SUMMARY_REPORT_FIVE_MODELS | outputs/reports/cv_summary_report_five_models.md |
| PASS | Files: Tuning and repeated CV (stories 7-10) | GRID_SEARCH_RESULTS_FIVE_MODELS | outputs/reports/grid_search_results_five_models.csv |
| PASS | Files: Tuning and repeated CV (stories 7-10) | SELECTED_HYPERPARAMS_FIVE_MODELS | outputs/reports/selected_hyperparams_five_models.json |
| PASS | Files: Tuning and repeated CV (stories 7-10) | REPEATED_CV_RESULTS | outputs/reports/repeated_cv_results.csv |
| PASS | Files: Tuning and repeated CV (stories 7-10) | SIGNIFICANCE_TESTS | outputs/reports/significance_tests.csv |
| PASS | Files: Tuning and repeated CV (stories 7-10) | SIGNIFICANCE_REPORT | outputs/reports/significance_report.md |
| PASS | Files: Held-out test (story 14) | TEST_RESULTS_FIVE_MODELS | outputs/reports/test_results_five_models.csv |
| PASS | Files: Held-out test (story 14) | TEST_PREDICTIONS_FIVE_MODELS | outputs/reports/test_predictions_five_models.csv |
| PASS | Files: Held-out test (story 14) | MCNEMAR_RESULTS | outputs/reports/mcnemar_results.csv |
| PASS | Files: Held-out test (story 14) | TEST_EVALUATION_REPORT_FIVE_MODELS | outputs/reports/test_evaluation_report_five_models.md |
| PASS | Files: Calibration (story 15) | CALIBRATION_RESULTS_FIVE_MODELS | outputs/reports/calibration_results_five_models.csv |
| PASS | Files: Calibration (story 15) | CALIBRATION_FOLD_RESULTS | outputs/reports/calibration_fold_results_five_models.csv |
| PASS | Files: Calibration (story 15) | RISK_BAND_CHECK | outputs/reports/risk_band_check_five_models.csv |
| PASS | Files: Calibration (story 15) | CALIBRATION_REPORT_FIVE_MODELS | outputs/reports/calibration_report_five_models.md |
| PASS | Files: Ablation and interpretability (stories 16-17) | ABLATION_RESULTS_FIVE_MODELS | outputs/reports/ablation_results_five_models.csv |
| PASS | Files: Ablation and interpretability (stories 16-17) | ABLATION_FOLD_RESULTS | outputs/reports/ablation_fold_results_five_models.csv |
| PASS | Files: Ablation and interpretability (stories 16-17) | ABLATION_REPORT_FIVE_MODELS | outputs/reports/ablation_report_five_models.md |
| PASS | Files: Ablation and interpretability (stories 16-17) | RF_FEATURE_IMPORTANCE | outputs/reports/rf_feature_importance.csv |
| PASS | Files: Ablation and interpretability (stories 16-17) | NB_CLASS_STATISTICS | outputs/reports/nb_class_statistics.csv |
| PASS | Files: Ablation and interpretability (stories 16-17) | KNN_NEIGHBOUR_EXAMPLES | outputs/reports/knn_neighbour_examples.csv |
| PASS | Files: Ablation and interpretability (stories 16-17) | INTERPRETABILITY_REPORT_FIVE_MODELS | outputs/reports/interpretability_report_five_models.md |
| PASS | Files: Extension: leave-one-type-out (story 18) | LOTO_RESULTS | outputs/reports/leave_one_type_out_results.csv |
| PASS | Files: Extension: leave-one-type-out (story 18) | LOTO_REPORT | outputs/reports/leave_one_type_out_report.md |
| PASS | Files: Documentation | CHALLENGES_LOG | docs/challenges_log.md |
| PASS | Consistency | Train/test split has no overlap | train 1280, test 320, total 1600, overlap 0 |
| PASS | Consistency | Test predictions are exactly the held-out test rows | 320 predicted conversations; 320 are test rows; 0 are training rows |
| PASS | Consistency | All models scored on the same test set size | Confusion-matrix totals: [320]; expected 320 for all 5 models |
| PASS | Consistency | Test metrics recompute from saved predictions | All 5 models x 7 values match |
| PASS | Consistency | Repeated CV has 50 paired folds per model | Folds per model: {'Logistic Regression': 50, 'Decision Tree': 50, 'Gaussian Naive Bayes': 50, 'Random Forest': 50, 'k-Nearest Neighbours': 50}; expected 50 |
| PASS | Consistency | Significance tests match the repeated-CV folds | All 6 mean differences match the folds |
| PASS | Consistency | Holm correction applied correctly (significance tests) | significance_tests.csv: 6 tests; Holm >= raw: True; 'significant' matches Holm p < 0.05: True |
| PASS | Consistency | Holm correction applied correctly (ablation) | ablation_results_five_models.csv: 20 tests; Holm >= raw: True; 'significant' matches Holm p < 0.05: True |
| PASS | Consistency | One calibration method selected per model, with lowest Brier | Logistic Regression = isotonic; Decision Tree = isotonic; Gaussian Naive Bayes = isotonic; Random Forest = isotonic; k-Nearest Neighbours = isotonic |
| PASS | Consistency | Ablation covers 5 models x 4 features | 20 rows (5 models x 4 features) |
| PASS | Consistency | Leave-one-type-out covers every scam type | Scam types ['refund', 'reward', 'ssn', 'support'] x 5 models present: True; gap formula consistent: True |
| PASS | Consistency | Selected hyperparameters saved for all 5 models | All 5 models present |
| PASS | Integrity | Repeated-CV seed differs from tuning seed | SEED = 42, REPEATED_CV_SEED = 43 |
| PASS | Integrity | Figures complete and up to date | 8 figures with summaries, all up to date |
| PASS | Integrity | Unit tests pass | 35 passed in 2.48s |

## Manual Checks

Tick each item by hand before the freeze tag is created.

- [ ] Every user story on the project board (except 22-24) is in Done
- [ ] All branches are merged into main and `git status` shows a clean working tree
- [ ] README lists every script, run command and output (story 22 will do the final sync)
- [ ] Challenges log reviewed: every entry is a real problem or decision, with its evidence
- [ ] Challenges log entry D4 says the extension was added after the significance tests (not planned in the proposal)
- [ ] The frozen numbers below match the numbers already quoted in Progress Reports 1 and 2
- [ ] Freeze tag created: `git tag freeze-v1` and `git push origin freeze-v1`

## Frozen Headline Numbers

These are the numbers the final report should quote. They are read from the checked files above.

| Model | Repeated-CV F1 | Test F1 | Test ROC-AUC | Test Brier | Missed scams | False alarms | Calibrated Brier (CV) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8889 | 0.8918 | 0.9594 | 0.0786 | 24 | 9 | 0.0827 (isotonic) |
| Decision Tree | 0.8880 | 0.8822 | 0.9415 | 0.0836 | 29 | 6 | 0.0802 (isotonic) |
| Gaussian Naive Bayes | 0.8809 | 0.8629 | 0.9503 | 0.0946 | 31 | 10 | 0.0861 (isotonic) |
| Random Forest | 0.8929 | 0.8792 | 0.9558 | 0.0762 | 29 | 7 | 0.0787 (isotonic) |
| k-Nearest Neighbours | 0.8913 | 0.8814 | 0.9488 | 0.0798 | 30 | 5 | 0.0774 (isotonic) |

| Metric | Model vs Logistic Regression | Mean difference | Holm p | Significant |
|---|---|---|---|---|
| f1 | Gaussian Naive Bayes | -0.0079 | 1.0000 | No |
| f1 | Random Forest | +0.0040 | 1.0000 | No |
| f1 | k-Nearest Neighbours | +0.0024 | 1.0000 | No |
| brier_score | Gaussian Naive Bayes | +0.0080 | 0.0540 | No |
| brier_score | Random Forest | -0.0057 | 0.0306 | Yes |
| brier_score | k-Nearest Neighbours | -0.0066 | 0.0050 | Yes |

## Freeze Rule

After the freeze tag, no script is re-run and no result file changes. If an error is found later, it is fixed on a new branch, this checklist is run again, and the change is recorded in the challenges log.
