# Challenges and Decisions Log

This log records real problems, their causes and resolutions, and the decisions and findings that changed the project plan. Entries are added when they happen and are not edited afterwards, except to update a status or add the resolving commit.

It is the source for the "problems encountered" and "how problems were dealt with" sections of each progress report.

## How to Add an Entry

- Add a new row at the bottom of the right table, with the date it happened.
- **Challenges** are things that went wrong or did not work as expected.
- **Decisions and findings** are choices or results that changed the plan or design.
- Link the evidence: a commit, pull request, issue or output file.
- Set the status to `Open` until it is resolved, then `Resolved` with the date.

## Challenges

| ID | Date | Story | Problem | Cause | Resolution | Evidence | Status |
|---|---|---|---|---|---|---|---|

## Decisions and Findings

| ID | Date | Stage | Decision or finding | Reason | Effect on the project | Evidence |
|---|---|---|---|---|---|---|
| D1 | 2026-10-06 | Proposal | The BothBosu dataset was found to be synthetic, generated with Llama-3-70B. This was not noted in COMP902. | Found while reviewing the dataset card on Hugging Face. | Added as a confounding factor and limitation in the proposal; results are claimed for this dataset only. | [Dataset card](https://huggingface.co/datasets/BothBosu/scam-dialogue) |
| D2 | 2026-10-06 | Proposal | Scam conversations are much longer than non-scam conversations (mean turn_count 16.7 vs 10.7), and count features rise with length (correlation 0.44 between turn_count and aggregate_money_terms). | Exploratory check of the thread features file. | Conversation length listed as a confounding factor in the research design. | `data/processed/scam_dialogue_thread_features.csv` |
| D3 | 2026-10-06 | Proposal | Support Vector Machines and Gradient Boosting were excluded; Gaussian Naive Bayes, Random Forest and k-Nearest Neighbours were selected. | Keep every model simple to rerun and interpret; each new model represents a different learning family. | Model list fixed in `config.py`. | Proposal section 3.2 |
| D4 | 2026-10-06 | Proposal | The leave-one-scam-type-out generalisation test was kept out of the proposal and held as a possible extension. | Introduce it only if the first results show the models performing about the same. | Recorded as story 18 with the `extension` label. | Proposal section 3.4 |
| D5 | 2026-10-20 | Story 6 | With default settings, all five models scored within 0.032 F1 of each other (0.852 to 0.884), and Random Forest was within 0.001 of Logistic Regression. | Single 5-fold cross-validation, untuned models. | Early sign that the four features may limit performance; to be confirmed after tuning and significance tests. | `outputs/reports/cv_summary_report_five_models.md` |
| D6 | 2026-10-20 | Story 6 | k-Nearest Neighbours had the lowest F1 (0.852) but the highest recall (0.883) with its default of 5 neighbours. | Default setting, not yet tuned. | Number of neighbours to be tuned in story 10; result to be compared after tuning. | `outputs/reports/cv_results_five_models.csv` |
| D7 | 2026-10-20 | Story 10 | Grid sizes were made similar for all five models (20 to 24 combinations). COMP902 searched 20 combinations for Logistic Regression but 180 for Decision Tree. | A model searched more widely has more chances to score well, which is a confounding factor named in the proposal. | The Decision Tree grid was reduced and changed; it still includes the COMP902 selected settings. Tuned Logistic Regression reproduced the COMP902 selection (C=0.1, l1_ratio=1.0, F1 0.8885). | `src/models/tune_five_models.py`, `selected_hyperparams_five_models.json` |
| D8 | 2026-10-20 | Story 10 | After tuning, k-Nearest Neighbours improved the most (F1 0.852 to 0.895, using 21 neighbours), and all five models fell within 0.015 F1 of each other (0.882 to 0.896). | Tuning removed most of the gap seen with default settings. | Resolves D6. Strengthens the early sign in D5 that the features, not the models, limit performance. To be tested with repeated cross-validation and significance tests. | `selected_hyperparams_five_models.json` |