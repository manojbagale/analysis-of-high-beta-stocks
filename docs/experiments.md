# Experiment plan

## Question and outcome

Can pre-cutoff market information and timestamped news help identify next-session absolute price moves that exceed a stock-specific historical threshold? The main label is defined using the 90th percentile of training-period absolute next-session returns. Repeat with 85th and 95th percentiles as sensitivity checks when data coverage allows.

## Model comparisons

E1: Dummy prior, logistic regression, decision tree, random forest, RBF SVM and MLP using market-only features. Linear regression predicts the continuous next-session absolute return for an additional comparison.

E2: Run the same models, dates and labels with FinBERT headline probabilities, news intensity, sentiment disagreement and explicit missing coverage added.

E3: PCA and K-means are available for train-only regime exploration; automated regime-specific model evaluation is planned.

E4: Reliability work should add calibration diagrams, validation-only threshold selection, SHAP explanations and failure-case analysis. These are planned extensions rather than already tested results.

## Validation rules

Split by session date, not by random rows. Fit imputation, scaling, model parameters and stock-specific target thresholds on training rows only. Use a one-session gap before the test period. Never tune the classification threshold or model parameters against the final test block.

Primary metrics: average precision / PR-AUC, precision, recall, F1, balanced accuracy, ROC-AUC and Brier for probability-emitting estimators. Raw SVM decision scores can rank cases but are not calibrated probabilities. Regression uses MAE. Report fold variability and class prevalence, not only an average.

Preserve provider names, retrieval dates, source publication timestamps, listing coverage, package versions, random seeds, ticker histories, dataset freezes and fold boundaries for reproducibility. Do not commit proprietary raw news.
