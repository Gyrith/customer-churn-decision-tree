# Customer Churn Decision Tree
# Customer Churn Decision Tree

Decision tree model predicting telecom customer churn, tuned for recall since missing a churner costs far more than a false alarm.

## Files

- **`customer_churn_dt.ipynb`** — Baseline decision tree, `RandomizedSearchCV` broad tuning, focused `GridSearchCV` refinement, final test-set evaluation (recall, confusion matrix, feature importance).
- **`telecom_churn_processed.csv`** — Preprocessed telecom customer dataset (20,000 rows, 13 features + `churn` target).

## Approach

1. Baseline `DecisionTreeClassifier` evaluated via 5-fold cross-validated recall
2. `RandomizedSearchCV` over `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
3. Focused `GridSearchCV` around the best random-search parameters
4. Final model evaluated on a held-out test set (recall, confusion matrix, feature importance)