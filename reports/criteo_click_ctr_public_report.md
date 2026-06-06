# Criteo Click Logs CTR/Calibration Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Rows: 250000
- Test rows: 75000
- Target column: `label`
- Positive rate: 0.031232
- Numeric features used: 13
- Categorical features used: 26
- Empirical-Bayes group column: `categorical_feature_1`

## Model comparison

| Metric | Logistic Regression | GBDT | GBDT + Empirical Bayes |
|---|---:|---:|---:|
| ROC-AUC | 0.7086663943114613 | 0.7002069626101098 | 0.6627800466601141 |
| PR-AUC | 0.08164290987648792 | 0.07315255700065285 | 0.06544415006672517 |
| Log loss | 0.130450 | 0.131486 | 0.133715 |
| Brier | 0.029596 | 0.029804 | 0.029866 |
| ECE | 0.001571 | 0.004141 | 0.000569 |

## Release decision

- Status: **ITERATE**
- Checks: {'gbdt_log_loss_not_worse_than_lr': True, 'gbdt_auc_not_worse_than_lr': False, 'eb_ece_improves_or_brier_stable': True}
- Reason: One or more CTR/calibration checks failed; keep as ITERATE.

## Interpretation

This validates the CTR prediction and probability calibration module on a real public Criteo click-log sample. The framework compares a linear CTR baseline, a nonlinear GBDT challenger, and empirical-Bayes probability smoothing under explicit release checks.
