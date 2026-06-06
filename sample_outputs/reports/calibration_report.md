# Calibration Report

## Raw vs Empirical-Bayes

| variant         |   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|:----------------|----------:|---------:|-----------:|----------:|----------:|
| raw             |  0.675095 | 0.159921 |   0.437834 | 0.0524543 | 0.0299055 |
| empirical_bayes |  0.734513 | 0.181747 |   0.201389 | 0.051994  | 0.0397222 |

## Worst Slice Summary

| worst_slice   |   worst_ece |   worst_slice_n |
|:--------------|------------:|----------------:|
| category=home |    0.106376 |              23 |
