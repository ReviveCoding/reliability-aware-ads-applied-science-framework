# Calibration Report

## Raw vs Empirical-Bayes

| variant         |   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|:----------------|----------:|---------:|-----------:|----------:|----------:|
| raw             |  0.591656 | 0.118254 |   0.546545 | 0.0569146 | 0.0467013 |
| empirical_bayes |  0.696587 | 0.142447 |   0.210862 | 0.0535939 | 0.0130419 |

## Worst Slice Summary

| worst_slice   |   worst_ece |   worst_slice_n |
|:--------------|------------:|----------------:|
| category=auto |    0.123389 |              15 |
