# Criteo Uplift Public-Data Validation Report

## Claim boundary

This run uses the local Criteo Uplift public dataset sample. It does not use proprietary Amazon Ads data, real advertiser data, real customer data, production traffic, or production deployment.

## Data summary

- Rows loaded: 240000
- Test rows: 84000
- Feature count: 12
- Target column: conversion
- Treatment counts: {'0': 120000, '1': 120000}
- Overall target rate: 0.004167
- Target rate by treatment: {'0': 0.0033166666666666665, '1': 0.005016666666666667}

## Baseline vs uplift model

| Metric | pCVR baseline | T-learner uplift model |
|---|---:|---:|
| AUUC | 54.452305 | 31.323247 |
| Qini proxy | 24.452663 | 1.323604 |
| Uplift@20% | 0.006609 | 0.003468 |
| Negative uplift rate | ? | 0.827833 |

## Interpretation

This validates the uplift / incrementality module on a real public dataset. The comparison is between a vanilla pCVR targeting baseline and a T-learner uplift targeting policy.

## Next actions

- Add sample-size sensitivity.
- Add treatment/control balance diagnostics.
- Add X-learner or causal forest optional benchmark.
- Connect this report to the global release gate.
