# Open Bandit OPE Public-Data Validation Report

## Claim boundary

This run uses a local Open Bandit public dataset sample. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Source file: `C:\Users\bjw-0\Downloads\Mini Project_Data\zr-obp-master_Open Bandit Dataset\obd\bts\all\all.csv`
- Rows: 10000
- Logged click rate: 0.004200

## OPE estimates

| Estimator | Value | CI low | CI high |
|---|---:|---:|---:|
| IPS | 0.002460 | 0.001036 | 0.004200 |
| SNIPS | 0.002435 | 0.001025 | 0.004158 |
| Doubly Robust | 0.002496 | 0.001125 | 0.004317 |

## Diagnostics

- Effective sample size: 342.56
- Max importance weight: 277.777778
- Mean importance weight: 1.010037
- Low candidate support rate: 0.000000

## Release decision

- Status: **ITERATE**
- Checks: {'dr_lower_bound_beats_logged_policy': False, 'effective_sample_size_ok': False, 'importance_weight_not_extreme': False}
- Reason: Candidate policy remains ITERATE because value or support diagnostics did not pass all promotion checks.
