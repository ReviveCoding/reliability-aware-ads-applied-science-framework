# Open Bandit OPE Public-Data Validation Report

## Claim boundary

This corrected run uses a local Open Bandit public dataset sample. Historical artifacts are preserved. DR integrates all candidate actions and uses five-fold out-of-fold predictions; SNIPS recomputes its denominator in every bootstrap resample. Intervals are exploratory IID-row intervals, not dependence-robust or production qualification evidence. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Provenance

```json
{
  "estimator_version": "crossfit-action-integrated-v1",
  "run_utc": "2026-10-22T18:36:06.157861+00:00",
  "source_sha256": "0ad874e4dbf6902f0845dd478ad8dde5ef6903583d3ffaace78411bdad064106",
  "source_url": "https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv",
  "python": "3.12.14",
  "numpy": "2.5.4",
  "pandas": "3.1.0",
  "sklearn": "1.9.1",
  "crossfit_folds": 5,
  "seed": 42,
  "bootstrap_replicates": 300,
  "policy": "softmax over all explicitly mapped item affinity columns",
  "estimand": "per-position marginal reward, averaged over observed positions; not slate value",
  "ci_method": "IID row percentile bootstrap; SNIPS ratio recomputed; DR fixed out-of-fold scores",
  "limitations": "No nuisance refit in bootstrap; ignores temporal/user dependence; not production qualification"
}
```

## Data summary

- Source file: `data/open_bandit_verified/all.csv`
- Rows: 10000
- Logged click rate: 0.004200

## OPE estimates

| Estimator | Value | CI low | CI high |
|---|---:|---:|---:|
| IPS | 0.002460 | 0.001036 | 0.004200 |
| SNIPS | 0.002435 | 0.001039 | 0.004186 |
| Doubly Robust | 0.002755 | 0.000821 | 0.004795 |

## Diagnostics

- Effective sample size: 342.56
- Max importance weight: 277.777778
- Mean importance weight: 1.010037
- Low candidate support rate: 0.000000

## Release decision

- Status: **ITERATE**
- Checks: {'dr_lower_bound_beats_logged_policy': False, 'effective_sample_size_ok': False, 'importance_weight_not_extreme': False}
- Reason: Candidate policy remains ITERATE because value or support diagnostics did not pass all promotion checks.
