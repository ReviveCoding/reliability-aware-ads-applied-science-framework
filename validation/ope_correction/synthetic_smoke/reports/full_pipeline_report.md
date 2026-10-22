# Full Pipeline Report

## Claim Boundary

Synthetic fallback data were used. No proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment is claimed.

## Schema Validation

| status   |   row_count |   column_count | missing_columns   | problems   |
|:---------|------------:|---------------:|:------------------|:-----------|
| PASS     |         600 |             16 | []                | []         |

## CTR Metrics

| model   |   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|:--------|----------:|---------:|-----------:|----------:|----------:|
| lr      |  0.826802 | 0.174941 |   0.19575  | 0.0556538 | 0.0277058 |
| gbdt    |  0.591656 | 0.118254 |   0.546545 | 0.0569146 | 0.0467013 |

## Calibration Metrics

| variant         |   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|:----------------|----------:|---------:|-----------:|----------:|----------:|
| raw             |  0.591656 | 0.118254 |   0.546545 | 0.0569146 | 0.0467013 |
| empirical_bayes |  0.696587 | 0.142447 |   0.210862 | 0.0535939 | 0.0130419 |

## Residual Bias Correction

|   roc_auc |   pr_auc |   log_loss |     brier |      ece |
|----------:|---------:|-----------:|----------:|---------:|
|  0.874842 | 0.249344 |   0.177542 | 0.0500857 | 0.040342 |

## Slice Reliability

| slice_col   | slice_value   |   n |   positive_rate |   log_loss |        ece |
|:------------|:--------------|----:|----------------:|-----------:|-----------:|
| category    | auto          |  15 |       0.0666667 |  0.29947   | 0.123389   |
| category    | beauty        |  12 |       0         |  0.0522727 | 0.0507705  |
| category    | books         |  13 |       0.0769231 |  0.269101  | 0.0371234  |
| category    | fashion       |  15 |       0.0666667 |  0.234373  | 0.00841406 |
| category    | grocery       |  17 |       0.0588235 |  0.251959  | 0.0026787  |
| category    | home          |  23 |       0.130435  |  0.321864  | 0.108481   |
| category    | sports        |  17 |       0         |  0.0553689 | 0.0536828  |

## Policy Comparison

| policy            |   selected_impressions |   clicks |   conversions |   expected_clicks |   revenue |   spend_proxy |   iroas_proxy |   advertiser_concentration_top1 |   long_tail_exposure |   false_high_rank_irrelevant_rate |
|:------------------|-----------------------:|---------:|--------------:|------------------:|----------:|--------------:|--------------:|--------------------------------:|---------------------:|----------------------------------:|
| score_bid_only    |                     24 |        1 |             1 |           2.04624 |   12.5151 |       91.4227 |      0.136893 |                        0.208333 |             0.5      |                          0.416667 |
| score_raw         |                     24 |        3 |             1 |           3.14401 |   12.5151 |       61.0831 |      0.204887 |                        0.208333 |             0.458333 |                          0.208333 |
| score_eb          |                     24 |        2 |             1 |           2.62448 |   12.5151 |       85.6401 |      0.146136 |                        0.208333 |             0.375    |                          0.333333 |
| score_reliability |                     24 |        4 |             1 |           2.86677 |   12.5151 |       70.9373 |      0.176425 |                        0.25     |             0.375    |                          0.125    |

## Relevance Metrics

| variant                      |   ndcg_at_10 |   recall_at_10 |   precision_at_10 |
|:-----------------------------|-------------:|---------------:|------------------:|
| bm25_tfidf                   |     0.804492 |       0.59559  |           1       |
| candidate_expansion_reranker |     0.992584 |       0.568246 |           0.95625 |

## Uplift Metrics

|     auuc |   qini_proxy |   uplift_at_20pct |   negative_uplift_rate |
|---------:|-------------:|------------------:|-----------------------:|
| 0.105972 |     0.545421 |         0.0181818 |               0.266667 |

## OPE Metrics

| estimator     |      value |     ci_low |    ci_high |   ci_width |   effective_sample_size |   max_importance_weight |   mean_importance_weight |   low_support_rate | method                                                     |   n_boot |   n_splits |   seed | limitation                                                                     |
|:--------------|-----------:|-----------:|-----------:|-----------:|------------------------:|------------------------:|-------------------------:|-------------------:|:-----------------------------------------------------------|---------:|-----------:|-------:|:-------------------------------------------------------------------------------|
| ips           |   0.310439 |   0.225744 |   0.394348 |   0.168604 |                 nan     |                nan      |                nan       |         nan        | nan                                                        |      nan |        nan |    nan | nan                                                                            |
| snips         |   0.284737 |   0.215203 |   0.351886 |   0.136683 |                 nan     |                nan      |                nan       |         nan        | nan                                                        |      nan |        nan |    nan | nan                                                                            |
| doubly_robust |   0.283626 |   0.20322  |   0.361868 |   0.158647 |                 nan     |                nan      |                nan       |         nan        | nan                                                        |      nan |        nan |    nan | nan                                                                            |
| diagnostics   | nan        | nan        | nan        | nan        |                 160.958 |                 13.6636 |                  1.09027 |           0.751667 | nan                                                        |      nan |        nan |    nan | nan                                                                            |
| inference     | nan        | nan        | nan        | nan        |                 nan     |                nan      |                nan       |         nan        | iid row percentile bootstrap; DR fixed cross-fitted scores |      300 |          5 |     42 | Not cluster/time-aware; no nuisance refitting or production coverage guarantee |

## Release Decision

Status: PASS

Reasons: ['All core calibration, slice reliability, relevance, and marketplace guardrails passed.']

Recommended actions: ['Proceed with broader sample/public-data validation.']
