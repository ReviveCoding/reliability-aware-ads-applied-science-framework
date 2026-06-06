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
| lr      |  0.830594 | 0.187872 |   0.193053 | 0.053672  | 0.0301618 |
| gbdt    |  0.675095 | 0.159921 |   0.437834 | 0.0524543 | 0.0299055 |

## Calibration Metrics

| variant         |   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|:----------------|----------:|---------:|-----------:|----------:|----------:|
| raw             |  0.675095 | 0.159921 |   0.437834 | 0.0524543 | 0.0299055 |
| empirical_bayes |  0.734513 | 0.181747 |   0.201389 | 0.051994  | 0.0397222 |

## Residual Bias Correction

|   roc_auc |   pr_auc |   log_loss |     brier |       ece |
|----------:|---------:|-----------:|----------:|----------:|
|  0.854614 | 0.245736 |   0.188554 | 0.0518991 | 0.0316652 |

## Slice Reliability

| slice_col   | slice_value   |   n |   positive_rate |   log_loss |       ece |
|:------------|:--------------|----:|----------------:|-----------:|----------:|
| category    | auto          |  15 |       0.0666667 |  0.294906  | 0.10536   |
| category    | beauty        |  12 |       0         |  0.054023  | 0.052392  |
| category    | books         |  13 |       0.0769231 |  0.267458  | 0.0387071 |
| category    | fashion       |  15 |       0.0666667 |  0.182813  | 0.0905625 |
| category    | grocery       |  16 |       0.0625    |  0.259877  | 0.0102921 |
| category    | home          |  23 |       0.130435  |  0.31867   | 0.106376  |
| category    | sports        |  17 |       0         |  0.0458496 | 0.044756  |

## Policy Comparison

| policy            |   selected_impressions |   clicks |   conversions |   expected_clicks |   revenue |   spend_proxy |   iroas_proxy |   advertiser_concentration_top1 |   long_tail_exposure |   false_high_rank_irrelevant_rate |
|:------------------|-----------------------:|---------:|--------------:|------------------:|----------:|--------------:|--------------:|--------------------------------:|---------------------:|----------------------------------:|
| score_bid_only    |                     24 |        1 |             1 |           1.74022 |   12.5151 |       91.4227 |      0.136893 |                        0.208333 |             0.5      |                         0.416667  |
| score_raw         |                     24 |        4 |             1 |           3.16015 |   12.5151 |       59.6124 |      0.209942 |                        0.208333 |             0.458333 |                         0.166667  |
| score_eb          |                     24 |        3 |             1 |           2.77797 |   12.5151 |       77.2924 |      0.161919 |                        0.208333 |             0.291667 |                         0.291667  |
| score_reliability |                     24 |        4 |             1 |           2.97879 |   12.5151 |       68.8852 |      0.181681 |                        0.25     |             0.375    |                         0.0416667 |

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

| estimator     |      value |     ci_low |    ci_high |   ci_width |   effective_sample_size |   max_importance_weight |   mean_importance_weight |   low_support_rate |
|:--------------|-----------:|-----------:|-----------:|-----------:|------------------------:|------------------------:|-------------------------:|-------------------:|
| ips           |   0.310439 |   0.22801  |   0.401401 |   0.173391 |                 nan     |                nan      |                nan       |         nan        |
| snips         |   0.284737 |   0.209133 |   0.368169 |   0.159036 |                 nan     |                nan      |                nan       |         nan        |
| doubly_robust |   0.253764 |   0.184982 |   0.319449 |   0.134467 |                 nan     |                nan      |                nan       |         nan        |
| diagnostics   | nan        | nan        | nan        | nan        |                 160.958 |                 13.6636 |                  1.09027 |           0.751667 |

## Release Decision

Status: PASS

Reasons: ['All core calibration, slice reliability, relevance, and marketplace guardrails passed.']

Recommended actions: ['Proceed with broader sample/public-data validation.']
