# ABC-style ESCI Reranker v2 Ablation Report

## Claim boundary

This run uses local public Amazon ESCI Shopping Queries data only. It does not use proprietary Amazon data, customer data, production traffic, or production deployment.

## Purpose

The prior ABC-style ESCI v2 run passed release gates, but the selected explicit overreach penalty was zero. This ablation checks which component actually drove the improvement:

- candidate expansion
- learned expected-gain scoring
- learned relevance scoring
- explicit overreach penalty
- combined expansion + expected-gain scoring
- forced overreach penalty variants

## Data summary

- Rows: 160000
- Queries: 7499
- Train / validation / test queries: 4499 / 1500 / 1500

## Baseline test metrics

- NDCG@10: 0.649766
- Recall@10: 0.572682
- Precision@10: 0.847089
- High-rank irrelevant rate: 0.152911

## Ablation comparison

| Variant | alpha expansion | beta expected gain | gamma relevance | delta overreach | NDCG@10 | Recall@10 | Precision@10 | High-rank irrelevant | Passes guardrails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| expansion_plus_expected_gain_forced_overreach_0_20 | 0.20 | 0.80 | 0.00 | 0.20 | 0.676193 | 0.573933 | 0.849422 | 0.150578 | True |
| expansion_plus_expected_gain | 0.20 | 0.80 | 0.00 | 0.00 | 0.676059 | 0.574212 | 0.850222 | 0.149778 | True |
| validation_selected_abc | 0.20 | 0.80 | 0.00 | 0.00 | 0.676059 | 0.574212 | 0.850222 | 0.149778 | True |
| expansion_plus_expected_gain_forced_overreach_0_50 | 0.20 | 0.80 | 0.00 | 0.50 | 0.675352 | 0.574882 | 0.850222 | 0.149778 | True |
| expected_gain_only | 0.00 | 0.80 | 0.00 | 0.00 | 0.675228 | 0.575819 | 0.850489 | 0.149511 | True |
| overreach_only | 0.00 | 0.00 | 0.00 | 0.35 | 0.657819 | 0.573512 | 0.847289 | 0.152711 | True |
| relevance_only | 0.00 | 0.00 | 0.20 | 0.00 | 0.656341 | 0.572582 | 0.847756 | 0.152244 | True |
| baseline_tfidf | 0.00 | 0.00 | 0.00 | 0.00 | 0.649766 | 0.572682 | 0.847089 | 0.152911 | True |
| expansion_only | 0.20 | 0.00 | 0.00 | 0.00 | 0.653443 | 0.569647 | 0.846622 | 0.153378 | False |

## Best variant by guardrail score

- Variant: **expansion_plus_expected_gain**
- Parameters: {'alpha_expansion': 0.2, 'beta_expected_gain': 0.8, 'gamma_relevance': 0.0, 'delta_overreach': 0.0}
- Test metrics: {'ndcg_at_10': 0.6760586452093351, 'recall_at_10': 0.5742123436684529, 'precision_at_10': 0.8502222222222223, 'high_rank_irrelevant_rate': 0.1497777777777778, 'query_count': 1500}

## Interpretation guide

If the best variants use beta expected gain and not delta overreach, then the main validated contribution is expected-gain reranking after expansion rather than explicit irrelevant penalty. If forced-overreach variants reduce high-rank irrelevant rate but hurt NDCG or recall, overreach control should remain a release guardrail rather than a promoted scoring term.
