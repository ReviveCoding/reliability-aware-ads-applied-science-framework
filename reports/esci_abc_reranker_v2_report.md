# ABC-style ESCI Reranker v2 Public-Data Validation Report

## Claim boundary

This run uses local public Amazon ESCI Shopping Queries data only. It does not use proprietary Amazon data, customer data, production traffic, or production deployment.

## Method

This experiment adapts the ABC idea of candidate expansion plus learned overreach correction to product-query relevance ranking.

- Baseline candidate: title-heavy TF-IDF relevance score.
- Expansion signal: description, bullet, all-text, character n-gram, coverage, and brand evidence.
- Bias/overreach correction: learned irrelevant-risk probability plus lexical overreach risk.
- Selection: validation query-level grid search under NDCG, recall, and high-rank irrelevant guardrails.

## Data summary

- Rows: 160000
- Queries: 7499
- Locale: `us`
- Train / validation / test queries: 4499 / 1500 / 1500

## Selected parameters

- alpha expansion: 0.2
- beta expected gain: 0.8
- gamma relevance: 0.0
- delta overreach: 0.0

## Test comparison

| Metric | TF-IDF baseline | ABC-style reranker |
|---|---:|---:|
| NDCG@10 | 0.649766 | 0.676059 |
| Recall@10 | 0.572682 | 0.574212 |
| Precision@10 | 0.847089 | 0.850222 |
| High-rank irrelevant rate | 0.152911 | 0.149778 |

## Release decision

- Status: **PASS**
- Checks: {'ndcg_not_worse_than_baseline': True, 'recall_not_worse_than_baseline': True, 'high_rank_irrelevant_not_worse': True, 'abc_has_expansion_or_overreach_control': True}
- Reason: ABC-style reranker passed NDCG, recall, and high-rank irrelevant guardrails.

## Interpretation

The release gate requires that expansion improves or preserves relevant coverage without creating high-rank irrelevant overreach. This mirrors the original ABC logic: expansion is useful only when paired with correction that controls false positive extension.
