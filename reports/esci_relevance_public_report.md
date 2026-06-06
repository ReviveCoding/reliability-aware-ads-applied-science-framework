# ESCI Product-Query Relevance Public-Data Validation Report

## Claim boundary

This run uses the local Amazon ESCI public dataset. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Rows: 120000
- Queries: 5273
- Label distribution: {'0': 20778, '1': 6313, '2': 42085, '3': 50824}

## Baseline vs proposed reranker

| Metric | TF-IDF baseline | Bias-corrected reranker |
|---|---:|---:|
| NDCG@10 | 0.753933 | 0.754143 |
| Recall@10 | 0.546836 | 0.542416 |
| Precision@10 | 0.861549 | 0.857472 |
| High-rank irrelevant rate | 0.140138 | 0.144368 |

## Release decision

- Status: **ITERATE**
- Checks: {'ndcg_improved': True, 'recall_not_worse': False, 'high_rank_irrelevant_not_worse': False}
- Reason: Reranker did not beat all promotion checks; keep as ITERATE.

## Interpretation

This validates the product-query relevance and reranking module on a real public ESCI dataset sample. The framework promotes the reranker only if it improves or preserves ranking quality and does not increase high-rank irrelevant products.
