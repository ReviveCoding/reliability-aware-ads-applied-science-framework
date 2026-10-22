# Relevance and Candidate Expansion Report

## BM25/TF-IDF vs Bias-Corrected Reranker

| variant                      |   ndcg_at_10 |   recall_at_10 |   precision_at_10 |
|:-----------------------------|-------------:|---------------:|------------------:|
| bm25_tfidf                   |     0.804492 |       0.59559  |           1       |
| candidate_expansion_reranker |     0.992584 |       0.568246 |           0.95625 |
