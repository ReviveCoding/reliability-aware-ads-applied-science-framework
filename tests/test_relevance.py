from src.relevance.synthetic_relevance import generate_synthetic_products
from src.relevance.bm25_tfidf import score_tfidf_relevance
from src.relevance.bias_corrected_reranker import add_candidate_expansion_scores, evaluate_relevance_by_query


def test_relevance_reranker_outputs_metrics():
    _, _, pairs = generate_synthetic_products(n_products=40, n_queries=8, seed=7)
    rel = add_candidate_expansion_scores(score_tfidf_relevance(pairs))
    metrics = evaluate_relevance_by_query(rel, "bias_corrected_rerank_score")
    assert {"ndcg_at_10", "recall_at_10", "precision_at_10"}.issubset(metrics.columns)
    assert len(metrics) > 0
