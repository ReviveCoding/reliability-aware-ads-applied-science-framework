from src.relevance.synthetic_relevance import generate_synthetic_products
from src.relevance.bm25_tfidf import score_tfidf_relevance
from src.relevance.bias_corrected_reranker import add_candidate_expansion_scores, evaluate_relevance_by_query

def run_relevance_track(config):
    products, queries, pairs = generate_synthetic_products(n_products=int(config.get("n_products", 90)), n_queries=int(config.get("n_queries", 36)), seed=int(config.get("random_seed", 42)))
    rel = add_candidate_expansion_scores(score_tfidf_relevance(pairs))
    rel_base = evaluate_relevance_by_query(rel, "bm25_tfidf_score"); rel_prop = evaluate_relevance_by_query(rel, "bias_corrected_rerank_score")
    metrics = {"bm25_tfidf": rel_base[["ndcg_at_10","recall_at_10","precision_at_10"]].mean().to_dict(), "candidate_expansion_reranker": rel_prop[["ndcg_at_10","recall_at_10","precision_at_10"]].mean().to_dict()}
    return {"pairs": rel, "base_query_metrics": rel_base, "proposed_query_metrics": rel_prop, "metrics": metrics}
