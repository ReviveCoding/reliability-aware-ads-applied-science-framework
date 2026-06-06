from src.common.metrics import ndcg_at_k, recall_at_k, precision_at_k

def add_candidate_expansion_scores(df):
    out = df.copy()
    category_match = (out["query_category"] == out["product_category"]).astype(float)
    modifier_match = (out["query_modifier"] == out["modifier"]).astype(float)
    long_tail = out["is_long_tail"].astype(float)
    out["candidate_expanded_score"] = 0.55*out["bm25_tfidf_score"] + 0.20*category_match + 0.10*modifier_match + 0.15*long_tail
    hard_negative_penalty = ((category_match == 0) & (modifier_match == 1)).astype(float)
    out["bias_corrected_rerank_score"] = 0.50*out["candidate_expanded_score"] + 0.28*category_match + 0.17*modifier_match - 0.18*hard_negative_penalty + 0.05*long_tail
    return out

def evaluate_relevance_by_query(df, score_col, k=10):
    rows = []
    for qid, g in df.groupby("query_id"):
        labels = g["esci_label_num"].values; scores = g[score_col].values
        rows.append({"query_id": qid, "ndcg_at_10": ndcg_at_k(labels, scores, k=k), "recall_at_10": recall_at_k(labels, scores, k=k), "precision_at_10": precision_at_k(labels, scores, k=k)})
    import pandas as pd
    return pd.DataFrame(rows)
