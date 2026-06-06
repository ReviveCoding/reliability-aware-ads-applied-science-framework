
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


LABEL_MAP = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0,
    "Exact": 3,
    "Substitute": 2,
    "Complement": 1,
    "Irrelevant": 0,
    "exact": 3,
    "substitute": 2,
    "complement": 1,
    "irrelevant": 0,
}


def find_file(root, keyword):
    root = Path(root)
    matches = list(root.rglob(f"*{keyword}*.parquet")) + list(root.rglob(f"*{keyword}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No file matching '*{keyword}*' under {root}")
    return matches[0]


def dcg(labels):
    labels = np.asarray(labels, dtype=float)
    gains = 2 ** labels - 1
    discounts = 1 / np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(labels, scores, k=10):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1][:k]
    ranked = labels[order]
    ideal = np.sort(labels)[::-1][:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(ranked) / denom


def recall_at_k(labels, scores, k=10):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    total_relevant = np.sum(labels > 0)
    if total_relevant == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float(np.sum(labels[order] > 0) / total_relevant)


def precision_at_k(labels, scores, k=10):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(labels[order] > 0))


def token_jaccard(a, b):
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_esci_sample(data_dir, max_rows=120000):
    examples_path = find_file(data_dir, "examples")
    products_path = find_file(data_dir, "products")

    print("Examples file:", examples_path)
    print("Products file:", products_path)

    examples = load_table(examples_path)
    products = load_table(products_path)

    print("Examples columns:", examples.columns.tolist())
    print("Products columns:", products.columns.tolist())

    if "product_id" not in examples.columns or "product_id" not in products.columns:
        raise ValueError("Expected product_id in both examples and products.")

    query_col = "query" if "query" in examples.columns else "query_text"
    label_col = "esci_label" if "esci_label" in examples.columns else "label"

    if query_col not in examples.columns:
        raise ValueError(f"Could not find query column. Examples columns: {examples.columns.tolist()}")

    if label_col not in examples.columns:
        raise ValueError(f"Could not find ESCI label column. Examples columns: {examples.columns.tolist()}")

    if "product_title" not in products.columns:
        title_candidates = [c for c in products.columns if "title" in c.lower()]
        if not title_candidates:
            raise ValueError(f"Could not find product title column. Products columns: {products.columns.tolist()}")
        products = products.rename(columns={title_candidates[0]: "product_title"})

    keep_products = ["product_id", "product_title"]
    for c in [
        "product_brand",
        "product_color",
        "product_locale",
        "product_description",
        "product_bullet_point",
    ]:
        if c in products.columns:
            keep_products.append(c)

    df = examples.merge(products[keep_products], on="product_id", how="left")
    df = df.dropna(subset=[query_col, "product_title", label_col]).copy()

    if "small_version" in df.columns:
        small = df[df["small_version"].astype(str).str.lower().isin(["1", "true"])]
        if len(small) >= 1000:
            df = small.copy()

    if "query_id" not in df.columns:
        df["query_id"] = df[query_col].astype(str)

    if len(df) > max_rows:
        qids = pd.Series(df["query_id"].drop_duplicates())
        keep_n = max(1000, min(len(qids), max_rows // 20))
        keep_qids = qids.sample(keep_n, random_state=42)
        df = df[df["query_id"].isin(keep_qids)].head(max_rows).copy()

    df["query_text"] = df[query_col].astype(str)
    df["title_text"] = df["product_title"].astype(str)

    df["label_num"] = df[label_col].map(LABEL_MAP)
    df["label_num"] = pd.to_numeric(df["label_num"], errors="coerce").fillna(0).astype(int)

    if "product_brand" in df.columns:
        df["brand_text"] = df["product_brand"].fillna("").astype(str)
    else:
        df["brand_text"] = ""

    if "product_description" in df.columns:
        df["description_text"] = df["product_description"].fillna("").astype(str)
    elif "product_bullet_point" in df.columns:
        df["description_text"] = df["product_bullet_point"].fillna("").astype(str)
    else:
        df["description_text"] = ""

    return df


def score_tfidf(df):
    corpus = pd.concat([df["query_text"], df["title_text"]]).astype(str)
    vec = TfidfVectorizer(min_df=2, max_features=80000, ngram_range=(1, 2))
    vec.fit(corpus)
    qv = vec.transform(df["query_text"])
    tv = vec.transform(df["title_text"])
    return np.asarray(qv.multiply(tv).sum(axis=1)).reshape(-1)


def evaluate_by_query(df, score_col):
    rows = []
    for qid, g in df.groupby("query_id"):
        if len(g) < 3:
            continue
        labels = g["label_num"].values
        scores = g[score_col].values
        top = g.sort_values(score_col, ascending=False).head(10)
        rows.append({
            "query_id": str(qid),
            "n": int(len(g)),
            "ndcg_at_10": ndcg_at_k(labels, scores, 10),
            "recall_at_10": recall_at_k(labels, scores, 10),
            "precision_at_10": precision_at_k(labels, scores, 10),
            "high_rank_irrelevant_rate": float((top["label_num"] == 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    data_dir = os.environ.get("ESCI_DATA_DIR")
    if not data_dir:
        raise SystemExit("Set ESCI_DATA_DIR first.")

    print("Loading ESCI from:", data_dir)
    df = load_esci_sample(data_dir)

    print("Loaded:", df.shape)
    print("Queries:", df["query_id"].nunique())
    print("Label distribution:")
    print(df["label_num"].value_counts().sort_index())

    df["tfidf_score"] = score_tfidf(df)
    df["title_jaccard"] = [token_jaccard(q, t) for q, t in zip(df["query_text"], df["title_text"])]
    df["brand_match"] = [
        1.0 if b and str(b).lower() in str(q).lower() else 0.0
        for q, b in zip(df["query_text"], df["brand_text"])
    ]
    df["desc_jaccard"] = [token_jaccard(q, d) for q, d in zip(df["query_text"], df["description_text"])]

    df["candidate_expansion_score"] = (
        0.70 * df["tfidf_score"]
        + 0.20 * df["title_jaccard"]
        + 0.05 * df["brand_match"]
        + 0.05 * df["desc_jaccard"]
    )

    lexical_only_risk = (
        (df["tfidf_score"] > df["tfidf_score"].median())
        & (df["title_jaccard"] < df["title_jaccard"].median())
    ).astype(float)

    df["bias_corrected_rerank_score"] = (
        0.55 * df["candidate_expansion_score"]
        + 0.35 * df["title_jaccard"]
        + 0.08 * df["brand_match"]
        - 0.10 * lexical_only_risk
    )

    base_q = evaluate_by_query(df, "tfidf_score")
    prop_q = evaluate_by_query(df, "bias_corrected_rerank_score")

    base = base_q[[
        "ndcg_at_10",
        "recall_at_10",
        "precision_at_10",
        "high_rank_irrelevant_rate",
    ]].mean().to_dict()

    prop = prop_q[[
        "ndcg_at_10",
        "recall_at_10",
        "precision_at_10",
        "high_rank_irrelevant_rate",
    ]].mean().to_dict()

    checks = {
        "ndcg_improved": prop["ndcg_at_10"] > base["ndcg_at_10"],
        "recall_not_worse": prop["recall_at_10"] >= base["recall_at_10"],
        "high_rank_irrelevant_not_worse": prop["high_rank_irrelevant_rate"] <= base["high_rank_irrelevant_rate"],
    }

    status = "PASS" if all(checks.values()) else "ITERATE"

    metrics = {
        "dataset": "Amazon ESCI Shopping Queries local public-data sample",
        "rows": int(len(df)),
        "queries": int(df["query_id"].nunique()),
        "label_distribution": {
            str(k): int(v)
            for k, v in df["label_num"].value_counts().sort_index().to_dict().items()
        },
        "tfidf_baseline": base,
        "bias_corrected_reranker": prop,
        "release_decision": {
            "status": status,
            "checks": checks,
            "reason": (
                "Bias-corrected reranker promoted."
                if status == "PASS"
                else "Reranker did not beat all promotion checks; keep as ITERATE."
            ),
        },
    }

    (OUTPUTS / "esci_relevance_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )
    prop_q.to_csv(OUTPUTS / "esci_relevance_query_metrics.csv", index=False)
    df.head(5000).to_csv(OUTPUTS / "esci_relevance_scored_sample.csv", index=False)

    report = f"""# ESCI Product-Query Relevance Public-Data Validation Report

## Claim boundary

This run uses the local Amazon ESCI public dataset. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Rows: {metrics["rows"]}
- Queries: {metrics["queries"]}
- Label distribution: {metrics["label_distribution"]}

## Baseline vs proposed reranker

| Metric | TF-IDF baseline | Bias-corrected reranker |
|---|---:|---:|
| NDCG@10 | {base["ndcg_at_10"]:.6f} | {prop["ndcg_at_10"]:.6f} |
| Recall@10 | {base["recall_at_10"]:.6f} | {prop["recall_at_10"]:.6f} |
| Precision@10 | {base["precision_at_10"]:.6f} | {prop["precision_at_10"]:.6f} |
| High-rank irrelevant rate | {base["high_rank_irrelevant_rate"]:.6f} | {prop["high_rank_irrelevant_rate"]:.6f} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {metrics["release_decision"]["reason"]}

## Interpretation

This validates the product-query relevance and reranking module on a real public ESCI dataset sample. The framework promotes the reranker only if it improves or preserves ranking quality and does not increase high-rank irrelevant products.
"""
    (REPORTS / "esci_relevance_public_report.md").write_text(report, encoding="utf-8")

    print("ESCI relevance public-data validation finished.")
    print("Report:", REPORTS / "esci_relevance_public_report.md")
    print("Metrics:", OUTPUTS / "esci_relevance_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
