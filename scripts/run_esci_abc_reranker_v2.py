import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

LABEL_TO_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
LABEL_TO_NUM = {"I": 0, "C": 1, "S": 2, "E": 3}

FEATURE_COLS = [
    "tfidf_title", "tfidf_desc", "tfidf_bullet", "tfidf_all",
    "char_tfidf_title", "char_tfidf_all",
    "title_jaccard", "desc_jaccard", "bullet_jaccard", "all_jaccard",
    "title_query_coverage", "all_query_coverage",
    "title_contains_all_query", "all_contains_all_query",
    "char_title_jaccard", "char_desc_jaccard",
    "brand_in_query", "brand_in_title",
    "query_len", "title_len", "desc_len", "len_ratio_title_query",
    "exact_query_in_title", "exact_query_in_all",
]


def find_esci_files(data_dir):
    root = Path(data_dir)
    examples = list(root.rglob("shopping_queries_dataset_examples.parquet"))
    products = list(root.rglob("shopping_queries_dataset_products.parquet"))
    if not examples or not products:
        raise FileNotFoundError(
            f"Could not find ESCI parquet files under {root}. Expected shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet."
        )
    return examples[0], products[0]


def read_esci(data_dir, max_rows=160000, locale="us"):
    examples_path, products_path = find_esci_files(data_dir)
    print("Examples file:", examples_path)
    print("Products file:", products_path)
    ex = pd.read_parquet(examples_path)
    prod = pd.read_parquet(products_path)
    if locale:
        if "product_locale" in ex.columns:
            ex = ex[ex["product_locale"].astype(str).str.lower().eq(locale.lower())].copy()
        if "product_locale" in prod.columns:
            prod = prod[prod["product_locale"].astype(str).str.lower().eq(locale.lower())].copy()
    if "small_version" in ex.columns:
        small = ex[ex["small_version"].astype(bool)].copy()
        if len(small) >= 10000:
            ex = small
    ex = ex.head(max_rows).copy()
    keep_prod = [
        "product_id", "product_title", "product_description", "product_bullet_point",
        "product_brand", "product_color",
    ]
    keep_prod = [c for c in keep_prod if c in prod.columns]
    df = ex.merge(prod[keep_prod], on="product_id", how="left")
    df["esci_label"] = df["esci_label"].astype(str).str.upper().str[0]
    df = df[df["esci_label"].isin(LABEL_TO_GAIN)].copy()
    df["gain"] = df["esci_label"].map(LABEL_TO_GAIN).astype(float)
    df["label_num"] = df["esci_label"].map(LABEL_TO_NUM).astype(int)
    for c in ["query", "product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
        else:
            df[c] = ""
    df["query_id_safe"] = df["query_id"].astype(str) if "query_id" in df.columns else df["query"].astype(str)
    return df, examples_path, products_path


def tokenize(text):
    text = str(text).lower()
    out, cur = [], []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out


def token_set(text):
    return set(tokenize(text))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def contains_all(query_tokens, product_tokens):
    if not query_tokens:
        return 0.0
    return float(query_tokens.issubset(product_tokens))


def char_ngrams(s, n=3):
    s = "".join(ch.lower() for ch in str(s) if ch.isalnum() or ch.isspace())
    s = " ".join(s.split())
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def add_lexical_features(df):
    rows = []
    for _, r in df.iterrows():
        q = r["query"]
        title = r["product_title"]
        desc = r["product_description"]
        bullet = r["product_bullet_point"]
        brand = r["product_brand"]
        q_tok = token_set(q)
        title_tok = token_set(title)
        desc_tok = token_set(desc)
        bullet_tok = token_set(bullet)
        all_tok = title_tok | desc_tok | bullet_tok | token_set(brand)
        q_char = char_ngrams(q)
        title_char = char_ngrams(title)
        desc_char = char_ngrams(desc)
        rows.append({
            "title_jaccard": jaccard(q_tok, title_tok),
            "desc_jaccard": jaccard(q_tok, desc_tok),
            "bullet_jaccard": jaccard(q_tok, bullet_tok),
            "all_jaccard": jaccard(q_tok, all_tok),
            "title_query_coverage": len(q_tok & title_tok) / max(len(q_tok), 1),
            "all_query_coverage": len(q_tok & all_tok) / max(len(q_tok), 1),
            "title_contains_all_query": contains_all(q_tok, title_tok),
            "all_contains_all_query": contains_all(q_tok, all_tok),
            "char_title_jaccard": jaccard(q_char, title_char),
            "char_desc_jaccard": jaccard(q_char, desc_char),
            "brand_in_query": float(bool(brand and str(brand).lower() in str(q).lower())),
            "brand_in_title": float(bool(brand and str(brand).lower() in str(title).lower())),
            "query_len": float(len(q_tok)),
            "title_len": float(len(title_tok)),
            "desc_len": float(len(desc_tok)),
            "len_ratio_title_query": float(len(title_tok) / max(len(q_tok), 1)),
            "exact_query_in_title": float(str(q).lower() in str(title).lower()),
            "exact_query_in_all": float(str(q).lower() in (str(title) + " " + str(desc) + " " + str(bullet)).lower()),
        })
    feat = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def fit_tfidf(train_df):
    corpus = (
        train_df["query"].astype(str).tolist()
        + train_df["product_title"].astype(str).tolist()
        + train_df["product_description"].astype(str).tolist()
        + train_df["product_bullet_point"].astype(str).tolist()
    )
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True, norm="l2")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True, norm="l2")
    print("Fitting TF-IDF word vectorizer...")
    word.fit(corpus)
    print("Fitting TF-IDF char vectorizer...")
    char.fit(corpus)
    return word, char


def sparse_cosine_same_rows(a, b):
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def add_tfidf_scores(df, word, char):
    query = df["query"].astype(str)
    title = df["product_title"].astype(str)
    desc = df["product_description"].astype(str)
    bullet = df["product_bullet_point"].astype(str)
    all_text = (title + " " + desc + " " + bullet).astype(str)
    q_word = word.transform(query)
    q_char = char.transform(query)
    out = df.copy()
    out["tfidf_title"] = sparse_cosine_same_rows(q_word, word.transform(title))
    out["tfidf_desc"] = sparse_cosine_same_rows(q_word, word.transform(desc))
    out["tfidf_bullet"] = sparse_cosine_same_rows(q_word, word.transform(bullet))
    out["tfidf_all"] = sparse_cosine_same_rows(q_word, word.transform(all_text))
    out["char_tfidf_title"] = sparse_cosine_same_rows(q_char, char.transform(title))
    out["char_tfidf_all"] = sparse_cosine_same_rows(q_char, char.transform(all_text))
    return out


def dcg(gains):
    gains = np.asarray(gains, dtype=float)
    if len(gains) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(group, score_col, k=10):
    ranked = group.sort_values(score_col, ascending=False)
    top = ranked.head(k)["gain"].values
    ideal = group.sort_values("gain", ascending=False).head(k)["gain"].values
    denom = dcg(ideal)
    return 0.0 if denom <= 0 else dcg(top) / denom


def recall_at_k(group, score_col, k=10):
    rel = group[group["gain"] > 0]
    if len(rel) == 0:
        return np.nan
    top = group.sort_values(score_col, ascending=False).head(k)
    return float((top["gain"] > 0).sum() / len(rel))


def precision_at_k(group, score_col, k=10):
    top = group.sort_values(score_col, ascending=False).head(k)
    return np.nan if len(top) == 0 else float((top["gain"] > 0).mean())


def high_rank_irrelevant_rate(group, score_col, k=10):
    top = group.sort_values(score_col, ascending=False).head(k)
    return np.nan if len(top) == 0 else float((top["gain"] <= 0).mean())


def evaluate_ranking(df, score_col, k=10):
    vals = []
    for _, g in df.groupby("query_id_safe"):
        vals.append({
            "ndcg_at_10": ndcg_at_k(g, score_col, k),
            "recall_at_10": recall_at_k(g, score_col, k),
            "precision_at_10": precision_at_k(g, score_col, k),
            "high_rank_irrelevant_rate": high_rank_irrelevant_rate(g, score_col, k),
        })
    m = pd.DataFrame(vals)
    return {
        "ndcg_at_10": float(m["ndcg_at_10"].mean()),
        "recall_at_10": float(m["recall_at_10"].dropna().mean()),
        "precision_at_10": float(m["precision_at_10"].dropna().mean()),
        "high_rank_irrelevant_rate": float(m["high_rank_irrelevant_rate"].dropna().mean()),
        "query_count": int(len(m)),
    }


def split_by_query(df, seed=42):
    qids = df["query_id_safe"].astype(str).drop_duplicates().to_numpy(dtype=object)
    train_q, temp_q = train_test_split(qids, test_size=0.40, random_state=seed)
    val_q, test_q = train_test_split(temp_q, test_size=0.50, random_state=seed)
    return (
        df[df["query_id_safe"].isin(train_q)].copy(),
        df[df["query_id_safe"].isin(val_q)].copy(),
        df[df["query_id_safe"].isin(test_q)].copy(),
    )


def train_models(train_df):
    x = train_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_rel = (train_df["gain"] > 0).astype(int)
    y_irrel = (train_df["gain"] <= 0).astype(int)
    y_label = train_df["label_num"].astype(int)
    print("Training relevance model...")
    rel_model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    rel_model.fit(x, y_rel)
    print("Training overreach/irrelevant-risk model...")
    irrel_model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))])
    irrel_model.fit(x, y_irrel)
    print("Training graded ESCI model...")
    graded_model = HistGradientBoostingClassifier(
        max_iter=160, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.02, random_state=42,
    )
    graded_model.fit(x, y_label)
    return rel_model, irrel_model, graded_model


def add_model_scores(df, rel_model, irrel_model, graded_model):
    out = df.copy()
    x = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["p_relevant"] = rel_model.predict_proba(x)[:, 1]
    out["p_irrelevant"] = irrel_model.predict_proba(x)[:, 1]
    proba = graded_model.predict_proba(x)
    classes = list(graded_model.classes_)
    expected_gain = np.zeros(len(out), dtype=float)
    for cls, gain in [(0, 0.0), (1, 0.01), (2, 0.1), (3, 1.0)]:
        if cls in classes:
            expected_gain += proba[:, classes.index(cls)] * gain
    out["expected_gain"] = expected_gain
    return out


def add_scores(df):
    out = df.copy()
    out["baseline_score"] = 0.70 * out["tfidf_title"] + 0.20 * out["char_tfidf_title"] + 0.10 * out["tfidf_all"]
    out["expansion_gain"] = (
        0.30 * out["tfidf_desc"] + 0.25 * out["tfidf_bullet"] + 0.20 * out["char_tfidf_all"]
        + 0.15 * out["all_query_coverage"] + 0.10 * out["brand_in_title"]
    )
    out["lexical_overreach_risk"] = (
        0.35 * (1.0 - out["title_query_coverage"].clip(0, 1))
        + 0.25 * (1.0 - out["title_jaccard"].clip(0, 1))
        + 0.20 * (out["tfidf_desc"] > out["tfidf_title"]).astype(float)
        + 0.20 * (out["len_ratio_title_query"] > 8).astype(float)
    )
    return out


def apply_candidate_score(df, alpha, beta, gamma, delta):
    out = df.copy()
    out["abc_score"] = (
        out["baseline_score"]
        + alpha * out["expansion_gain"]
        + beta * out["expected_gain"]
        + gamma * out["p_relevant"]
        - delta * (0.50 * out["p_irrelevant"] + 0.50 * out["lexical_overreach_risk"])
    )
    return out


def tune_abc(val_df, baseline_metrics):
    candidates = []
    for a in [0.0, 0.10, 0.20, 0.30, 0.45, 0.60]:
        for b in [0.0, 0.20, 0.40, 0.60, 0.80]:
            for g in [0.0, 0.10, 0.20, 0.35]:
                for d in [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]:
                    scored = apply_candidate_score(val_df, a, b, g, d)
                    m = evaluate_ranking(scored, "abc_score")
                    recall_drop = max(0.0, baseline_metrics["recall_at_10"] - m["recall_at_10"])
                    irrel_increase = max(0.0, m["high_rank_irrelevant_rate"] - baseline_metrics["high_rank_irrelevant_rate"])
                    ndcg_drop = max(0.0, baseline_metrics["ndcg_at_10"] - m["ndcg_at_10"])
                    score = m["ndcg_at_10"] + 0.25 * m["recall_at_10"] - 0.35 * m["high_rank_irrelevant_rate"] - recall_drop - irrel_increase - 0.50 * ndcg_drop
                    candidates.append({
                        "alpha_expansion": a, "beta_expected_gain": b, "gamma_relevance": g, "delta_overreach": d,
                        "metrics": m, "validation_selection_score": float(score),
                    })
    guarded = [
        c for c in candidates
        if c["metrics"]["ndcg_at_10"] >= baseline_metrics["ndcg_at_10"]
        and c["metrics"]["recall_at_10"] >= baseline_metrics["recall_at_10"]
        and c["metrics"]["high_rank_irrelevant_rate"] <= baseline_metrics["high_rank_irrelevant_rate"]
    ]
    return max(guarded if guarded else candidates, key=lambda c: c["validation_selection_score"]), candidates


def main():
    data_dir = os.environ.get("ESCI_DATA_DIR")
    if not data_dir:
        raise SystemExit("Set ESCI_DATA_DIR first.")
    max_rows = int(os.environ.get("ESCI_ABC_MAX_ROWS", "160000"))
    locale = os.environ.get("ESCI_ABC_LOCALE", "us")
    start = time.time()
    df, examples_path, products_path = read_esci(data_dir, max_rows=max_rows, locale=locale)
    print("Loaded rows:", len(df))
    print("Queries:", df["query_id_safe"].nunique())
    print("Label distribution:")
    print(df["esci_label"].value_counts())
    train_df, val_df, test_df = split_by_query(df)
    print("Split rows:", len(train_df), len(val_df), len(test_df))
    print("Split queries:", train_df["query_id_safe"].nunique(), val_df["query_id_safe"].nunique(), test_df["query_id_safe"].nunique())
    word, char = fit_tfidf(train_df)
    train_df = add_lexical_features(add_tfidf_scores(train_df, word, char))
    val_df = add_lexical_features(add_tfidf_scores(val_df, word, char))
    test_df = add_lexical_features(add_tfidf_scores(test_df, word, char))
    train_df = add_scores(train_df)
    val_df = add_scores(val_df)
    test_df = add_scores(test_df)
    rel_model, irrel_model, graded_model = train_models(train_df)
    train_df = add_model_scores(train_df, rel_model, irrel_model, graded_model)
    val_df = add_model_scores(val_df, rel_model, irrel_model, graded_model)
    test_df = add_model_scores(test_df, rel_model, irrel_model, graded_model)
    baseline_val = evaluate_ranking(val_df, "baseline_score")
    baseline_test = evaluate_ranking(test_df, "baseline_score")
    selected, candidates = tune_abc(val_df, baseline_val)
    val_scored = apply_candidate_score(val_df, selected["alpha_expansion"], selected["beta_expected_gain"], selected["gamma_relevance"], selected["delta_overreach"])
    test_scored = apply_candidate_score(test_df, selected["alpha_expansion"], selected["beta_expected_gain"], selected["gamma_relevance"], selected["delta_overreach"])
    abc_val = evaluate_ranking(val_scored, "abc_score")
    abc_test = evaluate_ranking(test_scored, "abc_score")
    checks = {
        "ndcg_not_worse_than_baseline": abc_test["ndcg_at_10"] >= baseline_test["ndcg_at_10"] - 0.0005,
        "recall_not_worse_than_baseline": abc_test["recall_at_10"] >= baseline_test["recall_at_10"] - 0.0005,
        "high_rank_irrelevant_not_worse": abc_test["high_rank_irrelevant_rate"] <= baseline_test["high_rank_irrelevant_rate"] + 0.0005,
        "abc_has_expansion_or_overreach_control": (
            selected["alpha_expansion"] > 0 or selected["delta_overreach"] > 0 or selected["beta_expected_gain"] > 0 or selected["gamma_relevance"] > 0
        ),
    }
    status = "PASS" if all(checks.values()) else "ITERATE"
    result = {
        "dataset": "Amazon ESCI Shopping Queries local public-data sample",
        "method": "ABC-style candidate expansion and overreach correction reranker",
        "claim_boundary": "Local public-data validation only. No proprietary Amazon data, production traffic, customer data, or deployment is claimed.",
        "source_files": {"examples": str(examples_path), "products": str(products_path)},
        "rows": int(len(df)), "queries": int(df["query_id_safe"].nunique()), "locale": locale,
        "label_distribution": {str(k): int(v) for k, v in df["esci_label"].value_counts().to_dict().items()},
        "split": {
            "train_rows": int(len(train_df)), "val_rows": int(len(val_df)), "test_rows": int(len(test_df)),
            "train_queries": int(train_df["query_id_safe"].nunique()), "val_queries": int(val_df["query_id_safe"].nunique()), "test_queries": int(test_df["query_id_safe"].nunique()),
        },
        "baseline_validation": baseline_val, "abc_validation": abc_val,
        "baseline_test": baseline_test, "abc_test": abc_test,
        "selected_parameters": {
            "alpha_expansion": selected["alpha_expansion"], "beta_expected_gain": selected["beta_expected_gain"],
            "gamma_relevance": selected["gamma_relevance"], "delta_overreach": selected["delta_overreach"],
        },
        "release_decision": {
            "status": status, "checks": checks,
            "reason": "ABC-style reranker passed NDCG, recall, and high-rank irrelevant guardrails." if status == "PASS" else "ABC-style reranker trained successfully but did not pass all ranking/overreach guardrails.",
        },
        "runtime_seconds": time.time() - start,
    }
    (OUTPUTS / "esci_abc_reranker_v2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(candidates).to_json(OUTPUTS / "esci_abc_reranker_v2_candidates.json", orient="records", indent=2)
    cols = ["query_id_safe", "query", "product_id", "product_title", "esci_label", "baseline_score", "abc_score", "expansion_gain", "p_relevant", "p_irrelevant", "expected_gain", "lexical_overreach_risk"]
    test_scored[[c for c in cols if c in test_scored.columns]].head(5000).to_csv(OUTPUTS / "esci_abc_reranker_v2_scored_sample.csv", index=False)
    report = f"""# ABC-style ESCI Reranker v2 Public-Data Validation Report

## Claim boundary

This run uses local public Amazon ESCI Shopping Queries data only. It does not use proprietary Amazon data, customer data, production traffic, or production deployment.

## Method

This experiment adapts the ABC idea of candidate expansion plus learned overreach correction to product-query relevance ranking.

- Baseline candidate: title-heavy TF-IDF relevance score.
- Expansion signal: description, bullet, all-text, character n-gram, coverage, and brand evidence.
- Bias/overreach correction: learned irrelevant-risk probability plus lexical overreach risk.
- Selection: validation query-level grid search under NDCG, recall, and high-rank irrelevant guardrails.

## Data summary

- Rows: {result['rows']}
- Queries: {result['queries']}
- Locale: `{locale}`
- Train / validation / test queries: {result['split']['train_queries']} / {result['split']['val_queries']} / {result['split']['test_queries']}

## Selected parameters

- alpha expansion: {selected['alpha_expansion']}
- beta expected gain: {selected['beta_expected_gain']}
- gamma relevance: {selected['gamma_relevance']}
- delta overreach: {selected['delta_overreach']}

## Test comparison

| Metric | TF-IDF baseline | ABC-style reranker |
|---|---:|---:|
| NDCG@10 | {baseline_test['ndcg_at_10']:.6f} | {abc_test['ndcg_at_10']:.6f} |
| Recall@10 | {baseline_test['recall_at_10']:.6f} | {abc_test['recall_at_10']:.6f} |
| Precision@10 | {baseline_test['precision_at_10']:.6f} | {abc_test['precision_at_10']:.6f} |
| High-rank irrelevant rate | {baseline_test['high_rank_irrelevant_rate']:.6f} | {abc_test['high_rank_irrelevant_rate']:.6f} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {result['release_decision']['reason']}

## Interpretation

The release gate requires that expansion improves or preserves relevant coverage without creating high-rank irrelevant overreach. This mirrors the original ABC logic: expansion is useful only when paired with correction that controls false positive extension.
"""
    (REPORTS / "esci_abc_reranker_v2_report.md").write_text(report, encoding="utf-8")
    print("ABC-style ESCI reranker v2 validation finished.")
    print("Report:", REPORTS / "esci_abc_reranker_v2_report.md")
    print("Metrics:", OUTPUTS / "esci_abc_reranker_v2_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
