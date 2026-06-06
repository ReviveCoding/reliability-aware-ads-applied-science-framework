import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the tested ESCI v2 implementation in the local scripts directory.
import run_esci_abc_reranker_v2 as base


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def score_variant(df, name, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
    out = df.copy()
    out[name] = (
        out["baseline_score"]
        + alpha * out["expansion_gain"]
        + beta * out["expected_gain"]
        + gamma * out["p_relevant"]
        - delta * (0.50 * out["p_irrelevant"] + 0.50 * out["lexical_overreach_risk"])
    )
    return out


def evaluate_variant(df, score_col):
    return base.evaluate_ranking(df, score_col)


def delta_metrics(metrics, baseline):
    return {
        "ndcg_delta": metrics["ndcg_at_10"] - baseline["ndcg_at_10"],
        "recall_delta": metrics["recall_at_10"] - baseline["recall_at_10"],
        "precision_delta": metrics["precision_at_10"] - baseline["precision_at_10"],
        "high_rank_irrelevant_delta": metrics["high_rank_irrelevant_rate"] - baseline["high_rank_irrelevant_rate"],
    }


def pass_checks(metrics, baseline):
    return {
        "ndcg_not_worse": metrics["ndcg_at_10"] >= baseline["ndcg_at_10"] - 0.0005,
        "recall_not_worse": metrics["recall_at_10"] >= baseline["recall_at_10"] - 0.0005,
        "irrelevant_not_worse": metrics["high_rank_irrelevant_rate"] <= baseline["high_rank_irrelevant_rate"] + 0.0005,
    }


def run_variant(test_df, name, alpha, beta, gamma, delta, baseline_test):
    scored = score_variant(test_df, name, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    metrics = evaluate_variant(scored, name)
    checks = pass_checks(metrics, baseline_test)
    return {
        "variant": name,
        "parameters": {
            "alpha_expansion": alpha,
            "beta_expected_gain": beta,
            "gamma_relevance": gamma,
            "delta_overreach": delta,
        },
        "test_metrics": metrics,
        "delta_vs_baseline": delta_metrics(metrics, baseline_test),
        "release_checks_vs_baseline": checks,
        "passes_all_guardrails": bool(all(checks.values())),
    }


def build_pipeline(data_dir, max_rows, locale):
    df, examples_path, products_path = base.read_esci(data_dir, max_rows=max_rows, locale=locale)

    print("Loaded rows:", len(df))
    print("Queries:", df["query_id_safe"].nunique())
    print("Label distribution:")
    print(df["esci_label"].value_counts())

    train_df, val_df, test_df = base.split_by_query(df)

    print("Split rows:", len(train_df), len(val_df), len(test_df))
    print("Split queries:", train_df["query_id_safe"].nunique(), val_df["query_id_safe"].nunique(), test_df["query_id_safe"].nunique())

    word, char = base.fit_tfidf(train_df)

    train_df = base.add_lexical_features(base.add_tfidf_scores(train_df, word, char))
    val_df = base.add_lexical_features(base.add_tfidf_scores(val_df, word, char))
    test_df = base.add_lexical_features(base.add_tfidf_scores(test_df, word, char))

    train_df = base.add_scores(train_df)
    val_df = base.add_scores(val_df)
    test_df = base.add_scores(test_df)

    rel_model, irrel_model, graded_model = base.train_models(train_df)

    train_df = base.add_model_scores(train_df, rel_model, irrel_model, graded_model)
    val_df = base.add_model_scores(val_df, rel_model, irrel_model, graded_model)
    test_df = base.add_model_scores(test_df, rel_model, irrel_model, graded_model)

    return df, train_df, val_df, test_df, examples_path, products_path


def main():
    data_dir = os.environ.get("ESCI_DATA_DIR")
    if not data_dir:
        raise SystemExit("Set ESCI_DATA_DIR first.")

    max_rows = int(os.environ.get("ESCI_ABLATION_MAX_ROWS", os.environ.get("ESCI_ABC_MAX_ROWS", "160000")))
    locale = os.environ.get("ESCI_ABLATION_LOCALE", os.environ.get("ESCI_ABC_LOCALE", "us"))

    start = time.time()
    df, train_df, val_df, test_df, examples_path, products_path = build_pipeline(data_dir, max_rows, locale)

    baseline_val = evaluate_variant(val_df, "baseline_score")
    baseline_test = evaluate_variant(test_df, "baseline_score")

    selected, candidates = base.tune_abc(val_df, baseline_val)
    selected_params = {
        "alpha_expansion": selected["alpha_expansion"],
        "beta_expected_gain": selected["beta_expected_gain"],
        "gamma_relevance": selected["gamma_relevance"],
        "delta_overreach": selected["delta_overreach"],
    }

    variants = []
    variants.append({
        "variant": "baseline_tfidf",
        "parameters": {"alpha_expansion": 0.0, "beta_expected_gain": 0.0, "gamma_relevance": 0.0, "delta_overreach": 0.0},
        "test_metrics": baseline_test,
        "delta_vs_baseline": delta_metrics(baseline_test, baseline_test),
        "release_checks_vs_baseline": pass_checks(baseline_test, baseline_test),
        "passes_all_guardrails": True,
    })

    # One-factor ablations.
    variants.append(run_variant(test_df, "expansion_only", alpha=0.20, beta=0.0, gamma=0.0, delta=0.0, baseline_test=baseline_test))
    variants.append(run_variant(test_df, "expected_gain_only", alpha=0.0, beta=0.80, gamma=0.0, delta=0.0, baseline_test=baseline_test))
    variants.append(run_variant(test_df, "relevance_only", alpha=0.0, beta=0.0, gamma=0.20, delta=0.0, baseline_test=baseline_test))
    variants.append(run_variant(test_df, "overreach_only", alpha=0.0, beta=0.0, gamma=0.0, delta=0.35, baseline_test=baseline_test))

    # Combined variants.
    variants.append(run_variant(test_df, "expansion_plus_expected_gain", alpha=0.20, beta=0.80, gamma=0.0, delta=0.0, baseline_test=baseline_test))
    variants.append(run_variant(test_df, "expansion_plus_expected_gain_forced_overreach_0_20", alpha=0.20, beta=0.80, gamma=0.0, delta=0.20, baseline_test=baseline_test))
    variants.append(run_variant(test_df, "expansion_plus_expected_gain_forced_overreach_0_50", alpha=0.20, beta=0.80, gamma=0.0, delta=0.50, baseline_test=baseline_test))

    # Validation-selected ABC model.
    variants.append(run_variant(
        test_df,
        "validation_selected_abc",
        alpha=selected["alpha_expansion"],
        beta=selected["beta_expected_gain"],
        gamma=selected["gamma_relevance"],
        delta=selected["delta_overreach"],
        baseline_test=baseline_test,
    ))

    def variant_score(v):
        m = v["test_metrics"]
        violations = 0 if v["passes_all_guardrails"] else 1
        return (
            m["ndcg_at_10"]
            + 0.25 * m["recall_at_10"]
            - 0.35 * m["high_rank_irrelevant_rate"]
            - 10.0 * violations
        )

    best = max(variants, key=variant_score)

    ablation_table = pd.DataFrame([
        {
            "variant": v["variant"],
            **v["parameters"],
            **v["test_metrics"],
            **v["delta_vs_baseline"],
            "passes_all_guardrails": v["passes_all_guardrails"],
        }
        for v in variants
    ]).sort_values(["passes_all_guardrails", "ndcg_at_10"], ascending=[False, False])

    result = {
        "dataset": "Amazon ESCI Shopping Queries local public-data sample",
        "method": "ABC-style ESCI v2 ablation study",
        "claim_boundary": "Local public-data validation only. No proprietary Amazon data, production traffic, customer data, or deployment is claimed.",
        "source_files": {"examples": str(examples_path), "products": str(products_path)},
        "rows": int(len(df)),
        "queries": int(df["query_id_safe"].nunique()),
        "locale": locale,
        "split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_queries": int(train_df["query_id_safe"].nunique()),
            "val_queries": int(val_df["query_id_safe"].nunique()),
            "test_queries": int(test_df["query_id_safe"].nunique()),
        },
        "baseline_validation": baseline_val,
        "baseline_test": baseline_test,
        "validation_selected_parameters": selected_params,
        "variants": variants,
        "best_variant_by_guardrail_score": best,
        "interpretation": {
            "primary_question": "Which ABC component drives the improvement: expansion, expected-gain scoring, relevance scoring, or explicit overreach penalty?",
            "notes": [
                "If expected_gain_only or expansion_plus_expected_gain is strongest, the observed ABC v2 gain is mainly from learned expected-gain reranking.",
                "If forced-overreach variants reduce high-rank irrelevant but hurt NDCG or recall, explicit overreach penalty is useful as a guardrail but not the promoted scoring component.",
                "If validation_selected_abc remains best, the original ABC v2 selection is stable under ablation."
            ],
        },
        "runtime_seconds": time.time() - start,
    }

    (OUTPUTS / "esci_abc_reranker_v2_ablation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ablation_table.to_csv(OUTPUTS / "esci_abc_reranker_v2_ablation_table.csv", index=False)

    report_rows = []
    for _, row in ablation_table.iterrows():
        report_rows.append(
            f"| {row['variant']} | {row['alpha_expansion']:.2f} | {row['beta_expected_gain']:.2f} | {row['gamma_relevance']:.2f} | {row['delta_overreach']:.2f} | "
            f"{row['ndcg_at_10']:.6f} | {row['recall_at_10']:.6f} | {row['precision_at_10']:.6f} | {row['high_rank_irrelevant_rate']:.6f} | {row['passes_all_guardrails']} |"
        )

    md = f"""# ABC-style ESCI Reranker v2 Ablation Report

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

- Rows: {result['rows']}
- Queries: {result['queries']}
- Train / validation / test queries: {result['split']['train_queries']} / {result['split']['val_queries']} / {result['split']['test_queries']}

## Baseline test metrics

- NDCG@10: {baseline_test['ndcg_at_10']:.6f}
- Recall@10: {baseline_test['recall_at_10']:.6f}
- Precision@10: {baseline_test['precision_at_10']:.6f}
- High-rank irrelevant rate: {baseline_test['high_rank_irrelevant_rate']:.6f}

## Ablation comparison

| Variant | alpha expansion | beta expected gain | gamma relevance | delta overreach | NDCG@10 | Recall@10 | Precision@10 | High-rank irrelevant | Passes guardrails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(report_rows)}

## Best variant by guardrail score

- Variant: **{best['variant']}**
- Parameters: {best['parameters']}
- Test metrics: {best['test_metrics']}

## Interpretation guide

If the best variants use beta expected gain and not delta overreach, then the main validated contribution is expected-gain reranking after expansion rather than explicit irrelevant penalty. If forced-overreach variants reduce high-rank irrelevant rate but hurt NDCG or recall, overreach control should remain a release guardrail rather than a promoted scoring term.
"""
    (REPORTS / "esci_abc_reranker_v2_ablation_report.md").write_text(md, encoding="utf-8")

    print("ABC-style ESCI reranker v2 ablation finished.")
    print("Report:", REPORTS / "esci_abc_reranker_v2_ablation_report.md")
    print("Metrics:", OUTPUTS / "esci_abc_reranker_v2_ablation_metrics.json")
    print("Table:", OUTPUTS / "esci_abc_reranker_v2_ablation_table.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
