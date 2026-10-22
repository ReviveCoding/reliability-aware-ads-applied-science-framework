import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


PUBLIC_METRIC_FILES = {
    "criteo_uplift": OUTPUTS / "criteo_uplift_metrics.json",
    "esci_relevance": OUTPUTS / "esci_relevance_metrics.json",
    "open_bandit_ope": (OUTPUTS / "open_bandit_ope_metrics_corrected.json"
                        if (OUTPUTS / "open_bandit_ope_metrics_corrected.json").exists()
                        else OUTPUTS / "open_bandit_ope_metrics.json"),
    "criteo_click_ctr": OUTPUTS / "criteo_click_ctr_metrics.json",
}


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def decision_status(metrics):
    if not metrics:
        return "MISSING"
    decision = metrics.get("release_decision")
    if isinstance(decision, dict):
        return decision.get("status", "UNKNOWN")
    return "COMPLETED"


def one_line_summary(name, metrics):
    if not metrics:
        return {
            "module": name,
            "dataset": "",
            "rows": "",
            "status": "MISSING",
            "primary_result": "Metrics file not found.",
        }

    status = decision_status(metrics)
    dataset = metrics.get("dataset", "")
    rows = metrics.get("rows", metrics.get("rows_loaded", ""))

    if name == "criteo_uplift":
        u = metrics.get("uplift_model", {})
        b = metrics.get("pcvr_baseline", {})
        primary = (
            f"T-learner AUUC={u.get('auuc')}, Qini={u.get('qini_proxy')}, "
            f"uplift@20={u.get('uplift_at_20pct')}; pCVR baseline AUUC={b.get('auuc')}, "
            f"Qini={b.get('qini_proxy')}, uplift@20={b.get('uplift_at_20pct')}."
        )
        status = "ITERATE" if b and u and b.get("auuc", 0) >= u.get("auuc", 0) else status
    elif name == "esci_relevance":
        base = metrics.get("tfidf_baseline", {})
        prop = metrics.get("bias_corrected_reranker", {})
        primary = (
            f"TF-IDF NDCG@10={base.get('ndcg_at_10')}, Recall@10={base.get('recall_at_10')}; "
            f"reranker NDCG@10={prop.get('ndcg_at_10')}, Recall@10={prop.get('recall_at_10')}."
        )
    elif name == "open_bandit_ope":
        dr = metrics.get("doubly_robust", {})
        diag = metrics.get("diagnostics", {})
        primary = (
            f"Logged value={metrics.get('logged_policy_value')}, DR value={dr.get('value')}, "
            f"DR CI=[{dr.get('ci_low')}, {dr.get('ci_high')}], ESS={diag.get('effective_sample_size')}."
        )
        if metrics.get("estimator_version") != "crossfit-action-integrated-v1":
            primary = ("Historical OPE estimates withdrawn pending corrected validation; "
                       f"historical ESS={diag.get('effective_sample_size')}. "
                       "Do not use historical DR/SNIPS intervals as validated evidence.")
            status = "ITERATE"
    elif name == "criteo_click_ctr":
        lr = metrics.get("logistic_regression", {})
        gbdt = metrics.get("gbdt", {})
        eb = metrics.get("gbdt_empirical_bayes", {})
        primary = (
            f"LR AUC={lr.get('roc_auc')}, ECE={lr.get('ece')}; "
            f"GBDT AUC={gbdt.get('roc_auc')}, ECE={gbdt.get('ece')}; "
            f"GBDT+EB AUC={eb.get('roc_auc')}, ECE={eb.get('ece')}."
        )
    else:
        primary = "Completed."

    return {
        "module": name,
        "dataset": dataset,
        "rows": rows,
        "status": status,
        "primary_result": primary,
    }


def main():
    metrics = {name: read_json(path) for name, path in PUBLIC_METRIC_FILES.items()}
    summaries = [one_line_summary(name, item) for name, item in metrics.items()]

    overall_status = "PASS_WITH_ITERATIONS"
    if any(s["status"] == "MISSING" for s in summaries):
        overall_status = "INCOMPLETE"
    elif all(s["status"] == "PASS" for s in summaries):
        overall_status = "PASS"
    elif any(s["status"] in {"ITERATE", "BLOCK"} for s in summaries):
        overall_status = "PUBLIC_RUNNERS_COMPLETED_WITH_ITERATE_DECISIONS"

    result = {
        "overall_status": overall_status,
        "claim_boundary": (
            "Public-data validation uses local public datasets and synthetic fallback only. "
            "No proprietary Amazon Ads data, advertiser data, customer data, production traffic, or deployment is claimed."
        ),
        "modules": summaries,
    }

    (OUTPUTS / "public_validation_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    rows = []
    for s in summaries:
        rows.append(
            f"| {s['module']} | {s['rows']} | {s['status']} | {s['primary_result']} |"
        )

    md = f"""# Public-Data Validation Summary

## Overall status

**{overall_status}**

## Claim boundary

Public-data validation uses local public datasets and synthetic fallback only. No proprietary Amazon Ads data, advertiser data, customer data, production traffic, or deployment is claimed.

## Module summary

| Module | Rows | Release status | Primary result |
|---|---:|---|---|
{chr(10).join(rows)}

## Interpretation

The table records module-specific evidence; historical Open Bandit estimates without corrected estimator provenance are not validated OPE evidence. Several candidate models are intentionally held at ITERATE because release gates require more than a single improved metric. This is expected and useful: the project demonstrates metric-driven promotion discipline rather than method-driven overclaiming.

## Safe resume wording

Built a local-runnable, public-data-compatible ads applied-science framework validating CTR prediction, empirical-Bayes calibration, product-query relevance, uplift measurement, off-policy evaluation, ranking-policy diagnostics, and release gates across Criteo Click Logs, Criteo Uplift, Amazon ESCI, and Open Bandit public-data samples.

## Avoid overclaiming

Do not claim production Amazon Ads deployment, proprietary customer/advertiser data, or that every proposed challenger beat every baseline. The current public-data evidence supports validated runners and release-gated ITERATE decisions.
"""

    (REPORTS / "public_validation_summary.md").write_text(md, encoding="utf-8")

    print("Wrote:", OUTPUTS / "public_validation_summary.json")
    print("Wrote:", REPORTS / "public_validation_summary.md")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
