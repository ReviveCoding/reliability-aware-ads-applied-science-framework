from pathlib import Path
import pandas as pd
from src.common.artifacts import ArtifactManager
from src.pipelines.ctr import run_ctr_track
from src.pipelines.relevance import run_relevance_track
from src.pipelines.uplift import run_uplift_track
from src.pipelines.ope import run_ope_track
from src.policy.auction_ranking import compare_policies
from src.policy.release_gate import decide_release
from src.model_ops.drift_monitor import drift_summary

def run_all(config, root="."):
    root = Path(root); am = ArtifactManager(root)
    ctr = run_ctr_track(config); relevance = run_relevance_track(config); uplift = run_uplift_track(config); ope = run_ope_track(config)
    policy_df = compare_policies(ctr["test"], top_fraction=float(config.get("top_fraction", 0.2)))
    metrics = {"schema": ctr["schema"], "leakage": ctr["leakage"], "ctr": ctr["ctr_metrics"], "calibration": ctr["calibration_metrics"], "residual_correction": ctr["residual_metrics"], "slice_summary": ctr["slice_summary"], "relevance": relevance["metrics"], "uplift": uplift["metrics"], "ope": ope["metrics"], "drift": drift_summary(ctr["train"], ctr["test"]), "policy_comparison": policy_df.to_dict(orient="records"), "config": config}
    decision = decide_release(metrics, policy_df, ctr["slice_summary"], thresholds=config.get("release", {}))
    am.csv("outputs/policy_comparison.csv", policy_df); am.csv("outputs/slice_metrics.csv", ctr["slice_df"]); am.csv("outputs/relevance_query_metrics.csv", relevance["proposed_query_metrics"]); am.json("outputs/metrics_summary.json", metrics); am.json("outputs/release_decision.json", decision)
    am.md("reports/full_pipeline_report.md", "Full Pipeline Report", [("Claim Boundary", "Synthetic fallback data were used. No proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment is claimed."), ("Schema Validation", pd.DataFrame([ctr["schema"]])), ("CTR Metrics", pd.DataFrame(ctr["ctr_metrics"]).T.reset_index().rename(columns={"index":"model"})), ("Calibration Metrics", pd.DataFrame({"raw": ctr["calibration_metrics"]["raw"], "empirical_bayes": ctr["calibration_metrics"]["eb"]}).T.reset_index().rename(columns={"index":"variant"})), ("Residual Bias Correction", pd.DataFrame([ctr["residual_metrics"]])), ("Slice Reliability", ctr["slice_df"]), ("Policy Comparison", policy_df), ("Relevance Metrics", pd.DataFrame(relevance["metrics"]).T.reset_index().rename(columns={"index":"variant"})), ("Uplift Metrics", pd.DataFrame([uplift["metrics"]])), ("OPE Metrics", pd.DataFrame(ope["metrics"]).T.reset_index().rename(columns={"index":"estimator"})), ("Release Decision", f"Status: {decision['status']}\n\nReasons: {decision['reasons']}\n\nRecommended actions: {decision['recommended_actions']}")])
    am.md("reports/calibration_report.md", "Calibration Report", [("Raw vs Empirical-Bayes", pd.DataFrame({"raw": ctr["calibration_metrics"]["raw"], "empirical_bayes": ctr["calibration_metrics"]["eb"]}).T.reset_index().rename(columns={"index":"variant"})), ("Worst Slice Summary", pd.DataFrame([ctr["slice_summary"]]))])
    am.md("reports/ranking_policy_report.md", "Ranking Policy Report", [("Policy Comparison", policy_df)])
    am.md("reports/relevance_report.md", "Relevance and Candidate Expansion Report", [("BM25/TF-IDF vs Bias-Corrected Reranker", pd.DataFrame(relevance["metrics"]).T.reset_index().rename(columns={"index":"variant"}))])
    am.md("reports/uplift_report.md", "Uplift Measurement Report", [("T-Learner Uplift Metrics", pd.DataFrame([uplift["metrics"]]))])
    am.md("reports/ope_report.md", "Off-Policy Evaluation Report", [("OPE Metrics", pd.DataFrame(ope["metrics"]).T.reset_index().rename(columns={"index":"estimator"}))])
    am.md("reports/release_decision.md", "Release Decision", [("Decision", f"Status: {decision['status']}"), ("Reasons", "\n".join(f"- {r}" for r in decision["reasons"])), ("Recommended Actions", "\n".join(f"- {a}" for a in decision["recommended_actions"]))])
    return {"metrics": metrics, "decision": decision}
