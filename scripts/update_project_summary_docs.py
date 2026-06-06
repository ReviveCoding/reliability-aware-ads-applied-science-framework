"""
Update README and public validation summary reports after GPU CTR and ESCI ABC ablation runs.

Run from repository root:
    python scripts/update_project_summary_docs.py

This script reads existing JSON outputs under ./outputs and updates:
- reports/public_validation_summary.md
- reports/final_model_evidence_summary.md
- outputs/public_validation_summary.json
- README.md between PUBLIC_VALIDATION_SUMMARY markers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
README = ROOT / "README.md"

START = "<!-- PUBLIC_VALIDATION_SUMMARY_START -->"
END = "<!-- PUBLIC_VALIDATION_SUMMARY_END -->"


def read_json(name: str) -> Optional[Dict[str, Any]]:
    path = OUTPUTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def pct_reduction(old: Optional[float], new: Optional[float]) -> Optional[float]:
    if old is None or new is None or old == 0:
        return None
    return (old - new) / old * 100.0


def get_path(obj: Optional[Dict[str, Any]], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def table_row(module: str, dataset: str, status: str, evidence: str) -> str:
    return f"| {module} | {dataset} | {status} | {evidence} |"


def build_summary() -> tuple[Dict[str, Any], str]:
    public_summary = read_json("public_validation_summary.json") or {}
    deepfm = read_json("criteo_click_deepctr_gpu_v2_metrics.json")
    autoint = read_json("criteo_click_autoint_gpu_metrics.json")
    resbias = read_json("criteo_click_residual_bias_gpu_v2_metrics.json")
    esci_abc = read_json("esci_abc_reranker_v2_metrics.json")
    esci_ablation = read_json("esci_abc_reranker_v2_ablation_metrics.json")

    rows = []
    highlights = []

    for m in public_summary.get("modules", []):
        rows.append(table_row(
            str(m.get("module", "unknown")),
            str(m.get("dataset", "unknown")),
            str(m.get("status", "UNKNOWN")),
            str(m.get("primary_result", "")),
        ))

    lr = get_path(deepfm, "same_split_lr_baseline", "test_metrics") or {}
    ens = get_path(deepfm, "selected_lr_deep_ensemble", "test_metrics") or {}
    deepfm_status = get_path(deepfm, "release_decision", "status") or get_path(deepfm, "release_decision", "overall_status") or "UNKNOWN"
    if lr and ens:
        ece_red = pct_reduction(lr.get("ece"), ens.get("ece"))
        rows.append(table_row(
            "GPU DeepCTR / DeepFM v2",
            "Criteo Click Logs public sample",
            str(deepfm_status),
            (
                f"LR+DeepFM ensemble: ROC-AUC {fmt(lr.get('roc_auc'), 4)} to {fmt(ens.get('roc_auc'), 4)}, "
                f"PR-AUC {fmt(lr.get('pr_auc'), 4)} to {fmt(ens.get('pr_auc'), 4)}, "
                f"log loss {fmt(lr.get('log_loss'), 5)} to {fmt(ens.get('log_loss'), 5)}, "
                f"ECE {fmt(lr.get('ece'), 5)} to {fmt(ens.get('ece'), 5)}"
                + (f" ({ece_red:.1f}% reduction)." if ece_red is not None else ".")
            ),
        ))
        highlights.append("GPU DeepFM v2 is the current CTR champion: a validation-selected LR+DeepFM ensemble improved ranking and calibration over the same-split LR baseline.")

    auto_lr = get_path(autoint, "same_split_lr_baseline", "test_metrics") or {}
    auto_ens = get_path(autoint, "selected_lr_autoint_ensemble", "test_metrics") or {}
    auto_status = get_path(autoint, "release_decision", "status") or "UNKNOWN"
    if auto_lr and auto_ens:
        rows.append(table_row(
            "GPU AutoInt attention CTR",
            "Criteo Click Logs public sample",
            str(auto_status),
            (
                f"LR+AutoInt ensemble: ROC-AUC {fmt(auto_lr.get('roc_auc'), 4)} to {fmt(auto_ens.get('roc_auc'), 4)}, "
                f"PR-AUC {fmt(auto_lr.get('pr_auc'), 4)} to {fmt(auto_ens.get('pr_auc'), 4)}, "
                f"ECE {fmt(auto_lr.get('ece'), 5)} to {fmt(auto_ens.get('ece'), 5)}; held at ITERATE versus DeepFM champion."
            ),
        ))

    res_lr = get_path(resbias, "same_split_lr_baseline", "test_metrics") or {}
    res = get_path(resbias, "selected_lambda_temperature_candidate", "test_metrics") or {}
    res_status = get_path(resbias, "release_decision", "overall_status") or get_path(resbias, "release_decision", "lr_gate_status") or "UNKNOWN"
    if res_lr and res:
        ece_red = pct_reduction(res_lr.get("ece"), res.get("ece"))
        rows.append(table_row(
            "GPU residual-bias CTR v2",
            "Criteo Click Logs public sample",
            str(res_status),
            (
                f"Paper-inspired LR-offset residual logit correction: ROC-AUC {fmt(res_lr.get('roc_auc'), 4)} to {fmt(res.get('roc_auc'), 4)}, "
                f"PR-AUC {fmt(res_lr.get('pr_auc'), 4)} to {fmt(res.get('pr_auc'), 4)}, "
                f"log loss {fmt(res_lr.get('log_loss'), 5)} to {fmt(res.get('log_loss'), 5)}, "
                f"ECE {fmt(res_lr.get('ece'), 5)} to {fmt(res.get('ece'), 5)}"
                + (f" ({ece_red:.1f}% reduction); passed LR gate but not DeepFM champion upgrade." if ece_red is not None else ".")
            ),
        ))
        highlights.append("Residual-bias CTR v2 validates the paper-inspired bias-correction specialty against LR, but DeepFM v2 remains the CTR champion.")

    base_test = get_path(esci_abc, "baseline_test") or {}
    abc_test = get_path(esci_abc, "abc_test") or {}
    abc_status = get_path(esci_abc, "release_decision", "status") or "UNKNOWN"
    if base_test and abc_test:
        rows.append(table_row(
            "ABC-style ESCI reranker v2",
            "Amazon ESCI public sample",
            str(abc_status),
            (
                f"NDCG@10 {fmt(base_test.get('ndcg_at_10'), 4)} to {fmt(abc_test.get('ndcg_at_10'), 4)}, "
                f"Recall@10 {fmt(base_test.get('recall_at_10'), 4)} to {fmt(abc_test.get('recall_at_10'), 4)}, "
                f"Precision@10 {fmt(base_test.get('precision_at_10'), 4)} to {fmt(abc_test.get('precision_at_10'), 4)}, "
                f"high-rank irrelevant {fmt(base_test.get('high_rank_irrelevant_rate'), 4)} to {fmt(abc_test.get('high_rank_irrelevant_rate'), 4)}."
            ),
        ))
        highlights.append("ABC-style ESCI reranker v2 is the strongest specialty result: it passed NDCG, recall, precision, and irrelevant-rate guardrails.")

    best = get_path(esci_ablation, "best_variant_by_guardrail_score") or {}
    variants = esci_ablation.get("variants", []) if esci_ablation else []
    expected_gain = next((v for v in variants if v.get("variant") == "expected_gain_only"), None)
    expansion_only = next((v for v in variants if v.get("variant") == "expansion_only"), None)
    if best:
        bm = best.get("test_metrics", {})
        bp = best.get("parameters", {})
        rows.append(table_row(
            "ABC-style ESCI ablation",
            "Amazon ESCI public sample",
            "EVIDENCE_STRENGTHENED",
            (
                f"Best variant {best.get('variant')} with alpha={bp.get('alpha_expansion')}, beta={bp.get('beta_expected_gain')}, "
                f"gamma={bp.get('gamma_relevance')}, delta={bp.get('delta_overreach')}: "
                f"NDCG@10 {fmt(bm.get('ndcg_at_10'), 4)}, Recall@10 {fmt(bm.get('recall_at_10'), 4)}, "
                f"high-rank irrelevant {fmt(bm.get('high_rank_irrelevant_rate'), 4)}."
            ),
        ))
        highlights.append("Ablation shows learned expected-gain scoring is the primary driver; expansion alone is unsafe because it improves NDCG but hurts recall and irrelevant-rate guardrails.")

    ablation_notes = ""
    if expected_gain and expansion_only:
        eg = expected_gain.get("test_metrics", {})
        ex = expansion_only.get("test_metrics", {})
        ablation_notes = f"""
### ESCI ABC ablation interpretation

The ablation strengthens the ABC-style ESCI result by separating the components.

- `expected_gain_only` passed all guardrails and was the main driver: NDCG@10 {fmt(eg.get("ndcg_at_10"))}, Recall@10 {fmt(eg.get("recall_at_10"))}, Precision@10 {fmt(eg.get("precision_at_10"))}, high-rank irrelevant {fmt(eg.get("high_rank_irrelevant_rate"))}.
- `expansion_only` improved NDCG@10 to {fmt(ex.get("ndcg_at_10"))}, but failed recall and increased high-rank irrelevant rate, so expansion alone is not safe.
- The promoted `expansion_plus_expected_gain` variant preserved all guardrails and achieved the best guardrail score.
- Explicit overreach penalty was evaluated, but not selected as the promoted scoring term; it is better framed as a release guardrail rather than the main performance driver.
"""

    summary_md = f"""# Public Validation Summary

## Claim boundary

This repository uses local public/proxy datasets and synthetic fallback only. It does not claim use of proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Current status

| Module | Dataset | Status | Evidence |
|---|---|---|---|
{chr(10).join(rows)}

## Key conclusions

{chr(10).join(f"- {h}" for h in highlights) if highlights else "- No advanced validation outputs found yet."}
{ablation_notes}
## Resume-safe claim language

- Built a local-runnable ads applied-science framework covering CTR prediction, calibration, product-query relevance, uplift, OPE, and release-gated model validation on public/proxy datasets.
- Added GPU-trained DeepFM and AutoInt CTR experiments; promoted a validation-selected LR+DeepFM ensemble as the CTR champion under ranking and calibration gates.
- Adapted paper-inspired residual-bias correction to CTR; passed the LR baseline gate by improving ROC-AUC, PR-AUC, log loss, and ECE, while preserving DeepFM v2 as champion.
- Adapted ABC-style expansion/scoring to Amazon ESCI product-query ranking; ablation showed learned expected-gain scoring was the primary driver and expansion alone was unsafe.

## Do-not-overclaim boundaries

- Do not claim production Amazon Ads deployment.
- Do not claim all public-data challengers beat all baselines.
- Do not claim AutoInt beat DeepFM.
- Do not claim explicit ESCI overreach penalty was the promoted performance driver.
"""

    summary_json = {
        "overall_status": "PUBLIC_VALIDATION_SUMMARY_UPDATED",
        "claim_boundary": "Public-data validation uses local public/proxy datasets and synthetic fallback only. No proprietary Amazon Ads data, advertiser data, customer data, production traffic, or deployment is claimed.",
        "highlights": highlights,
        "available_outputs_used": {
            "public_validation_summary": bool(public_summary),
            "deepfm_v2": deepfm is not None,
            "autoint": autoint is not None,
            "residual_bias_v2": resbias is not None,
            "esci_abc_v2": esci_abc is not None,
            "esci_abc_ablation": esci_ablation is not None,
        },
    }
    return summary_json, summary_md


def update_readme(summary_block: str) -> None:
    if README.exists():
        text = README.read_text(encoding="utf-8")
    else:
        text = "# Reliability-Aware Ads Applied Science Framework\n\n"

    block = f"{START}\n{summary_block}\n{END}"

    if START in text and END in text:
        pre = text.split(START)[0].rstrip()
        post = text.split(END, 1)[1].lstrip()
        new_text = f"{pre}\n\n{block}\n\n{post}".rstrip() + "\n"
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"

    README.write_text(new_text, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)

    summary_json, summary_md = build_summary()
    (REPORTS / "public_validation_summary.md").write_text(summary_md, encoding="utf-8")
    (REPORTS / "final_model_evidence_summary.md").write_text(summary_md, encoding="utf-8")
    (OUTPUTS / "public_validation_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    update_readme(summary_md)

    print("Updated:")
    print(REPORTS / "public_validation_summary.md")
    print(REPORTS / "final_model_evidence_summary.md")
    print(OUTPUTS / "public_validation_summary.json")
    print(README)


if __name__ == "__main__":
    main()
