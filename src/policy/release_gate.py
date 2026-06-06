import json


def decide_release(metrics, policy_df, slice_summary, thresholds=None):
    thresholds = thresholds or {}
    max_ece = thresholds.get("max_ece", 0.12)
    max_sparse_ece = thresholds.get("max_sparse_ece", 0.18)
    max_false_rate = thresholds.get("max_false_high_rank_irrelevant_rate", 0.35)
    min_long_tail = thresholds.get("min_long_tail_exposure", 0.08)
    min_slice_n = thresholds.get("min_slice_n", 30)

    reasons = []
    actions = []

    ece = metrics.get("calibration", {}).get("eb", {}).get("ece")
    if ece is not None and ece > max_ece:
        reasons.append(f"EB calibration ECE {ece:.3f} exceeds threshold {max_ece:.3f}.")
        actions.append("Tune prior strength or add more calibration data.")

    worst_ece = slice_summary.get("worst_ece")
    worst_n = slice_summary.get("worst_slice_n")
    if worst_ece is not None and worst_ece > max_sparse_ece:
        if worst_n is not None and worst_n < min_slice_n:
            reasons.append(f"Worst-slice ECE {worst_ece:.3f} exceeds threshold {max_sparse_ece:.3f}, but slice support is small (n={worst_n}); collect more data or validate with public/sample mode.")
            actions.append("Increase sample size for weak slices before final release decision.")
        else:
            reasons.append(f"Worst-slice ECE {worst_ece:.3f} exceeds threshold {max_sparse_ece:.3f}.")
            actions.append("Apply stronger uncertainty penalty to weak slices.")

    final_policy = policy_df[policy_df["policy"] == "score_reliability"].iloc[0].to_dict()
    if final_policy["false_high_rank_irrelevant_rate"] > max_false_rate:
        reasons.append("False high-rank irrelevant rate is too high.")
        actions.append("Strengthen relevance guardrail and reranking penalty.")

    if final_policy["long_tail_exposure"] < min_long_tail:
        reasons.append("Long-tail exposure is below floor.")
        actions.append("Increase long-tail exposure constraint.")

    if not reasons:
        status = "PASS"
        reasons = ["All core calibration, slice reliability, relevance, and marketplace guardrails passed."]
    elif len(reasons) <= 2:
        status = "ITERATE"
    else:
        status = "BLOCK"

    return {
        "status": status,
        "reasons": reasons,
        "recommended_actions": actions or ["Proceed with broader sample/public-data validation."],
        "final_policy_snapshot": final_policy,
    }
