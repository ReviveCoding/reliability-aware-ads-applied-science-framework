"""Release qualification for calibration, slice reliability, and ranking policy."""

from __future__ import annotations

import math


def _finite_fraction(value) -> bool:
    """Reject missing, malformed, and non-finite probability/rate evidence."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _positive_sample_count(value) -> bool:
    if value is None or isinstance(value, (bool, str, bytes)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number >= 1 and number.is_integer()


def decide_release(metrics, policy_df, slice_summary, thresholds=None):
    thresholds = thresholds or {}
    max_ece = thresholds.get("max_ece", 0.12)
    max_sparse_ece = thresholds.get("max_sparse_ece", 0.18)
    max_false_rate = thresholds.get("max_false_high_rank_irrelevant_rate", 0.35)
    min_long_tail = thresholds.get("min_long_tail_exposure", 0.08)
    min_slice_n = thresholds.get("min_slice_n", 30)

    reasons = []
    actions = []
    missing_evidence = False

    ece = metrics.get("calibration", {}).get("eb", {}).get("ece")
    if not _finite_fraction(ece):
        missing_evidence = True
        reasons.append("EB calibration ECE is missing or invalid.")
        actions.append("Generate valid held-out calibration evidence.")
    elif ece > max_ece:
        reasons.append(f"EB calibration ECE {ece:.3f} exceeds threshold {max_ece:.3f}.")
        actions.append("Tune prior strength or add more calibration data.")

    worst_ece = slice_summary.get("worst_ece")
    worst_n = slice_summary.get("worst_slice_n")
    if not _finite_fraction(worst_ece) or not _positive_sample_count(worst_n):
        missing_evidence = True
        reasons.append("Worst-slice ECE or its sample count is missing or invalid.")
        actions.append("Collect and validate slice-level calibration evidence.")
    elif worst_ece > max_sparse_ece:
        if worst_n < min_slice_n:
            reasons.append(f"Worst-slice ECE {worst_ece:.3f} exceeds threshold {max_sparse_ece:.3f}, but slice support is small (n={worst_n}); collect more data or validate with public/sample mode.")
            actions.append("Increase sample size for weak slices before final release decision.")
        else:
            reasons.append(f"Worst-slice ECE {worst_ece:.3f} exceeds threshold {max_sparse_ece:.3f}.")
            actions.append("Apply stronger uncertainty penalty to weak slices.")

    if "policy" not in policy_df.columns:
        selected = policy_df.iloc[0:0]
    else:
        selected = policy_df[policy_df["policy"] == "score_reliability"]
    if selected.empty:
        missing_evidence = True
        reasons.append("The score_reliability policy row is missing.")
        actions.append("Run the candidate policy evaluation before release.")
        final_policy = {}
    else:
        final_policy = selected.iloc[0].to_dict()
        false_rate = final_policy.get("false_high_rank_irrelevant_rate")
        long_tail = final_policy.get("long_tail_exposure")

        if not _finite_fraction(false_rate):
            missing_evidence = True
            reasons.append("False high-rank irrelevant rate is missing or invalid.")
            actions.append("Recompute the ranking relevance guardrail.")
        elif false_rate > max_false_rate:
            reasons.append("False high-rank irrelevant rate is too high.")
            actions.append("Strengthen relevance guardrail and reranking penalty.")

        if not _finite_fraction(long_tail):
            missing_evidence = True
            reasons.append("Long-tail exposure is missing or invalid.")
            actions.append("Recompute marketplace exposure.")
        elif long_tail < min_long_tail:
            reasons.append("Long-tail exposure is below floor.")
            actions.append("Increase long-tail exposure constraint.")

    if missing_evidence:
        status = "BLOCK"
    elif not reasons:
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
