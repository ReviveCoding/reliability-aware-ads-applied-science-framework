"""Guardrail regressions: incomplete evidence must not produce PASS."""

import pandas as pd

from src.policy.release_gate import decide_release


def _inputs():
    metrics = {"calibration": {"eb": {"ece": 0.05}}}
    slice_summary = {"worst_ece": 0.06, "worst_slice_n": 60}
    policy_df = pd.DataFrame([{
        "policy": "score_reliability",
        "false_high_rank_irrelevant_rate": 0.10,
        "long_tail_exposure": 0.20,
    }])
    return metrics, slice_summary, policy_df


def test_complete_evidence_can_pass():
    metrics, slices, policies = _inputs()
    result = decide_release(metrics, policies, slices)
    assert result["status"] == "PASS"


def test_missing_or_invalid_calibration_evidence_blocks():
    for value in (None, float("nan"), float("inf"), -0.01, 1.01):
        metrics, slices, policies = _inputs()
        metrics["calibration"]["eb"]["ece"] = value
        assert decide_release(metrics, policies, slices)["status"] == "BLOCK"
    _, slices, policies = _inputs()
    assert decide_release({}, policies, slices)["status"] == "BLOCK"


def test_missing_or_invalid_slice_evidence_blocks():
    for value in (None, float("nan"), float("inf"), -0.1, 1.1):
        metrics, slices, policies = _inputs()
        slices["worst_ece"] = value
        assert decide_release(metrics, policies, slices)["status"] == "BLOCK"
    for value in (None, 0, -1, 1.5, float("nan")):
        metrics, slices, policies = _inputs()
        slices["worst_slice_n"] = value
        assert decide_release(metrics, policies, slices)["status"] == "BLOCK"


def test_missing_or_invalid_policy_evidence_blocks():
    metrics, slices, policies = _inputs()
    assert decide_release(metrics, policies.iloc[0:0], slices)["status"] == "BLOCK"
    for key in ("false_high_rank_irrelevant_rate", "long_tail_exposure"):
        for value in (None, float("nan"), float("inf"), -0.2, 1.2):
            metrics, slices, policies = _inputs()
            policies.loc[0, key] = value
            assert decide_release(metrics, policies, slices)["status"] == "BLOCK"


def test_valid_threshold_violation_retains_iterate_semantics():
    metrics, slices, policies = _inputs()
    metrics["calibration"]["eb"]["ece"] = 0.13
    assert decide_release(metrics, policies, slices)["status"] == "ITERATE"
