import pandas as pd
from src.policy.release_gate import decide_release


def test_release_gate_returns_status():
    metrics = {"calibration": {"eb": {"ece": 0.05}}}
    policy_df = pd.DataFrame([
        {"policy": "score_reliability", "false_high_rank_irrelevant_rate": 0.1, "long_tail_exposure": 0.2}
    ])
    slice_summary = {"worst_ece": 0.05}
    d = decide_release(metrics, policy_df, slice_summary)
    assert d["status"] in {"PASS", "ITERATE", "BLOCK"}
