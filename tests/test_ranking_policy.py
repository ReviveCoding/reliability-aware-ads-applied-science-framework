from src.ctr_cvr.synthetic_ads_generator import generate_synthetic_ads
from src.ctr_cvr.feature_builder import build_features
from src.policy.auction_ranking import compare_policies


def test_policy_comparison_outputs_rows():
    df = build_features(generate_synthetic_ads(n_rows=300, seed=3))
    df["pctr_raw"] = df["true_pctr"]
    df["pctr_eb"] = df["true_pctr"]
    df["eb_uncertainty_penalty"] = 1.0
    out = compare_policies(df)
    assert len(out) >= 3
    assert "iroas_proxy" in out.columns
