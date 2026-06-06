import numpy as np
import pandas as pd


def add_policy_scores(df):
    out = df.copy()
    out["score_bid_only"] = out["bid"]
    out["score_raw"] = out["bid"] * out.get("pctr_raw", out.get("true_pctr", 0.05))
    out["score_eb"] = out["bid"] * out.get("pctr_eb", out.get("pctr_raw", 0.05))
    out["score_reliability"] = (
        out["bid"]
        * out.get("pctr_eb", out.get("pctr_raw", 0.05))
        * (0.5 + out.get("true_pcvr", out.get("conversion", 0.02)))
        * (0.3 + out["relevance"])
        * out.get("eb_uncertainty_penalty", 1.0)
    )
    return out


def evaluate_ranking_policy(df, score_col, top_fraction=0.2):
    k = max(1, int(len(df) * top_fraction))
    selected = df.sort_values(score_col, ascending=False).head(k)
    advertiser_share = selected["advertiser_id"].value_counts(normalize=True)
    long_tail_exposure = float(selected.get("is_long_tail_advertiser", False).mean())
    false_high_rank_irrelevant_rate = float((selected["relevance"] < 0.25).mean())
    spend = float(selected["bid"].sum())
    revenue = float(selected.get("revenue", 0).sum())
    return {
        "policy": score_col,
        "selected_impressions": int(k),
        "clicks": int(selected["click"].sum()),
        "conversions": int(selected["conversion"].sum()),
        "expected_clicks": float(selected.get("pctr_eb", selected.get("pctr_raw", 0)).sum()),
        "revenue": revenue,
        "spend_proxy": spend,
        "iroas_proxy": revenue / max(spend, 1e-6),
        "advertiser_concentration_top1": float(advertiser_share.iloc[0]) if len(advertiser_share) else 0.0,
        "long_tail_exposure": long_tail_exposure,
        "false_high_rank_irrelevant_rate": false_high_rank_irrelevant_rate,
    }


def compare_policies(df, top_fraction=0.2):
    df = add_policy_scores(df)
    policies = ["score_bid_only", "score_raw", "score_eb", "score_reliability"]
    return pd.DataFrame([evaluate_ranking_policy(df, p, top_fraction) for p in policies])
