import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_ads(n_rows=2500, n_products=160, n_queries=60, seed=42):
    rng = np.random.default_rng(seed)
    n_campaigns = max(30, n_products // 3)
    n_advertisers = max(12, n_products // 12)
    categories = np.array(["electronics", "home", "beauty", "grocery", "books", "fashion", "sports", "auto"])

    df = pd.DataFrame({
        "impression_id": np.arange(n_rows),
        "timestamp": rng.integers(0, 30, size=n_rows),
        "campaign_id": rng.integers(0, n_campaigns, size=n_rows).astype(str),
        "advertiser_id": rng.integers(0, n_advertisers, size=n_rows).astype(str),
        "product_id": rng.integers(0, n_products, size=n_rows).astype(str),
        "query_id": rng.integers(0, n_queries, size=n_rows).astype(str),
        "category": rng.choice(categories, size=n_rows),
        "position": rng.integers(1, 6, size=n_rows),
        "bid": rng.gamma(shape=2.2, scale=0.7, size=n_rows).clip(0.05, 8.0),
    })
    product_quality = rng.normal(0, 0.7, size=n_products)
    advertiser_quality = rng.normal(0, 0.4, size=n_advertisers)
    campaign_bias = rng.normal(0, 0.5, size=n_campaigns)
    category_effect = {c: v for c, v in zip(categories, rng.normal(0, 0.25, size=len(categories)))}
    df["relevance"] = rng.beta(2.5, 2.0, size=n_rows)
    # Add some deliberately low relevance impressions with high bids to test guardrails.
    low_rel_idx = rng.choice(df.index, size=max(1, n_rows // 10), replace=False)
    df.loc[low_rel_idx, "relevance"] = rng.beta(0.8, 5.0, size=len(low_rel_idx))
    df.loc[low_rel_idx, "bid"] *= rng.uniform(1.5, 3.0, size=len(low_rel_idx))

    p = (
        -3.0
        + 0.9 * df["relevance"].values
        + 0.18 * np.log1p(df["bid"].values)
        - 0.24 * (df["position"].values - 1)
        + np.array([product_quality[int(x)] for x in df["product_id"]])
        + np.array([advertiser_quality[int(x)] for x in df["advertiser_id"]])
        + np.array([campaign_bias[int(x)] for x in df["campaign_id"]])
        + np.array([category_effect[x] for x in df["category"]])
        + 0.025 * df["timestamp"].values
    )
    true_pctr = sigmoid(p)
    df["true_pctr"] = true_pctr
    df["click"] = rng.binomial(1, true_pctr)
    pcvr = sigmoid(-2.2 + 1.3 * df["relevance"].values + 0.8 * df["click"].values + 0.15 * np.log1p(df["bid"].values))
    df["true_pcvr"] = pcvr
    df["conversion"] = rng.binomial(1, pcvr * np.maximum(df["click"].values, rng.binomial(1, 0.15, n_rows)))
    df["revenue"] = df["conversion"] * rng.lognormal(mean=2.6, sigma=0.5, size=n_rows)
    df["is_long_tail_advertiser"] = df["advertiser_id"].astype(int) >= int(n_advertisers * 0.65)
    return df
