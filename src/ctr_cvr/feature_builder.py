import numpy as np
import pandas as pd


CATEGORICAL_COLS = ["campaign_id", "advertiser_id", "product_id", "query_id", "category"]
NUMERIC_COLS = [
    "bid", "position", "relevance", "timestamp", "campaign_age",
    "cum_impressions", "cum_clicks", "cum_conversions", "click_velocity",
    "conversion_maturity", "sparse_history_score", "lifecycle_state_index",
    "expected_pctr"
]


def add_lifecycle_features(df):
    df = df.sort_values(["campaign_id", "timestamp", "impression_id"]).copy()
    df["campaign_age"] = df.groupby("campaign_id")["timestamp"].transform(lambda x: x - x.min() + 1)
    df["cum_impressions"] = df.groupby("campaign_id").cumcount()
    df["cum_clicks"] = df.groupby("campaign_id")["click"].cumsum().shift(1).fillna(0)
    df["cum_conversions"] = df.groupby("campaign_id")["conversion"].cumsum().shift(1).fillna(0)
    df["click_velocity"] = df["cum_clicks"] / (df["cum_impressions"] + 1)
    df["conversion_maturity"] = np.log1p(df["cum_conversions"]) / np.log1p(df["cum_impressions"] + 2)
    df["sparse_history_score"] = 1 / np.sqrt(df["cum_impressions"] + 1)
    df["lifecycle_state_index"] = (
        0.35 * np.log1p(df["cum_impressions"]) / np.log1p(df["cum_impressions"].max() + 1)
        + 0.35 * df["click_velocity"].clip(0, 1)
        + 0.20 * df["conversion_maturity"].clip(0, 1)
        - 0.10 * df["sparse_history_score"].clip(0, 1)
    )
    df["lifecycle_state_index"] = df["lifecycle_state_index"].fillna(0).clip(0, 1)
    return df


def add_expected_pctr(df):
    cat_prior = df.groupby("category")["click"].transform("mean")
    pos_prior = 0.09 * np.exp(-0.28 * (df["position"] - 1))
    relevance_prior = 0.03 + 0.12 * df["relevance"]
    maturity_prior = 0.02 * df.get("lifecycle_state_index", 0)
    expected = 0.35 * cat_prior.fillna(df["click"].mean()) + 0.30 * pos_prior + 0.25 * relevance_prior + 0.10 * maturity_prior
    df = df.copy()
    df["expected_pctr"] = expected.clip(1e-4, 0.8)
    return df


def build_features(df):
    df = add_lifecycle_features(df)
    df = add_expected_pctr(df)
    return df


def feature_columns():
    return CATEGORICAL_COLS, NUMERIC_COLS
