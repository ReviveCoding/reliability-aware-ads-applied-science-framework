import numpy as np
import pandas as pd


def fit_beta_binomial_prior(train_df, target_col="click"):
    rate = float(train_df[target_col].mean())
    # Effective prior strength: conservative but not overwhelming.
    strength = max(20.0, min(200.0, len(train_df) * 0.02))
    alpha = max(rate * strength, 1e-3)
    beta = max((1 - rate) * strength, 1e-3)
    return {"global_rate": rate, "strength": strength, "alpha": alpha, "beta": beta}


def group_posterior_table(train_df, group_col="campaign_id", target_col="click", prior=None):
    if prior is None:
        prior = fit_beta_binomial_prior(train_df, target_col)
    g = train_df.groupby(group_col)[target_col].agg(["sum", "count"]).reset_index()
    g["posterior_mean"] = (g["sum"] + prior["alpha"]) / (g["count"] + prior["alpha"] + prior["beta"])
    g["uncertainty"] = 1 / np.sqrt(g["count"] + prior["strength"])
    return g[[group_col, "posterior_mean", "uncertainty", "count"]], prior


def apply_empirical_bayes(test_df, raw_prob_col="pctr_raw", group_col="campaign_id", posterior_table=None, prior=None, output_col="pctr_eb"):
    df = test_df.copy()
    if posterior_table is None or prior is None:
        raise ValueError("posterior_table and prior are required")
    df = df.merge(posterior_table, on=group_col, how="left")
    df["posterior_mean"] = df["posterior_mean"].fillna(prior["global_rate"])
    df["uncertainty"] = df["uncertainty"].fillna(1 / np.sqrt(prior["strength"]))
    df["count"] = df["count"].fillna(0)
    shrink_weight = df["count"] / (df["count"] + prior["strength"])
    df[output_col] = (shrink_weight * df[raw_prob_col] + (1 - shrink_weight) * df["posterior_mean"]).clip(1e-6, 1-1e-6)
    df["eb_uncertainty"] = df["uncertainty"]
    df["eb_uncertainty_penalty"] = (1 - 0.5 * df["uncertainty"]).clip(0.5, 1.0)
    return df


def apply_hierarchical_empirical_bayes(test_df, raw_prob_col="pctr_raw", campaign_table=None, category_table=None, prior=None):
    """Blend campaign-level and category-level EB posteriors.

    This provides a stronger sparse-slice fallback than campaign-only smoothing.
    The output is intended for release-gate use when sparse campaign ECE is high.
    """
    if prior is None or campaign_table is None or category_table is None:
        raise ValueError("prior, campaign_table, and category_table are required")
    df = test_df.copy()
    ctab = campaign_table.rename(columns={
        "posterior_mean": "campaign_posterior",
        "uncertainty": "campaign_uncertainty",
        "count": "campaign_count",
    })
    gtab = category_table.rename(columns={
        "posterior_mean": "category_posterior",
        "uncertainty": "category_uncertainty",
        "count": "category_count",
    })
    df = df.merge(ctab, on="campaign_id", how="left").merge(gtab, on="category", how="left")
    for col in ["campaign_posterior", "category_posterior"]:
        df[col] = df[col].fillna(prior["global_rate"])
    for col in ["campaign_count", "category_count"]:
        df[col] = df[col].fillna(0)
    # Sparse campaigns lean more on category posterior and global prior.
    campaign_w = df["campaign_count"] / (df["campaign_count"] + prior["strength"])
    category_w = (1 - campaign_w) * (df["category_count"] / (df["category_count"] + prior["strength"]))
    raw_w = 0.20 * campaign_w
    posterior_w = (campaign_w - raw_w).clip(0, 1)
    global_w = (1 - raw_w - posterior_w - category_w).clip(0, 1)
    total = raw_w + posterior_w + category_w + global_w
    df["pctr_eb_hier"] = (
        raw_w / total * df[raw_prob_col]
        + posterior_w / total * df["campaign_posterior"]
        + category_w / total * df["category_posterior"]
        + global_w / total * prior["global_rate"]
    ).clip(1e-6, 1 - 1e-6)
    combined_unc = 1 / np.sqrt(df["campaign_count"] + df["category_count"] + prior["strength"])
    df["eb_hier_uncertainty_penalty"] = (1 - 0.5 * combined_unc).clip(0.5, 1.0)
    return df
