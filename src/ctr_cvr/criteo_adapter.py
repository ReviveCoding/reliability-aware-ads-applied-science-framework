import pandas as pd


def load_criteo_click_sample(path, nrows=None):
    """Load a local Criteo-style click log into the standard ads schema.

    This adapter supports CSV/CSV.GZ files with either explicit named columns or
    common Criteo-style anonymous columns. It creates synthetic campaign/product
    identifiers when source columns are not available.
    """
    df = pd.read_csv(path, nrows=nrows)
    lower = {c.lower(): c for c in df.columns}
    click_col = lower.get("click") or lower.get("label") or df.columns[0]
    out = pd.DataFrame()
    out["impression_id"] = range(len(df))
    out["timestamp"] = df[lower.get("day")].astype(int) if lower.get("day") else (out["impression_id"] // 10000)
    out["campaign_id"] = df[lower.get("campaign_id")].astype(str) if lower.get("campaign_id") else (out["impression_id"] % 50).astype(str)
    out["advertiser_id"] = df[lower.get("advertiser_id")].astype(str) if lower.get("advertiser_id") else (out["impression_id"] % 20).astype(str)
    out["product_id"] = df[lower.get("product_id")].astype(str) if lower.get("product_id") else (out["impression_id"] % 200).astype(str)
    out["query_id"] = (out["impression_id"] % 80).astype(str)
    out["category"] = "criteo_proxy"
    out["bid"] = 1.0
    out["position"] = 1
    out["relevance"] = 0.5
    out["click"] = pd.to_numeric(df[click_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["conversion"] = 0
    out["revenue"] = 0.0
    out["true_pctr"] = out["click"].rolling(100, min_periods=1).mean().shift(1).fillna(out["click"].mean())
    out["true_pcvr"] = 0.02
    out["is_long_tail_advertiser"] = False
    return out
