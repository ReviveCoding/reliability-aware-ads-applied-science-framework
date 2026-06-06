import pandas as pd


ADS_REQUIRED_COLUMNS = [
    "impression_id", "timestamp", "campaign_id", "advertiser_id", "product_id",
    "query_id", "category", "bid", "position", "relevance", "click", "conversion"
]


def validate_ads_schema(df):
    missing = [c for c in ADS_REQUIRED_COLUMNS if c not in df.columns]
    problems = []
    if missing:
        problems.append(f"Missing required columns: {missing}")
    if "click" in df.columns and not set(df["click"].unique()).issubset({0, 1}):
        problems.append("click must be binary")
    if "conversion" in df.columns and not set(df["conversion"].unique()).issubset({0, 1}):
        problems.append("conversion must be binary")
    if "bid" in df.columns and (df["bid"] < 0).any():
        problems.append("bid must be non-negative")
    return {
        "status": "PASS" if not problems else "FAIL",
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_columns": missing,
        "problems": problems,
    }


def leakage_check(train, test, id_cols=("impression_id",)):
    issues = []
    for col in id_cols:
        if col in train.columns and col in test.columns:
            overlap = set(train[col]).intersection(set(test[col]))
            if overlap:
                issues.append({"column": col, "overlap_count": len(overlap)})
    return {"status": "PASS" if not issues else "FAIL", "issues": issues}
