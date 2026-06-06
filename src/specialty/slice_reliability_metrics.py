import pandas as pd
from src.common.metrics import expected_calibration_error, safe_log_loss


def compute_slice_metrics(df, slice_col="category", target_col="click", prob_col="pctr_eb"):
    rows = []
    for val, g in df.groupby(slice_col):
        if len(g) < 10:
            continue
        ece = expected_calibration_error(g[target_col], g[prob_col])[0]
        rows.append({
            "slice_col": slice_col,
            "slice_value": val,
            "n": int(len(g)),
            "positive_rate": float(g[target_col].mean()),
            "log_loss": safe_log_loss(g[target_col], g[prob_col]),
            "ece": ece,
        })
    return pd.DataFrame(rows)


def worst_slice_summary(slice_df):
    if slice_df.empty:
        return {"worst_slice": None, "worst_ece": None}
    row = slice_df.sort_values("ece", ascending=False).iloc[0]
    return {
        "worst_slice": f"{row['slice_col']}={row['slice_value']}",
        "worst_ece": float(row["ece"]),
        "worst_slice_n": int(row["n"]),
    }
