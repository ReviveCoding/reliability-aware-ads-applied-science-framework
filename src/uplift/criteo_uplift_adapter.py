import pandas as pd


def load_criteo_uplift(path, nrows=None, target_col="conversion"):
    """Load a local Criteo Uplift file into the uplift module schema.

    Expected source columns are usually f0..f11 plus treatment, conversion, and visit.
    The adapter is intentionally lightweight and works with .csv or .csv.gz files.
    """
    df = pd.read_csv(path, nrows=nrows)
    col_map = {c.lower(): c for c in df.columns}
    treatment = col_map.get("treatment") or col_map.get("exposure")
    conversion = col_map.get(target_col) or col_map.get("conversion") or col_map.get("visit") or col_map.get("visits")
    if treatment is None or conversion is None:
        raise ValueError("Criteo uplift file must include treatment/exposure and conversion/visit columns.")
    feature_cols = [c for c in df.columns if c.lower().startswith("f")]
    out = df[feature_cols + [treatment, conversion]].copy()
    out = out.rename(columns={treatment: "treatment", conversion: "conversion"})
    # The smoke T-learner uses x1/x2/x3, so expose a compact normalized view too.
    for i, c in enumerate(feature_cols[:3], start=1):
        out[f"x{i}"] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "x1" not in out:
        raise ValueError("No feature columns found. Expected f0..fN style Criteo columns.")
    out["treatment"] = out["treatment"].astype(int)
    out["conversion"] = out["conversion"].astype(int)
    return out
