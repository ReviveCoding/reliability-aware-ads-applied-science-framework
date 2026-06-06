import pandas as pd
from .base import BaseAdapter, AdapterResult

class CriteoUpliftAdapter(BaseAdapter):
    name = "criteo_uplift"
    def load_raw(self, path, nrows=None):
        return pd.read_csv(path, nrows=nrows)
    def validate_raw(self, df):
        required = {"treatment", "conversion", "visit"}
        missing = sorted(required - set(df.columns))
        feature_cols = [c for c in df.columns if c.startswith("f")]
        problems = []
        if missing: problems.append(f"Missing required columns: {missing}")
        if len(feature_cols) < 5: problems.append("Expected feature columns named f0, f1, ...")
        return {"status": "PASS" if not problems else "FAIL", "rows": len(df), "feature_count": len(feature_cols), "problems": problems}
    def to_standard_schema(self, df):
        feature_cols = [c for c in df.columns if c.startswith("f")]
        out = df[feature_cols + ["treatment", "conversion"]].copy()
        return out.rename(columns={"conversion": "outcome"})

def try_load(path, nrows=100000):
    adapter = CriteoUpliftAdapter(); df = adapter.load_raw(path, nrows=nrows); v = adapter.validate_raw(df)
    if v["status"] != "PASS": return AdapterResult(adapter.name, "FAIL", len(df), str(v["problems"]), None)
    return AdapterResult(adapter.name, "PASS", len(df), "Loaded Criteo uplift sample.", adapter.to_standard_schema(df))
