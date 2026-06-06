import pandas as pd
from .base import BaseAdapter, AdapterResult
LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0, "Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}

class ESCIAdapter(BaseAdapter):
    name = "amazon_esci"
    def load_raw(self, path, nrows=None):
        return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, nrows=nrows)
    def validate_raw(self, df):
        problems = []
        if not {"query", "query_text"}.intersection(df.columns): problems.append("Missing query/query_text column.")
        if not {"product_title", "title"}.intersection(df.columns): problems.append("Missing product_title/title column.")
        if not {"esci_label", "label"}.intersection(df.columns): problems.append("Missing esci_label/label column.")
        return {"status": "PASS" if not problems else "FAIL", "rows": len(df), "problems": problems}
    def to_standard_schema(self, df):
        out = pd.DataFrame()
        out["query_id"] = df.get("query_id", pd.Series(range(len(df)))).astype(str)
        out["query"] = df["query"] if "query" in df.columns else df["query_text"]
        out["title"] = df["product_title"] if "product_title" in df.columns else df["title"]
        label = df["esci_label"] if "esci_label" in df.columns else df["label"]
        out["esci_label_num"] = label.map(LABEL_MAP).fillna(label).astype(int)
        out["product_id"] = df.get("product_id", pd.Series(range(len(df)))).astype(str)
        out["product_category"] = df.get("product_category", "unknown")
        out["query_category"] = out["product_category"]
        out["brand"] = df.get("product_brand", "unknown")
        out["is_long_tail"] = False
        return out

def try_load(path, nrows=100000):
    adapter = ESCIAdapter(); df = adapter.load_raw(path, nrows=nrows); v = adapter.validate_raw(df)
    if v["status"] != "PASS": return AdapterResult(adapter.name, "FAIL", len(df), str(v["problems"]), None)
    return AdapterResult(adapter.name, "PASS", len(df), "Loaded ESCI sample.", adapter.to_standard_schema(df))
