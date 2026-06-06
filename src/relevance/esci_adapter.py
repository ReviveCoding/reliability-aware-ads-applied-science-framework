import pandas as pd

LABEL_MAP = {"E": 3, "S": 2, "C": 1, "I": 0, "Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}


def load_esci(path, nrows=None):
    """Load an Amazon ESCI Shopping Queries CSV into the relevance module schema.

    The official dataset has evolved across releases. This adapter accepts common
    column names and normalizes them into: query_id, query, product_id, title,
    query_category, product_category, esci_label_num, is_long_tail.
    """
    df = pd.read_csv(path, nrows=nrows)
    lower = {c.lower(): c for c in df.columns}
    query_col = lower.get("query") or lower.get("query_text")
    title_col = lower.get("product_title") or lower.get("title")
    label_col = lower.get("esci_label") or lower.get("label") or lower.get("relevance")
    product_col = lower.get("product_id") or lower.get("product_id_locale") or lower.get("asin")
    if query_col is None or title_col is None or label_col is None:
        raise ValueError("ESCI file must include query, product title, and ESCI relevance label columns.")
    out = pd.DataFrame()
    out["query"] = df[query_col].astype(str)
    out["title"] = df[title_col].astype(str)
    out["query_id"] = df[lower.get("query_id")].astype(str) if lower.get("query_id") else out["query"].factorize()[0].astype(str)
    out["product_id"] = df[product_col].astype(str) if product_col else range(len(df))
    if lower.get("product_category"):
        out["product_category"] = df[lower.get("product_category")].astype(str)
    else:
        out["product_category"] = "unknown"
    out["query_category"] = out["product_category"]
    out["query_modifier"] = out["query"].str.split().str[0].fillna("")
    out["brand"] = "unknown"
    out["is_long_tail"] = False
    out["esci_label_num"] = df[label_col].map(LABEL_MAP).fillna(pd.to_numeric(df[label_col], errors="coerce")).fillna(0).astype(int)
    return out
