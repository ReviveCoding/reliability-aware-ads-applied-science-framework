import numpy as np


def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, 1)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_summary(train_df, test_df, cols=("bid", "relevance", "lifecycle_state_index")):
    rows = []
    for c in cols:
        if c in train_df.columns and c in test_df.columns:
            rows.append({"feature": c, "psi": psi(train_df[c], test_df[c])})
    return rows
