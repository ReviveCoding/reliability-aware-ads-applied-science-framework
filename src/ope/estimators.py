import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.common.metrics import bootstrap_ci

def ips(df):
    w = df.candidate_prob / np.clip(df.pscore, 1e-6, 1); vals = w * df.reward
    return float(vals.mean()), vals.values

def snips(df):
    w = df.candidate_prob / np.clip(df.pscore, 1e-6, 1); vals = w * df.reward
    return float(vals.sum() / max(w.sum(), 1e-6)), vals.values / max(w.mean(), 1e-6)

def doubly_robust(df):
    features = df[["x1", "x2", "action"]]
    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42).fit(features, df.reward)
    q_logged = np.clip(model.predict(features), 0, 1); q_eval = q_logged
    w = df.candidate_prob / np.clip(df.pscore, 1e-6, 1)
    vals = q_eval + w * (df.reward - q_logged)
    return float(vals.mean()), vals

def overlap_diagnostics(df):
    w = df.candidate_prob / np.clip(df.pscore, 1e-6, 1)
    return {"effective_sample_size": float(w.sum()**2 / max((w**2).sum(), 1e-9)), "max_importance_weight": float(w.max()), "mean_importance_weight": float(w.mean()), "low_support_rate": float((df.candidate_prob < 0.05).mean())}

def ope_summary(df):
    out = {}
    for name, fn in [("ips", ips), ("snips", snips), ("doubly_robust", doubly_robust)]:
        value, vals = fn(df); lo, hi = bootstrap_ci(vals, n_boot=50)
        out[name] = {"value": value, "ci_low": lo, "ci_high": hi, "ci_width": hi-lo}
    out["diagnostics"] = overlap_diagnostics(df)
    return out
