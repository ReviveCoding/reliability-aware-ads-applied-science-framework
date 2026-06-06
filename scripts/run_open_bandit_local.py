import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def find_bandit_file(data_dir):
    root = Path(data_dir)
    preferred = list(root.rglob("all.csv"))
    if preferred:
        return preferred[0]
    candidates = [p for p in root.rglob("*.csv") if p.name.lower() != "item_context.csv"]
    if not candidates:
        raise FileNotFoundError(f"No Open Bandit CSV found under {root}")
    return candidates[0]


def bootstrap_ci(values, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_open_bandit(data_dir, max_rows=300000):
    path = find_bandit_file(data_dir)
    df = pd.read_csv(path, nrows=max_rows)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"item_id", "click", "propensity_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns: {df.columns.tolist()}")

    df = df.dropna(subset=["item_id", "click", "propensity_score"]).copy()
    df["reward"] = pd.to_numeric(df["click"], errors="coerce").fillna(0).astype(float)
    df["pscore"] = pd.to_numeric(df["propensity_score"], errors="coerce").clip(1e-6, 1.0)
    df["item_id_int"] = pd.to_numeric(df["item_id"], errors="coerce").fillna(-1).astype(int)
    return df, path


def add_candidate_policy_probability(df):
    affinity_cols = [c for c in df.columns if c.startswith("user-item_affinity_")]
    df = df.copy()

    if not affinity_cols:
        df["candidate_prob"] = 0.5
        df["observed_affinity"] = 0.0
        return df

    aff = df[affinity_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scores = aff.values
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)

    observed = []
    cand_prob = []
    col_index = {c: i for i, c in enumerate(affinity_cols)}

    for i, item in enumerate(df["item_id_int"].values):
        col = f"user-item_affinity_{item}"
        if col in col_index:
            j = col_index[col]
            observed.append(float(aff.iloc[i, j]))
            cand_prob.append(float(probs[i, j]))
        else:
            observed.append(float(aff.iloc[i].mean()))
            cand_prob.append(float(1.0 / len(affinity_cols)))

    df["observed_affinity"] = observed
    df["candidate_prob"] = np.clip(cand_prob, 1e-6, 1.0)
    return df


def ips_values(df):
    w = df["candidate_prob"] / df["pscore"]
    return (w * df["reward"]).values


def snips_value_and_values(df):
    w = df["candidate_prob"] / df["pscore"]
    vals = w * df["reward"]
    value = vals.sum() / max(w.sum(), 1e-9)
    normalized_vals = vals / max(w.mean(), 1e-9)
    return float(value), normalized_vals.values


def dr_values(df):
    feature_cols = ["observed_affinity"]
    if "position" in df.columns:
        feature_cols.append("position")
    for c in ["user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3"]:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["reward"].values

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    q_logged = np.clip(model.predict(X), 0, 1)
    q_eval = q_logged
    w = df["candidate_prob"] / df["pscore"]

    return q_eval + w * (df["reward"].values - q_logged)


def summarize(df):
    logged_value = float(df["reward"].mean())

    ips_vals = ips_values(df)
    ips_ci = bootstrap_ci(ips_vals)

    snips_val, snips_vals = snips_value_and_values(df)
    snips_ci = bootstrap_ci(snips_vals)

    dr_vals = dr_values(df)
    dr_ci = bootstrap_ci(dr_vals)

    w = df["candidate_prob"] / df["pscore"]

    diagnostics = {
        "effective_sample_size": float(w.sum() ** 2 / max((w ** 2).sum(), 1e-9)),
        "max_importance_weight": float(w.max()),
        "mean_importance_weight": float(w.mean()),
        "low_candidate_support_rate": float((df["candidate_prob"] < 0.01).mean()),
    }

    return {
        "logged_policy_value": logged_value,
        "ips": {"value": float(np.mean(ips_vals)), "ci_low": ips_ci[0], "ci_high": ips_ci[1]},
        "snips": {"value": snips_val, "ci_low": snips_ci[0], "ci_high": snips_ci[1]},
        "doubly_robust": {"value": float(np.mean(dr_vals)), "ci_low": dr_ci[0], "ci_high": dr_ci[1]},
        "diagnostics": diagnostics,
    }


def main():
    data_dir = os.environ.get("OPEN_BANDIT_DIR")
    if not data_dir:
        raise SystemExit("Set OPEN_BANDIT_DIR first.")

    df, path = load_open_bandit(data_dir)
    df = add_candidate_policy_probability(df)

    print("Loaded Open Bandit file:", path)
    print("Rows:", len(df))
    print("Click rate:", df["reward"].mean())

    metrics = summarize(df)

    dr_lcb_lift = metrics["doubly_robust"]["ci_low"] - metrics["logged_policy_value"]
    support_ok = metrics["diagnostics"]["effective_sample_size"] >= max(1000, 0.02 * len(df))
    weight_ok = metrics["diagnostics"]["max_importance_weight"] < 100

    checks = {
        "dr_lower_bound_beats_logged_policy": dr_lcb_lift > 0,
        "effective_sample_size_ok": support_ok,
        "importance_weight_not_extreme": weight_ok,
    }

    status = "PASS" if all(checks.values()) else "ITERATE"

    metrics["dataset"] = "Open Bandit local public-data sample"
    metrics["source_file"] = str(path)
    metrics["rows"] = int(len(df))
    metrics["release_decision"] = {
        "status": status,
        "checks": checks,
        "reason": (
            "Candidate policy passed DR lower-bound and support diagnostics."
            if status == "PASS"
            else "Candidate policy remains ITERATE because value or support diagnostics did not pass all promotion checks."
        ),
    }

    (OUTPUTS / "open_bandit_ope_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.head(5000).to_csv(OUTPUTS / "open_bandit_scored_sample.csv", index=False)

    report = f"""# Open Bandit OPE Public-Data Validation Report

## Claim boundary

This run uses a local Open Bandit public dataset sample. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Source file: `{path}`
- Rows: {len(df)}
- Logged click rate: {metrics["logged_policy_value"]:.6f}

## OPE estimates

| Estimator | Value | CI low | CI high |
|---|---:|---:|---:|
| IPS | {metrics["ips"]["value"]:.6f} | {metrics["ips"]["ci_low"]:.6f} | {metrics["ips"]["ci_high"]:.6f} |
| SNIPS | {metrics["snips"]["value"]:.6f} | {metrics["snips"]["ci_low"]:.6f} | {metrics["snips"]["ci_high"]:.6f} |
| Doubly Robust | {metrics["doubly_robust"]["value"]:.6f} | {metrics["doubly_robust"]["ci_low"]:.6f} | {metrics["doubly_robust"]["ci_high"]:.6f} |

## Diagnostics

- Effective sample size: {metrics["diagnostics"]["effective_sample_size"]:.2f}
- Max importance weight: {metrics["diagnostics"]["max_importance_weight"]:.6f}
- Mean importance weight: {metrics["diagnostics"]["mean_importance_weight"]:.6f}
- Low candidate support rate: {metrics["diagnostics"]["low_candidate_support_rate"]:.6f}

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {metrics["release_decision"]["reason"]}
"""
    (REPORTS / "open_bandit_ope_public_report.md").write_text(report, encoding="utf-8")

    print("Open Bandit OPE public-data validation finished.")
    print("Report:", REPORTS / "open_bandit_ope_public_report.md")
    print("Metrics:", OUTPUTS / "open_bandit_ope_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
