import os
import json
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.metrics import bootstrap_ci
from src.ope.estimators import cross_fitted_dr, bootstrap_snips_ci, snips

REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


def find_bandit_file(data_dir):
    root = Path(data_dir)
    if root.is_file():
        return root
    candidates = sorted(p for p in root.rglob("*.csv") if p.name.lower() != "item_context.csv")
    if len(candidates) != 1:
        raise ValueError("Provide an explicit Open Bandit CSV path, or a directory with exactly one feedback CSV.")
    return candidates[0]


def load_open_bandit(data_dir, max_rows=300000):
    path = find_bandit_file(data_dir)
    df = pd.read_csv(path, nrows=max_rows)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"item_id", "click", "propensity_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns: {df.columns.tolist()}")

    # Do not silently replace missing rewards, invalid actions, or unsupported propensities.
    for name in ["click", "propensity_score", "item_id"]:
        df[name] = pd.to_numeric(df[name], errors="raise")
        if not np.isfinite(df[name]).all():
            raise ValueError(f"{name} must be finite")
    if not df["click"].between(0, 1).all():
        raise ValueError("click must lie in [0, 1]")
    if not ((df.propensity_score > 0) & (df.propensity_score <= 1)).all():
        raise ValueError("propensity_score must lie in (0, 1]")
    if not (df.item_id == np.floor(df.item_id)).all():
        raise ValueError("item_id must contain integers")
    df["reward"] = df["click"].astype(float)
    df["pscore"] = df["propensity_score"].astype(float)
    df["item_id_int"] = df["item_id"].astype(int)
    return df.reset_index(drop=True), path


def candidate_policy(df):
    """Return the full softmax policy, with explicit affinity-suffix action IDs."""
    prefix = "user-item_affinity_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError("Full per-action affinity columns are required; no uniform fallback is inferred")
    try:
        action_values = np.array([int(c[len(prefix):]) for c in cols])
    except ValueError as exc:
        raise ValueError("Affinity column suffixes must be integer item IDs") from exc
    if len(np.unique(action_values)) != len(action_values):
        raise ValueError("Affinity action IDs must be unique")
    affinity = df[cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(affinity).all():
        raise ValueError("Affinity features must be finite")
    lookup = {a: j for j, a in enumerate(action_values)}
    if not set(df.item_id_int).issubset(lookup):
        raise ValueError("Every logged item must have an explicitly mapped affinity column")
    indices = np.array([lookup[a] for a in df.item_id_int])
    exp_scores = np.exp(affinity - affinity.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return action_values, affinity, probs, indices


def add_candidate_policy_probability(df):
    df = df.copy()
    _, affinity, probs, indices = candidate_policy(df)
    df["candidate_prob"] = probs[np.arange(len(df)), indices]
    df["observed_affinity"] = affinity[np.arange(len(df)), indices]
    return df


def ips_values(df):
    w = df["candidate_prob"] / df["pscore"]
    return (w * df["reward"]).values


def snips_value_and_values(df):
    """Return point estimate and paired numerator/denominator rows."""
    return snips(df)


def dr_values(df):
    actions, affinity, probs, _ = candidate_policy(df)
    # One-hot item IDs avoid an artificial ordinal action scale. Categorical user
    # features are encoded, not silently coerced to zero. No rewards enter encoding.
    context_cols = [c for c in df.columns if c.startswith("user_feature_")]
    if "position" in df:
        context_cols.append("position")
    context = pd.get_dummies(df[context_cols].astype(str), dtype=float).to_numpy()
    lookup = {a: j for j, a in enumerate(actions)}
    identity = np.eye(len(actions))

    def features(rows, action):
        row_actions = np.broadcast_to(action, (len(rows),))
        indices = np.array([lookup[a] for a in row_actions])
        return np.column_stack((context[rows], affinity[rows, indices], identity[indices]))

    model = RandomForestRegressor(
        n_estimators=80, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=1,
    )
    return cross_fitted_dr(
        df.reward.to_numpy(), df.item_id_int.to_numpy(), df.pscore.to_numpy(),
        probs, actions, features, n_splits=5, seed=42, model=model,
    )


def summarize(df):
    logged_value = float(df["reward"].mean())

    ips_vals = ips_values(df)
    ips_ci = bootstrap_ci(ips_vals, n_boot=300)

    snips_val, _ = snips_value_and_values(df)
    snips_ci = bootstrap_snips_ci(df.reward.to_numpy(), (df.candidate_prob / df.pscore).to_numpy(), n_boot=300)

    dr_vals = dr_values(df)
    dr_ci = bootstrap_ci(dr_vals, n_boot=300)

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

    REPORTS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    df, path = load_open_bandit(data_dir, max_rows=int(os.environ.get("OPEN_BANDIT_MAX_ROWS", "300000")))
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

    metrics["estimator_version"] = "crossfit-action-integrated-v1"
    metrics["provenance"] = {
        "estimator_version": "crossfit-action-integrated-v1",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_url": os.environ.get("OPEN_BANDIT_SOURCE_URL"),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "crossfit_folds": 5, "seed": 42, "bootstrap_replicates": 300,
        "policy": "softmax over all explicitly mapped item affinity columns",
        "estimand": "per-position marginal reward, averaged over observed positions; not slate value",
        "ci_method": "IID row percentile bootstrap; SNIPS ratio recomputed; DR fixed out-of-fold scores",
        "limitations": "No nuisance refit in bootstrap; ignores temporal/user dependence; not production qualification",
    }
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

    (OUTPUTS / "open_bandit_ope_metrics_corrected.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.head(5000).to_csv(OUTPUTS / "open_bandit_corrected_scored_sample.csv", index=False)

    report = f"""# Open Bandit OPE Public-Data Validation Report

## Claim boundary

This corrected run uses a local Open Bandit public dataset sample. Historical artifacts are preserved. DR integrates all candidate actions and uses five-fold out-of-fold predictions; SNIPS recomputes its denominator in every bootstrap resample. Intervals are exploratory IID-row intervals, not dependence-robust or production qualification evidence. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Provenance

```json
{json.dumps(metrics["provenance"], indent=2)}
```

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
    (REPORTS / "open_bandit_ope_corrected_report.md").write_text(report, encoding="utf-8")

    print("Open Bandit OPE public-data validation finished.")
    print("Report:", REPORTS / "open_bandit_ope_corrected_report.md")
    print("Metrics:", OUTPUTS / "open_bandit_ope_metrics_corrected.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
