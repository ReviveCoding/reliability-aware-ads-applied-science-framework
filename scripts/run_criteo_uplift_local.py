
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


def trapz_area(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values))
    if hasattr(np, "trapz"):
        return float(np.trapz(values))
    return float(np.sum((values[1:] + values[:-1]) / 2.0))




ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def to_binary(series):
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0
    })
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype(int)


def read_balanced_criteo(path, n_per_group=120000, chunksize=250000, target_col="conversion"):
    treated_parts = []
    control_parts = []
    feature_cols = None
    required = {"treatment", "conversion", "visit"}

    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available columns: {chunk.columns.tolist()}")

        feature_cols = [c for c in chunk.columns if c.startswith("f")]
        if not feature_cols:
            raise ValueError("No feature columns found. Expected f0, f1, ..., f11.")

        keep = feature_cols + ["treatment", "conversion", "visit"]
        chunk = chunk[keep].copy()

        chunk["treatment"] = to_binary(chunk["treatment"])
        chunk["conversion"] = to_binary(chunk["conversion"])
        chunk["visit"] = to_binary(chunk["visit"])

        treated = chunk[chunk["treatment"] == 1]
        control = chunk[chunk["treatment"] == 0]

        if len(treated_parts) == 0 or sum(len(x) for x in treated_parts) < n_per_group:
            treated_parts.append(treated.head(max(0, n_per_group - sum(len(x) for x in treated_parts))))

        if len(control_parts) == 0 or sum(len(x) for x in control_parts) < n_per_group:
            control_parts.append(control.head(max(0, n_per_group - sum(len(x) for x in control_parts))))

        n_treated = sum(len(x) for x in treated_parts)
        n_control = sum(len(x) for x in control_parts)

        if n_treated >= n_per_group and n_control >= n_per_group:
            break

    df = pd.concat(treated_parts + control_parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    if df["treatment"].nunique() < 2:
        raise ValueError(
            "Balanced read failed: treatment has only one class. "
            "Check whether the file is sorted, corrupted, or parsed incorrectly."
        )

    if df[target_col].nunique() < 2:
        alt = "visit" if target_col == "conversion" else "conversion"
        if df[alt].nunique() >= 2:
            print(f"[WARN] Target {target_col} has one class in sample. Switching to {alt}.")
            target_col = alt
        else:
            raise ValueError("Both conversion and visit have one class in sampled data.")

    return df, feature_cols, target_col


def uplift_gain(df, score_col, outcome_col):
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    treated = (ordered["treatment"] == 1).astype(float)
    control = (ordered["treatment"] == 0).astype(float)

    cum_t = treated.cumsum().replace(0, np.nan)
    cum_c = control.cumsum().replace(0, np.nan)

    gain = (
        (ordered[outcome_col] * treated).cumsum()
        - (ordered[outcome_col] * control).cumsum() * (cum_t / cum_c)
    ).fillna(0)

    return gain.values


def auuc(df, score_col, outcome_col):
    gain = uplift_gain(df, score_col, outcome_col)
    return float(trapz_area(gain) / max(len(gain), 1))


def qini_proxy(df, score_col, outcome_col):
    gain = uplift_gain(df, score_col, outcome_col)
    baseline = np.linspace(0, gain[-1] if len(gain) else 0, len(gain))
    return float(trapz_area(gain - baseline) / max(len(gain), 1))


def uplift_at_k(df, score_col, outcome_col, frac=0.2):
    top = df.sort_values(score_col, ascending=False).head(max(1, int(len(df) * frac)))
    t = top[top["treatment"] == 1][outcome_col].mean()
    c = top[top["treatment"] == 0][outcome_col].mean()
    return float(t - c)


def main():
    path = os.environ.get("CRITEO_UPLIFT_PATH")
    if not path:
        raise SystemExit("Set CRITEO_UPLIFT_PATH first.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading Criteo Uplift sample from: {path}")
    df, feature_cols, target_col = read_balanced_criteo(path, n_per_group=120000, target_col="conversion")

    print("Loaded sample:")
    print(df.shape)
    print("Treatment counts:")
    print(df["treatment"].value_counts())
    print("Target:", target_col)
    print("Target rate by treatment:")
    print(df.groupby("treatment")[target_col].mean())

    train, test = train_test_split(
        df,
        test_size=0.35,
        random_state=42,
        stratify=df["treatment"]
    )

    treated_train = train[train["treatment"] == 1]
    control_train = train[train["treatment"] == 0]

    if treated_train[target_col].nunique() < 2 or control_train[target_col].nunique() < 2:
        print("[WARN] One subgroup has single-class target. Switching target to visit.")
        target_col = "visit"
        if treated_train[target_col].nunique() < 2 or control_train[target_col].nunique() < 2:
            raise ValueError("Even visit target has single-class subgroup. Increase n_per_group.")

    mt = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, random_state=42)
    mc = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, random_state=43)
    pooled = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=31, random_state=44)

    mt.fit(treated_train[feature_cols], treated_train[target_col])
    mc.fit(control_train[feature_cols], control_train[target_col])
    pooled.fit(train[feature_cols], train[target_col])

    out = test.copy()
    out["p_treated"] = mt.predict_proba(out[feature_cols])[:, 1]
    out["p_control"] = mc.predict_proba(out[feature_cols])[:, 1]
    out["pred_uplift"] = out["p_treated"] - out["p_control"]
    out["pcvr_baseline"] = pooled.predict_proba(out[feature_cols])[:, 1]

    metrics = {
        "dataset": "Criteo Uplift v2.1 local public-data sample",
        "rows_loaded": int(len(df)),
        "test_rows": int(len(out)),
        "feature_count": int(len(feature_cols)),
        "target_col": target_col,
        "treatment_counts": {str(k): int(v) for k, v in df["treatment"].value_counts().to_dict().items()},
        "overall_target_rate": float(df[target_col].mean()),
        "target_rate_by_treatment": {str(k): float(v) for k, v in df.groupby("treatment")[target_col].mean().to_dict().items()},
        "uplift_model": {
            "auuc": auuc(out, "pred_uplift", target_col),
            "qini_proxy": qini_proxy(out, "pred_uplift", target_col),
            "uplift_at_20pct": uplift_at_k(out, "pred_uplift", target_col, frac=0.2),
            "negative_uplift_rate": float((out["pred_uplift"] < 0).mean()),
        },
        "pcvr_baseline": {
            "auuc": auuc(out, "pcvr_baseline", target_col),
            "qini_proxy": qini_proxy(out, "pcvr_baseline", target_col),
            "uplift_at_20pct": uplift_at_k(out, "pcvr_baseline", target_col, frac=0.2),
        }
    }

    (OUTPUT_DIR / "criteo_uplift_predictions_sample.csv").write_text(
        out.head(5000).to_csv(index=False),
        encoding="utf-8"
    )

    with open(OUTPUT_DIR / "criteo_uplift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = f"""# Criteo Uplift Public-Data Validation Report

## Claim boundary

This run uses the local Criteo Uplift public dataset sample. It does not use proprietary Amazon Ads data, real advertiser data, real customer data, production traffic, or production deployment.

## Data summary

- Rows loaded: {metrics["rows_loaded"]}
- Test rows: {metrics["test_rows"]}
- Feature count: {metrics["feature_count"]}
- Target column: {metrics["target_col"]}
- Treatment counts: {metrics["treatment_counts"]}
- Overall target rate: {metrics["overall_target_rate"]:.6f}
- Target rate by treatment: {metrics["target_rate_by_treatment"]}

## Baseline vs uplift model

| Metric | pCVR baseline | T-learner uplift model |
|---|---:|---:|
| AUUC | {metrics["pcvr_baseline"]["auuc"]:.6f} | {metrics["uplift_model"]["auuc"]:.6f} |
| Qini proxy | {metrics["pcvr_baseline"]["qini_proxy"]:.6f} | {metrics["uplift_model"]["qini_proxy"]:.6f} |
| Uplift@20% | {metrics["pcvr_baseline"]["uplift_at_20pct"]:.6f} | {metrics["uplift_model"]["uplift_at_20pct"]:.6f} |
| Negative uplift rate | ? | {metrics["uplift_model"]["negative_uplift_rate"]:.6f} |

## Interpretation

This validates the uplift / incrementality module on a real public dataset. The comparison is between a vanilla pCVR targeting baseline and a T-learner uplift targeting policy.

## Next actions

- Add sample-size sensitivity.
- Add treatment/control balance diagnostics.
- Add X-learner or causal forest optional benchmark.
- Connect this report to the global release gate.
"""

    (REPORT_DIR / "criteo_uplift_public_report.md").write_text(report, encoding="utf-8")

    print("Criteo uplift public-data validation finished.")
    print(f"Report: {REPORT_DIR / 'criteo_uplift_public_report.md'}")
    print(f"Metrics: {OUTPUT_DIR / 'criteo_uplift_metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
