import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def safe_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            max_categories=100,
            min_frequency=5,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def list_candidate_files(data_dir):
    root = Path(data_dir)
    patterns = ["*.parquet", "*.csv", "*.tsv", "*.txt", "*.gz"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No Criteo click-log sample files found under {root}")
    return sorted(files)


def read_one_file(path, nrows=None):
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    # Criteo click logs are often TSV-style. Some samples may be CSV.
    try:
        df = pd.read_csv(path, nrows=nrows)
        if df.shape[1] <= 2:
            raise ValueError("Likely not comma-separated.")
        return df
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None, nrows=nrows)

    # Standard Criteo format: label + 13 dense integer + 26 categorical.
    if df.shape[1] == 40:
        df.columns = ["label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
        df = df.rename(columns={"col_0": "label"})
    return df


def load_criteo_click_sample(data_dir, max_rows=250000, max_files=6):
    files = list_candidate_files(data_dir)
    parts = []
    rows = 0

    for file in files[:max_files]:
        print(f"Reading: {file}")
        df = read_one_file(file)
        if len(df) == 0:
            continue
        parts.append(df)
        rows += len(df)
        if rows >= max_rows:
            break

    if not parts:
        raise ValueError("No readable rows loaded from Criteo click-log sample files.")

    out = pd.concat(parts, ignore_index=True).head(max_rows).copy()
    return out, files[:max_files]


def detect_target(df):
    priority = ["label", "click", "clicked", "target", "y"]
    for col in priority:
        if col in df.columns:
            vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()[:10])
            if vals.issubset({0, 1, 0.0, 1.0}):
                return col

    for col in df.columns:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()[:10])
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            return col

    raise ValueError(f"Could not detect binary click target. Columns: {df.columns.tolist()}")


def prepare_features(raw, target_col):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target_col = str(target_col).strip()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1, 0.0, 1.0])].copy()
    df[target_col] = df[target_col].astype(int)

    # Drop obviously non-feature columns if present.
    drop_like = {"timestamp", "date", "day"}
    feature_cols = [
        c for c in df.columns
        if c != target_col and c.lower() not in drop_like
    ]

    numeric_cols = []
    categorical_cols = []
    for col in feature_cols:
        series = df[col]
        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio > 0.95:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Keep the runner local-runnable and robust on laptop hardware.
    numeric_cols = numeric_cols[:20]
    categorical_cols = categorical_cols[:30]

    keep = [target_col] + numeric_cols + categorical_cols
    df = df[keep].replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        df[col] = df[col].fillna("missing").astype(str)

    if len(df[target_col].unique()) < 2:
        raise ValueError("Loaded sample has a single target class. Increase max_rows or choose another sample file.")

    return df, numeric_cols, categorical_cols


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())

    return float(total)


def model_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)

    out = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def fit_logistic_regression(train, numeric_cols, categorical_cols, target_col):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]), numeric_cols),
            ("cat", safe_ohe(), categorical_cols),
        ],
        sparse_threshold=0.3,
    )

    model = LogisticRegression(
        max_iter=400,
        solver="saga",
        n_jobs=-1,
        class_weight=None,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train[numeric_cols + categorical_cols], train[target_col])
    return pipe


def fit_gbdt(train, numeric_cols, categorical_cols, target_col):
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_cols),
        ],
        sparse_threshold=0.0,
    )

    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train[numeric_cols + categorical_cols], train[target_col])
    return pipe


def apply_empirical_bayes(train, test, group_col, raw_prob_col, target_col):
    prior_rate = float(train[target_col].mean())
    strength = max(20.0, min(500.0, len(train) * 0.01))
    alpha = prior_rate * strength
    beta = (1.0 - prior_rate) * strength

    group_stats = train.groupby(group_col)[target_col].agg(["sum", "count"]).reset_index()
    group_stats["posterior_mean"] = (
        group_stats["sum"] + alpha
    ) / (
        group_stats["count"] + alpha + beta
    )

    out = test.merge(group_stats[[group_col, "posterior_mean", "count"]], on=group_col, how="left")
    out["posterior_mean"] = out["posterior_mean"].fillna(prior_rate)
    out["count"] = out["count"].fillna(0.0)

    weight = out["count"] / (out["count"] + strength)
    out["pctr_eb"] = (
        weight * out[raw_prob_col] + (1.0 - weight) * out["posterior_mean"]
    ).clip(1e-6, 1 - 1e-6)

    return out


def main():
    data_dir = os.environ.get("CRITEO_CLICK_DIR")
    if not data_dir:
        raise SystemExit("Set CRITEO_CLICK_DIR first.")

    max_rows = int(os.environ.get("CRITEO_CLICK_MAX_ROWS", "250000"))

    raw, source_files = load_criteo_click_sample(data_dir, max_rows=max_rows)
    target_col = detect_target(raw)
    df, numeric_cols, categorical_cols = prepare_features(raw, target_col)

    print("Loaded Criteo Click sample")
    print("Rows:", len(df))
    print("Target:", target_col)
    print("Positive rate:", df[target_col].mean())
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    train, test = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[target_col],
    )

    lr = fit_logistic_regression(train, numeric_cols, categorical_cols, target_col)
    gbdt = fit_gbdt(train, numeric_cols, categorical_cols, target_col)

    test = test.copy()
    feature_cols = numeric_cols + categorical_cols

    test["pctr_lr"] = lr.predict_proba(test[feature_cols])[:, 1]
    test["pctr_gbdt"] = gbdt.predict_proba(test[feature_cols])[:, 1]

    group_col = categorical_cols[0] if categorical_cols else numeric_cols[0]
    test_eb = apply_empirical_bayes(train, test, group_col, "pctr_gbdt", target_col)

    lr_metrics = model_metrics(test_eb[target_col], test_eb["pctr_lr"])
    gbdt_metrics = model_metrics(test_eb[target_col], test_eb["pctr_gbdt"])
    eb_metrics = model_metrics(test_eb[target_col], test_eb["pctr_eb"])

    checks = {
        "gbdt_log_loss_not_worse_than_lr": gbdt_metrics["log_loss"] <= lr_metrics["log_loss"] * 1.02,
        "gbdt_auc_not_worse_than_lr": (gbdt_metrics.get("roc_auc") or 0.0) >= (lr_metrics.get("roc_auc") or 0.0) - 0.002,
        "eb_ece_improves_or_brier_stable": (
            eb_metrics["ece"] <= gbdt_metrics["ece"]
            or eb_metrics["brier"] <= gbdt_metrics["brier"] * 1.02
        ),
    }

    status = "PASS" if all(checks.values()) else "ITERATE"

    result = {
        "dataset": "Criteo Click Logs local public-data sample",
        "source_files": [str(p) for p in source_files],
        "rows": int(len(df)),
        "test_rows": int(len(test_eb)),
        "target_col": target_col,
        "positive_rate": float(df[target_col].mean()),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "numeric_features_used": [str(c) for c in numeric_cols],
        "categorical_features_used": [str(c) for c in categorical_cols],
        "eb_group_col": str(group_col),
        "logistic_regression": lr_metrics,
        "gbdt": gbdt_metrics,
        "gbdt_empirical_bayes": eb_metrics,
        "release_decision": {
            "status": status,
            "checks": checks,
            "reason": (
                "CTR/calibration sample passed promotion checks."
                if status == "PASS"
                else "One or more CTR/calibration checks failed; keep as ITERATE."
            ),
        },
    }

    (OUTPUTS / "criteo_click_ctr_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    test_eb.head(5000).to_csv(OUTPUTS / "criteo_click_scored_sample.csv", index=False)

    report = f"""# Criteo Click Logs CTR/Calibration Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample. It does not use proprietary Amazon Ads data, real advertiser data, customer data, production traffic, or production deployment.

## Data summary

- Rows: {result["rows"]}
- Test rows: {result["test_rows"]}
- Target column: `{target_col}`
- Positive rate: {result["positive_rate"]:.6f}
- Numeric features used: {len(numeric_cols)}
- Categorical features used: {len(categorical_cols)}
- Empirical-Bayes group column: `{group_col}`

## Model comparison

| Metric | Logistic Regression | GBDT | GBDT + Empirical Bayes |
|---|---:|---:|---:|
| ROC-AUC | {lr_metrics.get("roc_auc")} | {gbdt_metrics.get("roc_auc")} | {eb_metrics.get("roc_auc")} |
| PR-AUC | {lr_metrics.get("pr_auc")} | {gbdt_metrics.get("pr_auc")} | {eb_metrics.get("pr_auc")} |
| Log loss | {lr_metrics["log_loss"]:.6f} | {gbdt_metrics["log_loss"]:.6f} | {eb_metrics["log_loss"]:.6f} |
| Brier | {lr_metrics["brier"]:.6f} | {gbdt_metrics["brier"]:.6f} | {eb_metrics["brier"]:.6f} |
| ECE | {lr_metrics["ece"]:.6f} | {gbdt_metrics["ece"]:.6f} | {eb_metrics["ece"]:.6f} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {result["release_decision"]["reason"]}

## Interpretation

This validates the CTR prediction and probability calibration module on a real public Criteo click-log sample. The framework compares a linear CTR baseline, a nonlinear GBDT challenger, and empirical-Bayes probability smoothing under explicit release checks.
"""

    (REPORTS / "criteo_click_ctr_public_report.md").write_text(report, encoding="utf-8")

    print("Criteo Click CTR/calibration public-data validation finished.")
    print("Report:", REPORTS / "criteo_click_ctr_public_report.md")
    print("Metrics:", OUTPUTS / "criteo_click_ctr_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
