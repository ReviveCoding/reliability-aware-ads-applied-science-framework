import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_log_loss(y_true, y_score):
    y_score = np.clip(np.asarray(y_score), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, y_score, labels=[0, 1]))


def safe_brier(y_true, y_score):
    y_score = np.clip(np.asarray(y_score), 1e-6, 1 - 1e-6)
    return float(brier_score_loss(y_true, y_score))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            rows.append({"bin": i, "count": 0, "confidence": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        gap = abs(acc - conf)
        ece += (mask.sum() / len(y_true)) * gap
        rows.append({"bin": i, "count": int(mask.sum()), "confidence": conf, "accuracy": acc, "gap": gap})
    return float(ece), pd.DataFrame(rows)


def classification_report_dict(y_true, y_prob):
    return {
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "log_loss": safe_log_loss(y_true, y_prob),
        "brier": safe_brier(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob)[0],
    }


def ndcg_at_k(labels, scores, k=10):
    df = pd.DataFrame({"label": labels, "score": scores}).sort_values("score", ascending=False).head(k)
    gains = (2 ** df["label"].values - 1)
    discounts = 1 / np.log2(np.arange(2, len(df) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels)[::-1][:k]
    ideal_gains = (2 ** ideal - 1)
    idcg = float(np.sum(ideal_gains * discounts[: len(ideal)]))
    return 0.0 if idcg == 0 else dcg / idcg


def recall_at_k(labels, scores, k=10):
    labels = np.asarray(labels)
    total_relevant = np.sum(labels > 0)
    if total_relevant == 0:
        return 0.0
    idx = np.argsort(scores)[::-1][:k]
    return float(np.sum(labels[idx] > 0) / total_relevant)


def precision_at_k(labels, scores, k=10):
    idx = np.argsort(scores)[::-1][:k]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(np.asarray(labels)[idx] > 0))


def auuc_score(treatment, outcome, uplift_score):
    df = pd.DataFrame({"t": treatment, "y": outcome, "u": uplift_score}).sort_values("u", ascending=False)
    treated = (df["t"] == 1).astype(float)
    control = (df["t"] == 0).astype(float)
    cum_t = treated.cumsum().replace(0, np.nan)
    cum_c = control.cumsum().replace(0, np.nan)
    cum_y_t = (df["y"] * treated).cumsum()
    cum_y_c = (df["y"] * control).cumsum()
    uplift_curve = (cum_y_t / cum_t - cum_y_c / cum_c).fillna(0)
    return float(np.trapezoid(uplift_curve.values, dx=1.0 / max(len(df), 1)))


def bootstrap_ci(values, n_boot=30, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
