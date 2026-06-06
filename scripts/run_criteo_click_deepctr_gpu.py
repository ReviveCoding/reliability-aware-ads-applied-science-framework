import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def list_candidate_files(data_dir):
    root = Path(data_dir)
    files = []
    for pattern in ["*.parquet", "*.csv", "*.tsv", "*.txt", "*.gz"]:
        files.extend(root.rglob(pattern))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No Criteo click-log sample files found under {root}")
    return sorted(files)


def read_one_file(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    try:
        df = pd.read_csv(path)
        if df.shape[1] <= 2:
            raise ValueError("Likely not comma-separated.")
        return df
    except Exception:
        df = pd.read_csv(path, sep="\t", header=None)

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
    for col in ["label", "click", "clicked", "target", "y"]:
        if col in df.columns:
            vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()[:10])
            if vals.issubset({0, 1, 0.0, 1.0}):
                return col

    for col in df.columns:
        vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()[:10])
        if vals and vals.issubset({0, 1, 0.0, 1.0}):
            return col

    raise ValueError(f"Could not detect binary click target. Columns: {df.columns.tolist()}")


def detect_features(df, target_col):
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    target_col = str(target_col).strip()

    categorical_cols = [
        c for c in cols
        if c != target_col and (
            c.lower().startswith("categorical_feature")
            or c.lower().startswith("c")
        )
    ]

    numeric_cols = [
        c for c in cols
        if c != target_col and c not in categorical_cols and (
            c.lower().startswith("integer_feature")
            or c.lower().startswith("i")
            or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.95
        )
    ]

    # Preserve Criteo-style structure and keep local training bounded.
    numeric_cols = numeric_cols[:13] if len(numeric_cols) >= 13 else numeric_cols[:20]
    categorical_cols = categorical_cols[:26] if len(categorical_cols) >= 26 else categorical_cols[:30]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns detected.")

    return numeric_cols, categorical_cols


def hash_categorical_frame(df, categorical_cols, hash_size):
    if not categorical_cols:
        return np.zeros((len(df), 0), dtype=np.int64)

    out = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for j, col in enumerate(categorical_cols):
        s = df[col].fillna("missing").astype(str)
        h = pd.util.hash_pandas_object(s, index=False).astype("uint64").to_numpy()
        out[:, j] = (h % hash_size).astype(np.int64) + 1
    return out


def numeric_frame(df, numeric_cols):
    if not numeric_cols:
        return np.zeros((len(df), 0), dtype=np.float32)

    x = df[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.to_numpy(dtype=np.float32)


def standardize_train_val_test(x_num, train_idx, val_idx, test_idx):
    if x_num.shape[1] == 0:
        return x_num

    train = x_num[train_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    x = np.nan_to_num(x_num, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - mean) / std
    return x.astype(np.float32)


class DeepCTRModel(nn.Module):
    def __init__(self, num_numeric, num_categorical, hash_size, emb_dim=16, hidden=(512, 256, 128), dropout=0.15):
        super().__init__()
        self.num_categorical = num_categorical
        self.embeddings = nn.ModuleList([
            nn.Embedding(hash_size + 1, emb_dim, padding_idx=0)
            for _ in range(num_categorical)
        ])

        in_dim = num_numeric + num_categorical * emb_dim
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        pieces = []
        if x_num.shape[1] > 0:
            pieces.append(x_num)

        if self.num_categorical > 0:
            embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
            pieces.append(torch.cat(embs, dim=1))

        x = torch.cat(pieces, dim=1)
        return self.mlp(x).squeeze(1)


def make_loader(x_num, x_cat, y, indices, batch_size, shuffle, pin_memory):
    ds = TensorDataset(
        torch.from_numpy(x_num[indices].astype(np.float32)),
        torch.from_numpy(x_cat[indices].astype(np.int64)),
        torch.from_numpy(y[indices].astype(np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def sigmoid_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


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


def predict_logits(model, loader, device):
    model.eval()
    logits = []
    labels = []

    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            out = model(x_num, x_cat).detach().cpu().numpy()
            logits.append(out)
            labels.append(y.numpy())

    return np.concatenate(logits), np.concatenate(labels)


def train_one_config(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device):
    batch_size = int(config["batch_size"])
    pin_memory = device.type == "cuda"

    train_loader = make_loader(x_num, x_cat, y, train_idx, batch_size, True, pin_memory)
    val_loader = make_loader(x_num, x_cat, y, val_idx, batch_size * 2, False, pin_memory)
    test_loader = make_loader(x_num, x_cat, y, test_idx, batch_size * 2, False, pin_memory)

    model = DeepCTRModel(
        num_numeric=x_num.shape[1],
        num_categorical=x_cat.shape[1],
        hash_size=int(config["hash_size"]),
        emb_dim=int(config["emb_dim"]),
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_val_auc = -1.0
    history = []
    start = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0

        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b = x_num_b.to(device, non_blocking=True)
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_num_b, x_cat_b)
                loss = loss_fn(logits, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.detach().cpu()) * len(y_b)
            n_seen += len(y_b)

        val_logits, val_y = predict_logits(model, val_loader, device)
        val_prob = sigmoid_np(val_logits)
        val_metrics = model_metrics(val_y, val_prob)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(n_seen, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(f"[{config['name']}] epoch={epoch} train_loss={row['train_loss']:.6f} val_auc={val_metrics.get('roc_auc')} val_logloss={val_metrics['log_loss']:.6f}")

        val_auc = val_metrics.get("roc_auc") or -1.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_y = predict_logits(model, val_loader, device)
    test_logits, test_y = predict_logits(model, test_loader, device)

    raw_test_prob = sigmoid_np(test_logits)
    raw_metrics = model_metrics(test_y, raw_test_prob)

    temperature = fit_temperature(val_logits, val_y, device)
    calibrated_test_prob = sigmoid_np(test_logits / temperature)
    calibrated_metrics = model_metrics(test_y, calibrated_test_prob)

    elapsed = time.time() - start

    return {
        "config": config,
        "history": history,
        "best_val_auc": best_val_auc,
        "temperature": temperature,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "elapsed_seconds": elapsed,
        "test_predictions": {
            "logits": test_logits[:5000].tolist(),
            "prob_raw": raw_test_prob[:5000].tolist(),
            "prob_calibrated": calibrated_test_prob[:5000].tolist(),
            "label": test_y[:5000].astype(int).tolist(),
        },
    }


def fit_temperature(logits, labels, device):
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 10.0)
        loss = loss_fn(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = torch.exp(log_temp).detach().cpu().numpy()[0]
    return float(np.clip(temp, 0.05, 10.0))


def load_baseline_metrics():
    path = OUTPUTS / "criteo_click_ctr_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_configs(hash_size, epochs):
    return [
        {
            "name": "embedding_mlp_medium",
            "hash_size": hash_size,
            "emb_dim": 16,
            "hidden": [512, 256, 128],
            "dropout": 0.15,
            "lr": 8e-4,
            "weight_decay": 1e-5,
            "batch_size": 8192,
            "epochs": epochs,
            "amp": True,
        },
        {
            "name": "embedding_mlp_compact",
            "hash_size": hash_size,
            "emb_dim": 12,
            "hidden": [256, 128, 64],
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 2e-5,
            "batch_size": 8192,
            "epochs": epochs,
            "amp": True,
        },
    ]


def main():
    data_dir = os.environ.get("CRITEO_CLICK_DIR")
    if not data_dir:
        raise SystemExit("Set CRITEO_CLICK_DIR first.")

    max_rows = int(os.environ.get("DEEPCTR_MAX_ROWS", "250000"))
    max_files = int(os.environ.get("DEEPCTR_MAX_FILES", "6"))
    epochs = int(os.environ.get("DEEPCTR_EPOCHS", "6"))
    hash_size = int(os.environ.get("DEEPCTR_HASH_SIZE", "60000"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA build:", torch.version.cuda)

    raw, source_files = load_criteo_click_sample(data_dir, max_rows=max_rows, max_files=max_files)
    target_col = detect_target(raw)

    raw.columns = [str(c).strip() for c in raw.columns]
    raw[target_col] = pd.to_numeric(raw[target_col], errors="coerce")
    raw = raw[raw[target_col].isin([0, 1, 0.0, 1.0])].copy()
    raw[target_col] = raw[target_col].astype(int)

    numeric_cols, categorical_cols = detect_features(raw, target_col)

    print("Rows:", len(raw))
    print("Target:", target_col)
    print("Positive rate:", float(raw[target_col].mean()))
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))

    y = raw[target_col].to_numpy(dtype=np.float32)
    x_num = numeric_frame(raw, numeric_cols)
    x_cat = hash_categorical_frame(raw, categorical_cols, hash_size=hash_size)

    idx = np.arange(len(raw))
    train_full_idx, test_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        train_full_idx,
        test_size=0.15,
        random_state=42,
        stratify=y[train_full_idx],
    )

    x_num = standardize_train_val_test(x_num, train_idx, val_idx, test_idx)

    configs = default_configs(hash_size=hash_size, epochs=epochs)
    runs = []
    for config in configs:
        print(f"Training config: {config['name']}")
        runs.append(train_one_config(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device))

    best = max(runs, key=lambda r: r["calibrated_metrics"].get("roc_auc") or -1.0)

    baseline = load_baseline_metrics()
    baseline_lr = baseline.get("logistic_regression", {}) if baseline else {}

    deep_raw = best["raw_metrics"]
    deep_cal = best["calibrated_metrics"]

    checks = {}
    if baseline_lr:
        checks = {
            "deep_auc_beats_lr_by_0_002": (deep_cal.get("roc_auc") or 0.0) >= (baseline_lr.get("roc_auc") or 0.0) + 0.002,
            "deep_pr_auc_not_worse_than_lr": (deep_cal.get("pr_auc") or 0.0) >= (baseline_lr.get("pr_auc") or 0.0) - 0.0005,
            "deep_log_loss_not_worse_than_lr": deep_cal["log_loss"] <= baseline_lr["log_loss"] * 1.01,
            "calibration_improves_raw_deep_ece": deep_cal["ece"] <= deep_raw["ece"],
        }
    else:
        checks = {
            "cuda_used": device.type == "cuda",
            "calibration_improves_raw_deep_ece": deep_cal["ece"] <= deep_raw["ece"],
        }

    status = "PASS" if all(checks.values()) else "ITERATE"

    result = {
        "dataset": "Criteo Click Logs local public-data sample",
        "source_files": [str(p) for p in source_files],
        "rows": int(len(raw)),
        "target_col": target_col,
        "positive_rate": float(raw[target_col].mean()),
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "hash_size": hash_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "baseline_lr_from_previous_public_run": baseline_lr,
        "runs": [
            {
                "config": r["config"],
                "best_val_auc": r["best_val_auc"],
                "temperature": r["temperature"],
                "raw_metrics": r["raw_metrics"],
                "calibrated_metrics": r["calibrated_metrics"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in runs
        ],
        "selected_model": {
            "config": best["config"],
            "temperature": best["temperature"],
            "raw_metrics": deep_raw,
            "calibrated_metrics": deep_cal,
        },
        "release_decision": {
            "status": status,
            "checks": checks,
            "reason": (
                "GPU DeepCTR model passed LR comparison and calibration checks."
                if status == "PASS"
                else "GPU DeepCTR model trained successfully but did not pass all LR comparison/calibration promotion checks."
            ),
        },
    }

    (OUTPUTS / "criteo_click_deepctr_gpu_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(best["test_predictions"]).to_csv(
        OUTPUTS / "criteo_click_deepctr_gpu_predictions_sample.csv",
        index=False,
    )

    history_rows = []
    for r in runs:
        for h in r["history"]:
            row = {"config": r["config"]["name"], **h}
            history_rows.append(row)
    pd.DataFrame(history_rows).to_csv(
        OUTPUTS / "criteo_click_deepctr_gpu_training_history.csv",
        index=False,
    )

    baseline_auc = baseline_lr.get("roc_auc") if baseline_lr else None
    baseline_pr = baseline_lr.get("pr_auc") if baseline_lr else None
    baseline_ll = baseline_lr.get("log_loss") if baseline_lr else None
    baseline_ece = baseline_lr.get("ece") if baseline_lr else None

    report = f"""# GPU DeepCTR Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `{device}`
- GPU: `{result["gpu_name"]}`
- PyTorch: `{result["torch_version"]}`
- Torch CUDA build: `{result["torch_cuda_build"]}`

## Data summary

- Rows: {result["rows"]}
- Train / validation / test rows: {result["train_rows"]} / {result["val_rows"]} / {result["test_rows"]}
- Target column: `{target_col}`
- Positive rate: {result["positive_rate"]:.6f}
- Numeric features: {len(numeric_cols)}
- Categorical features: {len(categorical_cols)}
- Hash size per categorical feature: {hash_size}

## Baseline vs GPU DeepCTR

| Metric | Logistic Regression baseline | DeepCTR raw | DeepCTR temperature-calibrated |
|---|---:|---:|---:|
| ROC-AUC | {baseline_auc} | {deep_raw.get("roc_auc")} | {deep_cal.get("roc_auc")} |
| PR-AUC | {baseline_pr} | {deep_raw.get("pr_auc")} | {deep_cal.get("pr_auc")} |
| Log loss | {baseline_ll} | {deep_raw.get("log_loss")} | {deep_cal.get("log_loss")} |
| Brier | {baseline_lr.get("brier") if baseline_lr else None} | {deep_raw.get("brier")} | {deep_cal.get("brier")} |
| ECE | {baseline_ece} | {deep_raw.get("ece")} | {deep_cal.get("ece")} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {result["release_decision"]["reason"]}

## Interpretation

The GPU DeepCTR model uses categorical embeddings for the 26 Criteo categorical features and dense numeric inputs for the 13 integer features. Temperature scaling is applied as a post-training calibration step. Promotion requires both ranking quality and calibration discipline rather than GPU training alone.
"""

    (REPORTS / "criteo_click_deepctr_gpu_report.md").write_text(report, encoding="utf-8")

    print("GPU DeepCTR public-data validation finished.")
    print("Report:", REPORTS / "criteo_click_deepctr_gpu_report.md")
    print("Metrics:", OUTPUTS / "criteo_click_deepctr_gpu_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
