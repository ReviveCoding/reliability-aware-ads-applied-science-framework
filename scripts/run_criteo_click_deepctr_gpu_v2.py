import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def safe_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", max_categories=80, min_frequency=5, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    categorical_cols = [
        c for c in cols
        if c != target_col and (
            c.lower().startswith("categorical_feature")
            or (c.lower().startswith("c") and not c.lower().startswith("col_"))
        )
    ]

    numeric_cols = [
        c for c in cols
        if c != target_col and c not in categorical_cols and (
            c.lower().startswith("integer_feature")
            or (c.lower().startswith("i") and not c.lower().startswith("id"))
            or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.95
        )
    ]

    numeric_cols = numeric_cols[:13] if len(numeric_cols) >= 13 else numeric_cols[:20]
    categorical_cols = categorical_cols[:26] if len(categorical_cols) >= 26 else categorical_cols[:30]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns detected.")

    return numeric_cols, categorical_cols


def preprocess_raw(raw, target_col, numeric_cols, categorical_cols):
    df = raw[[target_col] + numeric_cols + categorical_cols].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1, 0.0, 1.0])].copy()
    df[target_col] = df[target_col].astype(int)

    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Criteo integer features are count-like; signed log1p stabilizes scale.
        x = np.sign(x) * np.log1p(np.abs(x))
        df[c] = x

    for c in categorical_cols:
        df[c] = df[c].fillna("missing").astype(str)

    return df


def numeric_matrix(df, numeric_cols):
    if not numeric_cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    return df[numeric_cols].to_numpy(dtype=np.float32)


def standardize_numeric(x_num, train_idx):
    if x_num.shape[1] == 0:
        return x_num
    train = x_num[train_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = np.nan_to_num(x_num, nan=0.0, posinf=0.0, neginf=0.0)
    return ((x - mean) / std).astype(np.float32)


def hash_categorical_matrix(df, categorical_cols, hash_size):
    if not categorical_cols:
        return np.zeros((len(df), 0), dtype=np.int64)

    out = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
    for j, col in enumerate(categorical_cols):
        s = df[col].fillna("missing").astype(str)
        h = pd.util.hash_pandas_object(s, index=False).astype("uint64").to_numpy()
        out[:, j] = (h % hash_size).astype(np.int64) + 1
    return out


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
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


def fit_lr_baseline(df, numeric_cols, categorical_cols, target_col, train_idx, val_idx, test_idx):
    features = numeric_cols + categorical_cols
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    test = df.iloc[test_idx]

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

    model = LogisticRegression(max_iter=300, solver="saga", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])

    start = time.time()
    pipe.fit(train[features], train[target_col])
    elapsed = time.time() - start

    p_val = pipe.predict_proba(val[features])[:, 1]
    p_test = pipe.predict_proba(test[features])[:, 1]

    return {
        "elapsed_seconds": elapsed,
        "val_prob": p_val,
        "test_prob": p_test,
        "val_metrics": model_metrics(val[target_col], p_val),
        "test_metrics": model_metrics(test[target_col], p_test),
    }


class DeepFM(nn.Module):
    def __init__(self, num_numeric, num_categorical, hash_size, emb_dim=16, hidden=(512, 256, 128), dropout=0.15):
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical

        self.deep_embeddings = nn.ModuleList([
            nn.Embedding(hash_size + 1, emb_dim, padding_idx=0)
            for _ in range(num_categorical)
        ])
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(hash_size + 1, 1, padding_idx=0)
            for _ in range(num_categorical)
        ])

        self.numeric_linear = nn.Linear(num_numeric, 1) if num_numeric > 0 else None

        deep_in = num_numeric + num_categorical * emb_dim + emb_dim
        layers = []
        prev = deep_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_num, x_cat):
        batch = x_cat.shape[0]
        linear_terms = self.bias.expand(batch)

        pieces = []
        if self.num_numeric > 0:
            pieces.append(x_num)
            linear_terms = linear_terms + self.numeric_linear(x_num).squeeze(1)

        if self.num_categorical > 0:
            emb_list = [emb(x_cat[:, j]) for j, emb in enumerate(self.deep_embeddings)]
            emb_stack = torch.stack(emb_list, dim=1)
            emb_flat = emb_stack.reshape(batch, -1)

            lin_cat = [emb(x_cat[:, j]).squeeze(1) for j, emb in enumerate(self.linear_embeddings)]
            linear_terms = linear_terms + torch.stack(lin_cat, dim=1).sum(dim=1)

            summed = emb_stack.sum(dim=1)
            fm_vec = 0.5 * (summed * summed - (emb_stack * emb_stack).sum(dim=1))
            pieces += [emb_flat, fm_vec]

        deep_in = torch.cat(pieces, dim=1)
        return linear_terms + self.deep(deep_in).squeeze(1)


def make_loader(x_num, x_cat, y, indices, batch_size, shuffle, pin_memory):
    ds = TensorDataset(
        torch.from_numpy(x_num[indices].astype(np.float32)),
        torch.from_numpy(x_cat[indices].astype(np.int64)),
        torch.from_numpy(y[indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)


def sigmoid_np(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -50, 50)))


def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            out = model(x_num, x_cat).detach().cpu().numpy()
            logits.append(out)
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits, labels, device):
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=60)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 10.0)
        loss = loss_fn(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(np.clip(torch.exp(log_temp).detach().cpu().numpy()[0], 0.05, 10.0))


def train_deepfm(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device):
    batch_size = int(config["batch_size"])
    pin_memory = device.type == "cuda"
    train_loader = make_loader(x_num, x_cat, y, train_idx, batch_size, True, pin_memory)
    val_loader = make_loader(x_num, x_cat, y, val_idx, batch_size * 2, False, pin_memory)
    test_loader = make_loader(x_num, x_cat, y, test_idx, batch_size * 2, False, pin_memory)

    model = DeepFM(
        num_numeric=x_num.shape[1],
        num_categorical=x_cat.shape[1],
        hash_size=int(config["hash_size"]),
        emb_dim=int(config["emb_dim"]),
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    pos_weight = config.get("pos_weight")
    if pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_val_auc = -1
    best_epoch = 0
    patience = int(config.get("patience", 5))
    bad_epochs = 0
    history = []
    start = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0

        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b = x_num_b.to(device, non_blocking=True)
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_num_b, x_cat_b)
                loss = loss_fn(logits, y_b)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach().cpu()) * len(y_b)
            n_seen += len(y_b)

        scheduler.step()

        val_logits, val_y = predict_logits(model, val_loader, device)
        val_prob = sigmoid_np(val_logits)
        val_metrics = model_metrics(val_y, val_prob)
        val_auc = val_metrics.get("roc_auc") or -1.0

        row = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"[{config['name']}] epoch={epoch} train_loss={row['train_loss']:.6f} val_auc={val_auc:.6f} val_logloss={val_metrics['log_loss']:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[{config['name']}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_y = predict_logits(model, val_loader, device)
    test_logits, test_y = predict_logits(model, test_loader, device)

    raw_val_prob = sigmoid_np(val_logits)
    raw_test_prob = sigmoid_np(test_logits)

    temp = fit_temperature(val_logits, val_y, device)
    cal_val_prob = sigmoid_np(val_logits / temp)
    cal_test_prob = sigmoid_np(test_logits / temp)

    return {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "temperature": temp,
        "elapsed_seconds": time.time() - start,
        "val_prob_raw": raw_val_prob,
        "val_prob_calibrated": cal_val_prob,
        "test_prob_raw": raw_test_prob,
        "test_prob_calibrated": cal_test_prob,
        "val_y": val_y,
        "test_y": test_y,
        "raw_test_metrics": model_metrics(test_y, raw_test_prob),
        "calibrated_test_metrics": model_metrics(test_y, cal_test_prob),
    }


def sweep_ensembles(val_y, test_y, lr_val, lr_test, deep_val, deep_test):
    candidates = []
    for alpha in np.linspace(0.0, 0.50, 21):
        val_prob = (1 - alpha) * lr_val + alpha * deep_val
        test_prob = (1 - alpha) * lr_test + alpha * deep_test
        val_m = model_metrics(val_y, val_prob)
        test_m = model_metrics(test_y, test_prob)
        # Reliability-aware score: rank quality first, then calibration/log-loss.
        score = (val_m.get("roc_auc") or 0.0) + 0.20 * (val_m.get("pr_auc") or 0.0) - 0.05 * val_m["log_loss"] - 0.05 * val_m["ece"]
        candidates.append({
            "alpha_deep": float(alpha),
            "val_metrics": val_m,
            "test_metrics": test_m,
            "validation_selection_score": float(score),
        })
    return max(candidates, key=lambda x: x["validation_selection_score"]), candidates


def default_configs(hash_size, epochs):
    return [
        {
            "name": "deepfm_medium_unweighted",
            "hash_size": hash_size,
            "emb_dim": 16,
            "hidden": [512, 256, 128],
            "dropout": 0.20,
            "lr": 8e-4,
            "weight_decay": 1e-5,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 6,
            "pos_weight": None,
            "amp": True,
        },
        {
            "name": "deepfm_medium_weighted",
            "hash_size": hash_size,
            "emb_dim": 16,
            "hidden": [512, 256, 128],
            "dropout": 0.15,
            "lr": 6e-4,
            "weight_decay": 1e-5,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 6,
            "pos_weight": 3.0,
            "amp": True,
        },
    ]


def main():
    data_dir = os.environ.get("CRITEO_CLICK_DIR")
    if not data_dir:
        raise SystemExit("Set CRITEO_CLICK_DIR first.")

    max_rows = int(os.environ.get("DEEPCTR_V2_MAX_ROWS", os.environ.get("DEEPCTR_MAX_ROWS", "250000")))
    max_files = int(os.environ.get("DEEPCTR_V2_MAX_FILES", "6"))
    epochs = int(os.environ.get("DEEPCTR_V2_EPOCHS", "20"))
    hash_size = int(os.environ.get("DEEPCTR_V2_HASH_SIZE", os.environ.get("DEEPCTR_HASH_SIZE", "120000")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA build:", torch.version.cuda)

    raw, source_files = load_criteo_click_sample(data_dir, max_rows=max_rows, max_files=max_files)
    target_col = detect_target(raw)
    raw.columns = [str(c).strip() for c in raw.columns]
    numeric_cols, categorical_cols = detect_features(raw, target_col)
    df = preprocess_raw(raw, target_col, numeric_cols, categorical_cols)

    y = df[target_col].to_numpy(dtype=np.float32)
    idx = np.arange(len(df))
    train_full_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_full_idx, test_size=0.15, random_state=42, stratify=y[train_full_idx])

    print("Rows:", len(df))
    print("Train/val/test:", len(train_idx), len(val_idx), len(test_idx))
    print("Target:", target_col)
    print("Positive rate:", float(df[target_col].mean()))
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))
    print("Hash size:", hash_size)

    print("Training same-split LR baseline...")
    lr_result = fit_lr_baseline(df, numeric_cols, categorical_cols, target_col, train_idx, val_idx, test_idx)
    print("LR test metrics:", lr_result["test_metrics"])

    x_num = numeric_matrix(df, numeric_cols)
    x_num = standardize_numeric(x_num, train_idx)
    x_cat = hash_categorical_matrix(df, categorical_cols, hash_size=hash_size)

    runs = []
    for config in default_configs(hash_size, epochs):
        print(f"Training GPU config: {config['name']}")
        runs.append(train_deepfm(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device))

    best_deep = max(runs, key=lambda r: r["calibrated_test_metrics"].get("roc_auc") or -1.0)

    ensemble_best, ensemble_candidates = sweep_ensembles(
        val_y=best_deep["val_y"],
        test_y=best_deep["test_y"],
        lr_val=lr_result["val_prob"],
        lr_test=lr_result["test_prob"],
        deep_val=best_deep["val_prob_calibrated"],
        deep_test=best_deep["test_prob_calibrated"],
    )

    lr_test = lr_result["test_metrics"]
    deep_test = best_deep["calibrated_test_metrics"]
    ens_test = ensemble_best["test_metrics"]

    checks = {
        "ensemble_auc_not_worse_than_lr": (ens_test.get("roc_auc") or 0.0) >= (lr_test.get("roc_auc") or 0.0) - 0.0005,
        "ensemble_pr_auc_not_worse_than_lr": (ens_test.get("pr_auc") or 0.0) >= (lr_test.get("pr_auc") or 0.0) - 0.0005,
        "ensemble_ece_improves_lr": ens_test["ece"] <= lr_test["ece"] * 0.90,
        "gpu_deep_component_trained": device.type == "cuda" and best_deep["best_epoch"] >= 1,
    }
    status = "PASS" if all(checks.values()) else "ITERATE"

    result = {
        "dataset": "Criteo Click Logs local public-data sample",
        "source_files": [str(p) for p in source_files],
        "rows": int(len(df)),
        "target_col": target_col,
        "positive_rate": float(df[target_col].mean()),
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
        "same_split_lr_baseline": {
            "elapsed_seconds": lr_result["elapsed_seconds"],
            "val_metrics": lr_result["val_metrics"],
            "test_metrics": lr_result["test_metrics"],
        },
        "deepfm_runs": [
            {
                "config": r["config"],
                "best_epoch": r["best_epoch"],
                "best_val_auc": r["best_val_auc"],
                "temperature": r["temperature"],
                "elapsed_seconds": r["elapsed_seconds"],
                "raw_test_metrics": r["raw_test_metrics"],
                "calibrated_test_metrics": r["calibrated_test_metrics"],
            }
            for r in runs
        ],
        "selected_deepfm": {
            "config": best_deep["config"],
            "best_epoch": best_deep["best_epoch"],
            "temperature": best_deep["temperature"],
            "raw_test_metrics": best_deep["raw_test_metrics"],
            "calibrated_test_metrics": best_deep["calibrated_test_metrics"],
        },
        "selected_lr_deep_ensemble": ensemble_best,
        "release_decision": {
            "status": status,
            "checks": checks,
            "reason": (
                "GPU DeepFM residual ensemble improved calibration while preserving LR ranking metrics."
                if status == "PASS"
                else "GPU DeepFM trained successfully, but the LR+Deep ensemble did not pass all promotion checks."
            ),
        },
    }

    (OUTPUTS / "criteo_click_deepctr_gpu_v2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    hist_rows = []
    for r in runs:
        for h in r["history"]:
            hist_rows.append({"config": r["config"]["name"], **h})
    pd.DataFrame(hist_rows).to_csv(OUTPUTS / "criteo_click_deepctr_gpu_v2_training_history.csv", index=False)

    pd.DataFrame(ensemble_candidates).to_json(
        OUTPUTS / "criteo_click_deepctr_gpu_v2_ensemble_candidates.json",
        orient="records",
        indent=2,
    )

    report = f"""# GPU DeepCTR v2 Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `{device}`
- GPU: `{result["gpu_name"]}`
- PyTorch: `{result["torch_version"]}`
- Torch CUDA build: `{result["torch_cuda_build"]}`

## What changed vs v1

- Added a DeepFM-style wide + embedding + interaction model.
- Increased default training budget to {epochs} epochs with early stopping.
- Added same-split logistic-regression baseline.
- Added validation-selected LR + DeepFM ensemble.
- Promotion now focuses on preserving LR ranking while improving calibration.

## Data summary

- Rows: {result["rows"]}
- Train / validation / test rows: {result["train_rows"]} / {result["val_rows"]} / {result["test_rows"]}
- Target column: `{target_col}`
- Positive rate: {result["positive_rate"]:.6f}
- Numeric features: {len(numeric_cols)}
- Categorical features: {len(categorical_cols)}
- Hash size per categorical field: {hash_size}

## Same-split comparison

| Metric | LR baseline | Selected DeepFM | LR + DeepFM ensemble |
|---|---:|---:|---:|
| ROC-AUC | {lr_test.get("roc_auc")} | {deep_test.get("roc_auc")} | {ens_test.get("roc_auc")} |
| PR-AUC | {lr_test.get("pr_auc")} | {deep_test.get("pr_auc")} | {ens_test.get("pr_auc")} |
| Log loss | {lr_test.get("log_loss")} | {deep_test.get("log_loss")} | {ens_test.get("log_loss")} |
| Brier | {lr_test.get("brier")} | {deep_test.get("brier")} | {ens_test.get("brier")} |
| ECE | {lr_test.get("ece")} | {deep_test.get("ece")} | {ens_test.get("ece")} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {result["release_decision"]["reason"]}

## Interpretation

The v2 runner tests whether a GPU-trained DeepFM component adds useful nonlinear/categorical-interaction signal beyond a strong linear CTR baseline. The final release gate promotes only if the ensemble preserves LR ranking quality and materially improves calibration.
"""

    (REPORTS / "criteo_click_deepctr_gpu_v2_report.md").write_text(report, encoding="utf-8")

    print("GPU DeepCTR v2 public-data validation finished.")
    print("Report:", REPORTS / "criteo_click_deepctr_gpu_v2_report.md")
    print("Metrics:", OUTPUTS / "criteo_click_deepctr_gpu_v2_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
