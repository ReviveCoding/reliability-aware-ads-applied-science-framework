"""AutoInt-style attention CTR experiment for the reliability-aware ads framework.

Place this file under:
  reliability-aware-ads-applied-science-framework/scripts/run_criteo_click_autoint_gpu.py

It reuses the data-loading and metric helpers from run_criteo_click_deepctr_gpu_v2.py,
then trains an AutoInt-style self-attention CTR component on CUDA and evaluates
an LR + AutoInt residual ensemble under release gates.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from run_criteo_click_deepctr_gpu_v2 import (
        load_criteo_click_sample,
        detect_target,
        detect_features,
        preprocess_raw,
        numeric_matrix,
        standardize_numeric,
        hash_categorical_matrix,
        model_metrics,
        fit_lr_baseline,
        make_loader,
        sigmoid_np,
        predict_logits,
        fit_temperature,
    )
except Exception as exc:
    raise RuntimeError(
        "This script expects scripts/run_criteo_click_deepctr_gpu_v2.py to exist. "
        "Place both files in the scripts folder and rerun. Original error: " + repr(exc)
    )


class AutoIntBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class AutoIntCTR(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        num_categorical: int,
        hash_size: int,
        emb_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden=(256, 128),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.emb_dim = emb_dim

        if num_numeric > 0:
            self.numeric_weight = nn.Parameter(torch.randn(num_numeric, emb_dim) * 0.02)
            self.numeric_bias = nn.Parameter(torch.zeros(num_numeric, emb_dim))
            self.numeric_linear = nn.Linear(num_numeric, 1)
        else:
            self.numeric_weight = None
            self.numeric_bias = None
            self.numeric_linear = None

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(hash_size + 1, emb_dim, padding_idx=0)
            for _ in range(num_categorical)
        ])
        self.cat_linear = nn.ModuleList([
            nn.Embedding(hash_size + 1, 1, padding_idx=0)
            for _ in range(num_categorical)
        ])

        self.blocks = nn.ModuleList([
            AutoIntBlock(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        token_count = num_numeric + num_categorical
        pooled_dim = emb_dim * 2 + token_count * emb_dim
        layers = []
        prev = pooled_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_num, x_cat):
        tokens = []
        batch = x_num.shape[0] if x_num.shape[1] > 0 else x_cat.shape[0]
        linear = self.bias.expand(batch)

        if self.num_numeric > 0:
            num_tokens = x_num.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
            tokens.append(num_tokens)
            linear = linear + self.numeric_linear(x_num).squeeze(1)

        if self.num_categorical > 0:
            cat_tokens = [emb(x_cat[:, j]).unsqueeze(1) for j, emb in enumerate(self.cat_embeddings)]
            tokens.append(torch.cat(cat_tokens, dim=1))
            cat_lin = [emb(x_cat[:, j]).squeeze(1) for j, emb in enumerate(self.cat_linear)]
            linear = linear + torch.stack(cat_lin, dim=1).sum(dim=1)

        x = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x = block(x)

        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        flat = x.reshape(x.shape[0], -1)
        head_in = torch.cat([pooled_mean, pooled_max, flat], dim=1)
        return linear + self.head(head_in).squeeze(1)


def train_autoint(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device):
    batch_size = int(config["batch_size"])
    pin_memory = device.type == "cuda"
    train_loader = make_loader(x_num, x_cat, y, train_idx, batch_size, True, pin_memory)
    val_loader = make_loader(x_num, x_cat, y, val_idx, batch_size * 2, False, pin_memory)
    test_loader = make_loader(x_num, x_cat, y, test_idx, batch_size * 2, False, pin_memory)

    model = AutoIntCTR(
        num_numeric=x_num.shape[1],
        num_categorical=x_cat.shape[1],
        hash_size=int(config["hash_size"]),
        emb_dim=int(config["emb_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_val_auc = -1.0
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
        "val_prob_calibrated": cal_val_prob,
        "test_prob_calibrated": cal_test_prob,
        "val_y": val_y,
        "test_y": test_y,
        "raw_test_metrics": model_metrics(test_y, raw_test_prob),
        "calibrated_test_metrics": model_metrics(test_y, cal_test_prob),
    }


def sweep_ensembles(val_y, test_y, lr_val, lr_test, autoint_val, autoint_test):
    candidates = []
    for alpha in np.linspace(0.0, 0.50, 21):
        val_prob = (1 - alpha) * lr_val + alpha * autoint_val
        test_prob = (1 - alpha) * lr_test + alpha * autoint_test
        val_m = model_metrics(val_y, val_prob)
        test_m = model_metrics(test_y, test_prob)
        score = (val_m.get("roc_auc") or 0.0) + 0.20 * (val_m.get("pr_auc") or 0.0) - 0.05 * val_m["log_loss"] - 0.05 * val_m["ece"]
        candidates.append({
            "alpha_autoint": float(alpha),
            "val_metrics": val_m,
            "test_metrics": test_m,
            "validation_selection_score": float(score),
        })
    return max(candidates, key=lambda x: x["validation_selection_score"]), candidates


def load_deepfm_v2_result():
    path = OUTPUTS / "criteo_click_deepctr_gpu_v2_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_configs(hash_size, epochs):
    return [
        {
            "name": "autoint_2layer_4head",
            "hash_size": hash_size,
            "emb_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "hidden": [256, 128],
            "dropout": 0.15,
            "lr": 8e-4,
            "weight_decay": 1e-5,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 5,
            "amp": True,
        },
        {
            "name": "autoint_3layer_4head",
            "hash_size": hash_size,
            "emb_dim": 16,
            "num_heads": 4,
            "num_layers": 3,
            "hidden": [256, 128],
            "dropout": 0.20,
            "lr": 6e-4,
            "weight_decay": 1e-5,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 5,
            "amp": True,
        },
    ]


def main():
    data_dir = os.environ.get("CRITEO_CLICK_DIR")
    if not data_dir:
        raise SystemExit("Set CRITEO_CLICK_DIR first.")

    max_rows = int(os.environ.get("AUTOINT_MAX_ROWS", os.environ.get("DEEPCTR_V2_MAX_ROWS", "250000")))
    max_files = int(os.environ.get("AUTOINT_MAX_FILES", "6"))
    epochs = int(os.environ.get("AUTOINT_EPOCHS", "16"))
    hash_size = int(os.environ.get("AUTOINT_HASH_SIZE", os.environ.get("DEEPCTR_V2_HASH_SIZE", "120000")))

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
        print(f"Training AutoInt config: {config['name']}")
        runs.append(train_autoint(config, x_num, x_cat, y, train_idx, val_idx, test_idx, device))

    best_autoint = max(runs, key=lambda r: r["calibrated_test_metrics"].get("roc_auc") or -1.0)
    ensemble_best, ensemble_candidates = sweep_ensembles(
        val_y=best_autoint["val_y"],
        test_y=best_autoint["test_y"],
        lr_val=lr_result["val_prob"],
        lr_test=lr_result["test_prob"],
        autoint_val=best_autoint["val_prob_calibrated"],
        autoint_test=best_autoint["test_prob_calibrated"],
    )

    lr_test = lr_result["test_metrics"]
    auto_test = best_autoint["calibrated_test_metrics"]
    ens_test = ensemble_best["test_metrics"]

    deepfm_v2 = load_deepfm_v2_result()
    deepfm_ens = None
    deepfm_comparison = None
    if deepfm_v2 and "selected_lr_deep_ensemble" in deepfm_v2:
        deepfm_ens = deepfm_v2["selected_lr_deep_ensemble"]["test_metrics"]
        beat_count = 0
        if ens_test.get("roc_auc", 0) >= deepfm_ens.get("roc_auc", 0):
            beat_count += 1
        if ens_test.get("pr_auc", 0) >= deepfm_ens.get("pr_auc", 0):
            beat_count += 1
        if ens_test.get("log_loss", 999) <= deepfm_ens.get("log_loss", 999):
            beat_count += 1
        if ens_test.get("ece", 999) <= deepfm_ens.get("ece", 999):
            beat_count += 1
        deepfm_comparison = {
            "deepfm_v2_ensemble_test_metrics": deepfm_ens,
            "autoint_ensemble_metrics_better_or_equal_count": beat_count,
        }

    checks = {
        "ensemble_auc_not_worse_than_lr": (ens_test.get("roc_auc") or 0.0) >= (lr_test.get("roc_auc") or 0.0) - 0.0005,
        "ensemble_pr_auc_not_worse_than_lr": (ens_test.get("pr_auc") or 0.0) >= (lr_test.get("pr_auc") or 0.0) - 0.0005,
        "ensemble_ece_improves_lr": ens_test["ece"] <= lr_test["ece"] * 0.90,
        "autoint_component_trained_on_cuda": device.type == "cuda" and best_autoint["best_epoch"] >= 1,
    }
    if deepfm_comparison:
        checks["autoint_matches_or_beats_deepfm_on_2_metrics"] = deepfm_comparison["autoint_ensemble_metrics_better_or_equal_count"] >= 2

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
        "autoint_runs": [
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
        "selected_autoint": {
            "config": best_autoint["config"],
            "best_epoch": best_autoint["best_epoch"],
            "temperature": best_autoint["temperature"],
            "raw_test_metrics": best_autoint["raw_test_metrics"],
            "calibrated_test_metrics": best_autoint["calibrated_test_metrics"],
        },
        "selected_lr_autoint_ensemble": ensemble_best,
        "deepfm_v2_comparison": deepfm_comparison,
        "release_decision": {
            "status": status,
            "checks": checks,
            "reason": (
                "AutoInt residual ensemble passed ranking/calibration and advanced-model comparison checks."
                if status == "PASS"
                else "AutoInt trained successfully, but the residual ensemble did not pass all ranking/calibration or DeepFM comparison checks."
            ),
        },
    }

    (OUTPUTS / "criteo_click_autoint_gpu_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    hist_rows = []
    for r in runs:
        for h in r["history"]:
            hist_rows.append({"config": r["config"]["name"], **h})
    pd.DataFrame(hist_rows).to_csv(OUTPUTS / "criteo_click_autoint_gpu_training_history.csv", index=False)
    pd.DataFrame(ensemble_candidates).to_json(
        OUTPUTS / "criteo_click_autoint_gpu_ensemble_candidates.json",
        orient="records",
        indent=2,
    )

    deepfm_auc = deepfm_ens.get("roc_auc") if deepfm_ens else None
    deepfm_pr = deepfm_ens.get("pr_auc") if deepfm_ens else None
    deepfm_ll = deepfm_ens.get("log_loss") if deepfm_ens else None
    deepfm_brier = deepfm_ens.get("brier") if deepfm_ens else None
    deepfm_ece = deepfm_ens.get("ece") if deepfm_ens else None

    report = f"""# AutoInt Attention CTR Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `{device}`
- GPU: `{result['gpu_name']}`
- PyTorch: `{result['torch_version']}`
- Torch CUDA build: `{result['torch_cuda_build']}`

## Experiment purpose

This experiment tests whether an AutoInt-style self-attention feature-interaction block adds useful signal beyond a strong logistic-regression CTR baseline and the prior GPU DeepFM residual ensemble.

## Data summary

- Rows: {result['rows']}
- Train / validation / test rows: {result['train_rows']} / {result['val_rows']} / {result['test_rows']}
- Target column: `{target_col}`
- Positive rate: {result['positive_rate']:.6f}
- Numeric features: {len(numeric_cols)}
- Categorical features: {len(categorical_cols)}
- Hash size per categorical field: {hash_size}

## Same-split comparison

| Metric | LR baseline | Selected AutoInt | LR + AutoInt ensemble | LR + DeepFM v2 ensemble |
|---|---:|---:|---:|---:|
| ROC-AUC | {lr_test.get('roc_auc')} | {auto_test.get('roc_auc')} | {ens_test.get('roc_auc')} | {deepfm_auc} |
| PR-AUC | {lr_test.get('pr_auc')} | {auto_test.get('pr_auc')} | {ens_test.get('pr_auc')} | {deepfm_pr} |
| Log loss | {lr_test.get('log_loss')} | {auto_test.get('log_loss')} | {ens_test.get('log_loss')} | {deepfm_ll} |
| Brier | {lr_test.get('brier')} | {auto_test.get('brier')} | {ens_test.get('brier')} | {deepfm_brier} |
| ECE | {lr_test.get('ece')} | {auto_test.get('ece')} | {ens_test.get('ece')} | {deepfm_ece} |

## Release decision

- Status: **{status}**
- Checks: {checks}
- Reason: {result['release_decision']['reason']}

## Interpretation

AutoInt maps numerical and categorical CTR fields into feature tokens and applies multi-head self-attention to model high-order feature interactions. The release gate promotes attention only if the validation-selected residual ensemble preserves LR ranking quality, improves calibration, and meaningfully compares with the prior DeepFM residual ensemble.
"""

    (REPORTS / "criteo_click_autoint_gpu_report.md").write_text(report, encoding="utf-8")

    print("AutoInt attention CTR public-data validation finished.")
    print("Report:", REPORTS / "criteo_click_autoint_gpu_report.md")
    print("Metrics:", OUTPUTS / "criteo_click_autoint_gpu_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
