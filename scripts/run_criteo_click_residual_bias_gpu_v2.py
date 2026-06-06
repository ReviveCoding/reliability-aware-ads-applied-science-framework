import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn

# Reuse the tested v1 loader/model/metric utilities.
import run_criteo_click_residual_bias_gpu as base


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
REPORTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)


def fit_lr_oof_and_final(df, numeric_cols, categorical_cols, target_col, train_idx, val_idx, test_idx, n_folds=3):
    features = numeric_cols + categorical_cols
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    test = df.iloc[test_idx]

    oof_train_prob = np.zeros(len(train_idx), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    start = time.time()
    for fold, (tr_rel, hold_rel) in enumerate(skf.split(train[features], train[target_col]), start=1):
        print(f"Training LR OOF fold {fold}/{n_folds}...")
        pipe = base.Pipeline([
            ("pre", base.ColumnTransformer(
                transformers=[
                    ("num", base.Pipeline([
                        ("impute", base.SimpleImputer(strategy="median")),
                        ("scale", base.StandardScaler(with_mean=False)),
                    ]), numeric_cols),
                    ("cat", base.safe_ohe(), categorical_cols),
                ],
                sparse_threshold=0.3,
            )),
            ("model", base.LogisticRegression(max_iter=300, solver="saga")),
        ])
        pipe.fit(train.iloc[tr_rel][features], train.iloc[tr_rel][target_col])
        oof_train_prob[hold_rel] = pipe.predict_proba(train.iloc[hold_rel][features])[:, 1]

    final_pipe = base.Pipeline([
        ("pre", base.ColumnTransformer(
            transformers=[
                ("num", base.Pipeline([
                    ("impute", base.SimpleImputer(strategy="median")),
                    ("scale", base.StandardScaler(with_mean=False)),
                ]), numeric_cols),
                ("cat", base.safe_ohe(), categorical_cols),
            ],
            sparse_threshold=0.3,
        )),
        ("model", base.LogisticRegression(max_iter=300, solver="saga")),
    ])
    final_pipe.fit(train[features], train[target_col])

    train_prob_final = final_pipe.predict_proba(train[features])[:, 1]
    val_prob = final_pipe.predict_proba(val[features])[:, 1]
    test_prob = final_pipe.predict_proba(test[features])[:, 1]

    return {
        "elapsed_seconds": time.time() - start,
        "oof_train_prob": oof_train_prob,
        "train_prob_final": train_prob_final,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "oof_train_metrics": base.model_metrics(train[target_col], oof_train_prob),
        "train_final_metrics": base.model_metrics(train[target_col], train_prob_final),
        "val_metrics": base.model_metrics(val[target_col], val_prob),
        "test_metrics": base.model_metrics(test[target_col], test_prob),
    }


def clipped_delta(raw_delta, delta_clip):
    return delta_clip * np.tanh(raw_delta / max(delta_clip, 1e-6))


def train_safe_residual(config, x_num, x_cat, y, base_logit_all, train_idx, val_idx, test_idx, device):
    pin_memory = device.type == "cuda"
    batch_size = int(config["batch_size"])
    train_loader = base.make_loader(x_num, x_cat, y, base_logit_all, train_idx, batch_size, True, pin_memory)
    val_loader = base.make_loader(x_num, x_cat, y, base_logit_all, val_idx, batch_size * 2, False, pin_memory)
    test_loader = base.make_loader(x_num, x_cat, y, base_logit_all, test_idx, batch_size * 2, False, pin_memory)

    model = base.ResidualDeepFM(
        num_numeric=x_num.shape[1],
        num_categorical=x_cat.shape[1],
        hash_size=int(config["hash_size"]),
        emb_dim=int(config["emb_dim"]),
        hidden=tuple(config["hidden"]),
        dropout=float(config["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(config["epochs"]))
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.get("amp", True)))

    delta_l2 = float(config["delta_l2"])
    delta_clip = float(config["delta_clip"])
    best_state, best_score, best_epoch, bad = None, -1e9, 0, 0
    history = []
    start = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x_num_b, x_cat_b, y_b, base_b in train_loader:
            x_num_b = x_num_b.to(device, non_blocking=True)
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            base_b = base_b.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and config.get("amp", True))):
                raw_delta = model(x_num_b, x_cat_b)
                delta = delta_clip * torch.tanh(raw_delta / max(delta_clip, 1e-6))
                logits = base_b + delta
                loss = loss_fn(logits, y_b) + delta_l2 * torch.mean(delta * delta)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(y_b)
            n_seen += len(y_b)

        sched.step()

        raw_val_delta, val_base, val_y = base.predict_delta(model, val_loader, device)
        val_delta = clipped_delta(raw_val_delta, delta_clip)
        val_prob = base.sigmoid_np(val_base + val_delta)
        val_m = base.model_metrics(val_y, val_prob)
        score = (
            (val_m.get("roc_auc") or 0.0)
            + 0.50 * (val_m.get("pr_auc") or 0.0)
            - 0.20 * val_m["log_loss"]
            - 0.10 * val_m["ece"]
            - 0.005 * float(np.mean(np.abs(val_delta)))
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "mean_abs_delta_val": float(np.mean(np.abs(val_delta))),
            **{f"val_{k}": v for k, v in val_m.items()},
            "val_selection_score": float(score),
        }
        history.append(row)
        print(f"[{config['name']}] epoch={epoch} train_loss={row['train_loss']:.6f} val_auc={val_m.get('roc_auc'):.6f} val_pr={val_m.get('pr_auc'):.6f} val_logloss={val_m['log_loss']:.6f} mean_abs_delta={row['mean_abs_delta_val']:.6f}")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(config["patience"]):
            print(f"[{config['name']}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    raw_val_delta, val_base, val_y = base.predict_delta(model, val_loader, device)
    raw_test_delta, test_base, test_y = base.predict_delta(model, test_loader, device)

    return {
        "config": config,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "elapsed_seconds": time.time() - start,
        "val_delta": clipped_delta(raw_val_delta, delta_clip),
        "val_base": val_base,
        "val_y": val_y,
        "test_delta": clipped_delta(raw_test_delta, delta_clip),
        "test_base": test_base,
        "test_y": test_y,
    }


def fit_temperature_from_logits(logits, labels, device):
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    log_temp = nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=60)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.2, 5.0)
        loss = loss_fn(logits_t / temp, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(torch.exp(log_temp).detach().cpu().numpy()[0], 0.2, 5.0))


def sweep_safe_lambda(val_y, test_y, val_base, test_base, val_delta, test_delta, lr_val_m, device):
    candidates = []
    for lam in np.linspace(0.0, 0.35, 36):
        val_logits = val_base + lam * val_delta
        test_logits = test_base + lam * test_delta

        temp = fit_temperature_from_logits(val_logits, val_y, device)
        val_raw = base.model_metrics(val_y, base.sigmoid_np(val_logits))
        test_raw = base.model_metrics(test_y, base.sigmoid_np(test_logits))
        val_temp = base.model_metrics(val_y, base.sigmoid_np(val_logits / temp))
        test_temp = base.model_metrics(test_y, base.sigmoid_np(test_logits / temp))

        use_temp = (
            val_temp["log_loss"] <= val_raw["log_loss"]
            and (val_temp.get("pr_auc") or 0.0) >= (val_raw.get("pr_auc") or 0.0) - 1e-12
        )
        val_m = val_temp if use_temp else val_raw
        test_m = test_temp if use_temp else test_raw

        violations = 0
        if (val_m.get("pr_auc") or 0.0) < (lr_val_m.get("pr_auc") or 0.0) - 0.0002:
            violations += 1
        if val_m["log_loss"] > lr_val_m["log_loss"] * 1.001:
            violations += 1
        if (val_m.get("roc_auc") or 0.0) < (lr_val_m.get("roc_auc") or 0.0) - 0.0002:
            violations += 1

        score = (
            (val_m.get("roc_auc") or 0.0)
            + 0.50 * (val_m.get("pr_auc") or 0.0)
            - 0.20 * val_m["log_loss"]
            - 0.10 * val_m["ece"]
            - 10.0 * violations
        )
        candidates.append({
            "lambda": float(lam),
            "temperature": float(temp),
            "used_temperature": bool(use_temp),
            "constraint_violations": int(violations),
            "val_metrics": val_m,
            "test_metrics": test_m,
            "val_raw_metrics": val_raw,
            "test_raw_metrics": test_raw,
            "val_temperature_metrics": val_temp,
            "test_temperature_metrics": test_temp,
            "validation_selection_score": float(score),
        })

    constrained = [c for c in candidates if c["constraint_violations"] == 0]
    return max(constrained if constrained else candidates, key=lambda x: x["validation_selection_score"]), candidates


def load_deepfm_v2_champion():
    path = OUTPUTS / "criteo_click_deepctr_gpu_v2_metrics.json"
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("selected_lr_deep_ensemble", {}).get("test_metrics")


def compare_metric_count(a, b):
    if not b:
        return None
    return sum([
        (a.get("roc_auc") or 0.0) >= (b.get("roc_auc") or 0.0),
        (a.get("pr_auc") or 0.0) >= (b.get("pr_auc") or 0.0),
        a.get("log_loss", 999.0) <= b.get("log_loss", 999.0),
        a.get("brier", 999.0) <= b.get("brier", 999.0),
        a.get("ece", 999.0) <= b.get("ece", 999.0),
    ])


def default_configs(hash_size, epochs):
    return [
        {
            "name": "safe_residual_tiny",
            "hash_size": hash_size,
            "emb_dim": 8,
            "hidden": [128, 64],
            "dropout": 0.15,
            "lr": 4e-4,
            "weight_decay": 5e-5,
            "delta_l2": 0.05,
            "delta_clip": 0.20,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 4,
            "amp": True,
        },
        {
            "name": "safe_residual_small",
            "hash_size": hash_size,
            "emb_dim": 12,
            "hidden": [192, 96],
            "dropout": 0.20,
            "lr": 3e-4,
            "weight_decay": 5e-5,
            "delta_l2": 0.08,
            "delta_clip": 0.15,
            "batch_size": 4096,
            "epochs": epochs,
            "patience": 4,
            "amp": True,
        },
    ]


def main():
    data_dir = os.environ.get("CRITEO_CLICK_DIR")
    if not data_dir:
        raise SystemExit("Set CRITEO_CLICK_DIR first.")

    max_rows = int(os.environ.get("RESBIAS_V2_MAX_ROWS", os.environ.get("RESBIAS_MAX_ROWS", "250000")))
    max_files = int(os.environ.get("RESBIAS_V2_MAX_FILES", "6"))
    epochs = int(os.environ.get("RESBIAS_V2_EPOCHS", "14"))
    hash_size = int(os.environ.get("RESBIAS_V2_HASH_SIZE", os.environ.get("RESBIAS_HASH_SIZE", "120000")))
    oof_folds = int(os.environ.get("RESBIAS_V2_OOF_FOLDS", "3"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Torch CUDA build:", torch.version.cuda)

    raw, source_files = base.load_criteo_click_sample(data_dir, max_rows=max_rows, max_files=max_files)
    target_col = base.detect_target(raw)
    raw.columns = [str(c).strip() for c in raw.columns]
    numeric_cols, categorical_cols = base.detect_features(raw, target_col)
    df = base.preprocess_raw(raw, target_col, numeric_cols, categorical_cols)

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
    print("OOF folds:", oof_folds)

    print("Training OOF + final LR baseline...")
    lr = fit_lr_oof_and_final(df, numeric_cols, categorical_cols, target_col, train_idx, val_idx, test_idx, n_folds=oof_folds)
    print("LR test metrics:", lr["test_metrics"])

    base_logit_all = np.zeros(len(df), dtype=np.float32)
    base_logit_all[train_idx] = base.logit_np(lr["oof_train_prob"]).astype(np.float32)
    base_logit_all[val_idx] = base.logit_np(lr["val_prob"]).astype(np.float32)
    base_logit_all[test_idx] = base.logit_np(lr["test_prob"]).astype(np.float32)

    x_num = base.numeric_matrix(df, numeric_cols)
    x_num = base.standardize_numeric(x_num, train_idx)
    x_cat = base.hash_categorical_matrix(df, categorical_cols, hash_size=hash_size)

    runs = []
    for config in default_configs(hash_size, epochs):
        print(f"Training residual-bias v2 config: {config['name']}")
        runs.append(train_safe_residual(config, x_num, x_cat, y, base_logit_all, train_idx, val_idx, test_idx, device))

    selected_run = max(runs, key=lambda r: r["best_val_score"])
    selected_candidate, lambda_candidates = sweep_safe_lambda(
        val_y=selected_run["val_y"],
        test_y=selected_run["test_y"],
        val_base=selected_run["val_base"],
        test_base=selected_run["test_base"],
        val_delta=selected_run["val_delta"],
        test_delta=selected_run["test_delta"],
        lr_val_m=lr["val_metrics"],
        device=device,
    )

    lr_test = lr["test_metrics"]
    res_test = selected_candidate["test_metrics"]
    deepfm_v2 = load_deepfm_v2_champion()
    deepfm_count = compare_metric_count(res_test, deepfm_v2)

    lr_checks = {
        "residual_auc_not_worse_than_lr": (res_test.get("roc_auc") or 0.0) >= (lr_test.get("roc_auc") or 0.0) - 0.0005,
        "residual_pr_auc_not_worse_than_lr": (res_test.get("pr_auc") or 0.0) >= (lr_test.get("pr_auc") or 0.0) - 0.0005,
        "residual_log_loss_not_worse_than_lr": res_test["log_loss"] <= lr_test["log_loss"] * 1.002,
        "residual_ece_improves_lr": res_test["ece"] <= lr_test["ece"] * 0.90,
        "gpu_residual_component_trained": device.type == "cuda" and selected_run["best_epoch"] >= 1,
    }
    lr_status = "PASS" if all(lr_checks.values()) else "ITERATE"

    champion_checks = None
    champion_status = "NOT_AVAILABLE"
    if deepfm_v2:
        champion_checks = {
            "matches_or_beats_deepfm_v2_on_at_least_3_of_5_metrics": deepfm_count >= 3,
            "log_loss_not_worse_than_deepfm_v2": res_test["log_loss"] <= deepfm_v2.get("log_loss", 999.0) * 1.001,
            "ece_not_worse_than_deepfm_v2": res_test["ece"] <= deepfm_v2.get("ece", 999.0) * 1.05,
        }
        champion_status = "PASS" if all(champion_checks.values()) else "ITERATE"

    result = {
        "dataset": "Criteo Click Logs local public-data sample",
        "method": "Safe OOF LR-offset GPU residual logit bias correction",
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
        "oof_folds": oof_folds,
        "same_split_lr_baseline": {
            "elapsed_seconds": lr["elapsed_seconds"],
            "oof_train_metrics": lr["oof_train_metrics"],
            "train_final_metrics": lr["train_final_metrics"],
            "val_metrics": lr["val_metrics"],
            "test_metrics": lr["test_metrics"],
        },
        "residual_runs": [
            {
                "config": r["config"],
                "best_epoch": r["best_epoch"],
                "best_val_score": r["best_val_score"],
                "elapsed_seconds": r["elapsed_seconds"],
                "final_history_row": r["history"][-1] if r["history"] else {},
            }
            for r in runs
        ],
        "selected_residual_run": {
            "config": selected_run["config"],
            "best_epoch": selected_run["best_epoch"],
            "best_val_score": selected_run["best_val_score"],
            "mean_abs_val_delta": float(np.mean(np.abs(selected_run["val_delta"]))),
            "mean_abs_test_delta": float(np.mean(np.abs(selected_run["test_delta"]))),
        },
        "selected_lambda_temperature_candidate": selected_candidate,
        "deepfm_v2_champion_test_metrics": deepfm_v2,
        "deepfm_v2_metric_match_or_beat_count": deepfm_count,
        "release_decision": {
            "lr_gate_status": lr_status,
            "lr_gate_checks": lr_checks,
            "champion_upgrade_status": champion_status,
            "champion_upgrade_checks": champion_checks,
            "overall_status": (
                "PASS_CHAMPION_UPGRADE" if champion_status == "PASS"
                else "PASS_LR_BASELINE" if lr_status == "PASS"
                else "ITERATE"
            ),
            "reason": (
                "Safe residual-bias correction passed both LR and current-champion upgrade checks."
                if champion_status == "PASS"
                else "Safe residual-bias correction passed the LR baseline gate but did not beat the current DeepFM v2 champion."
                if lr_status == "PASS"
                else "Safe residual-bias correction trained successfully but did not pass the LR baseline gate."
            ),
        },
    }

    (OUTPUTS / "criteo_click_residual_bias_gpu_v2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    history_rows = []
    for r in runs:
        for h in r["history"]:
            history_rows.append({"config": r["config"]["name"], **h})
    pd.DataFrame(history_rows).to_csv(OUTPUTS / "criteo_click_residual_bias_gpu_v2_training_history.csv", index=False)

    pd.DataFrame(lambda_candidates).to_json(
        OUTPUTS / "criteo_click_residual_bias_gpu_v2_lambda_candidates.json",
        orient="records",
        indent=2,
    )

    report = f"""# GPU Residual-Bias CTR v2 Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Method

This v2 runner makes the residual correction safer than v1:

- Uses out-of-fold LR logits for residual training to reduce in-sample baseline leakage.
- Clips residual logits with `delta_clip`.
- Uses stronger residual L2 penalties.
- Restricts lambda selection to a conservative range.
- Applies validation constraints for PR-AUC, ROC-AUC, and log loss before choosing a correction.

Formula:

`corrected_logit = LR_baseline_logit + lambda * clipped_neural_residual_bias`

## Same-split comparison

| Metric | LR baseline | Safe residual-bias v2 | DeepFM v2 champion |
|---|---:|---:|---:|
| ROC-AUC | {lr_test.get("roc_auc")} | {res_test.get("roc_auc")} | {deepfm_v2.get("roc_auc") if deepfm_v2 else None} |
| PR-AUC | {lr_test.get("pr_auc")} | {res_test.get("pr_auc")} | {deepfm_v2.get("pr_auc") if deepfm_v2 else None} |
| Log loss | {lr_test.get("log_loss")} | {res_test.get("log_loss")} | {deepfm_v2.get("log_loss") if deepfm_v2 else None} |
| Brier | {lr_test.get("brier")} | {res_test.get("brier")} | {deepfm_v2.get("brier") if deepfm_v2 else None} |
| ECE | {lr_test.get("ece")} | {res_test.get("ece")} | {deepfm_v2.get("ece") if deepfm_v2 else None} |

## Selected correction

- Lambda: {selected_candidate["lambda"]}
- Temperature: {selected_candidate["temperature"]}
- Used temperature: {selected_candidate["used_temperature"]}
- Constraint violations: {selected_candidate["constraint_violations"]}
- Mean absolute validation delta: {result["selected_residual_run"]["mean_abs_val_delta"]:.6f}
- Mean absolute test delta: {result["selected_residual_run"]["mean_abs_test_delta"]:.6f}

## Release decision

- LR gate status: **{lr_status}**
- LR gate checks: {lr_checks}
- Current champion upgrade status: **{champion_status}**
- Champion checks: {champion_checks}
- Overall status: **{result["release_decision"]["overall_status"]}**
- Reason: {result["release_decision"]["reason"]}
"""
    (REPORTS / "criteo_click_residual_bias_gpu_v2_report.md").write_text(report, encoding="utf-8")

    print("GPU residual-bias CTR v2 validation finished.")
    print("Report:", REPORTS / "criteo_click_residual_bias_gpu_v2_report.md")
    print("Metrics:", OUTPUTS / "criteo_click_residual_bias_gpu_v2_metrics.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
