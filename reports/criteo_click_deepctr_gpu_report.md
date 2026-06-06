# GPU DeepCTR Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- PyTorch: `2.12.0+cu126`
- Torch CUDA build: `12.6`

## Data summary

- Rows: 250000
- Train / validation / test rows: 148750 / 26250 / 75000
- Target column: `label`
- Positive rate: 0.031232
- Numeric features: 13
- Categorical features: 26
- Hash size per categorical feature: 60000

## Baseline vs GPU DeepCTR

| Metric | Logistic Regression baseline | DeepCTR raw | DeepCTR temperature-calibrated |
|---|---:|---:|---:|
| ROC-AUC | 0.7086663943114613 | 0.6901132086852437 | 0.6901132086852437 |
| PR-AUC | 0.08164290987648792 | 0.07213083119259982 | 0.07213083119259982 |
| Log loss | 0.13044985183494257 | 0.13861193297854493 | 0.1323254055453186 |
| Brier | 0.029595548945100286 | 0.03021507578123573 | 0.029764835227142874 |
| ECE | 0.0015713278297538127 | 0.022128958263768716 | 0.00014414195539219854 |

## Release decision

- Status: **ITERATE**
- Checks: {'deep_auc_beats_lr_by_0_002': False, 'deep_pr_auc_not_worse_than_lr': False, 'deep_log_loss_not_worse_than_lr': False, 'calibration_improves_raw_deep_ece': True}
- Reason: GPU DeepCTR model trained successfully but did not pass all LR comparison/calibration promotion checks.

## Interpretation

The GPU DeepCTR model uses categorical embeddings for the 26 Criteo categorical features and dense numeric inputs for the 13 integer features. Temperature scaling is applied as a post-training calibration step. Promotion requires both ranking quality and calibration discipline rather than GPU training alone.
